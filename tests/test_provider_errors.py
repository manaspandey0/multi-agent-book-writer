"""Provider behaviour, using real ``groq`` exception classes and a fake client.

No network call is made: the fake client raises or returns scripted values.
"""

import json
import logging
from types import SimpleNamespace

import groq
import httpx
import pytest

from config import ConfigurationError, Settings
from models.book_models import BookOutline, EditorReview
from providers.groq_provider import GroqProvider
from providers.llm_provider import (
    AuthenticationFailedError,
    InvalidResponseError,
    ModelUnavailableError,
    NetworkBlockedError,
    PermanentProviderError,
    QuotaExhaustedError,
    StructuredRequest,
    TransientProviderError,
)
from providers.mock_provider import MockProvider
from utils.logging_config import attach_run_log_file, configure_logging, detach_run_log_file

FAKE_KEY = "gsk_" + "Z9y8X7w6" * 4
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
REQUEST = StructuredRequest(
    agent_name="editor", system_prompt="system", task_instructions="Return JSON."
)
VALID_REVIEW = json.dumps({"approved": True, "summary": "Reads well."})


def http_response(status: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", GROQ_URL), headers=headers or {})


def status_error(cls, status: int, message: str, headers: dict | None = None, code: str | None = None):
    body = {"error": {"message": message, **({"code": code} if code else {})}}
    return cls(message, response=http_response(status, headers), body=body)


def completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class ScriptedClient:
    """Mimics ``client.chat.completions.create``; each call takes the next scripted item."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return completion(item)


@pytest.fixture
def groq_settings() -> Settings:
    return Settings(
        _env_file=None, llm_provider="groq", groq_api_key=FAKE_KEY, groq_model="test-model",
        llm_max_retries=2, llm_max_rate_limit_wait_seconds=30,
    )


def make_provider(settings, *script):
    sleeps: list[float] = []
    client = ScriptedClient(*script)
    provider = GroqProvider(settings, client=client, sleep=sleeps.append)
    return provider, client, sleeps


class TestGroqProvider:
    def test_requires_credentials(self):
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
            GroqProvider(Settings(_env_file=None, llm_provider="groq"))

    def test_successful_structured_call_uses_json_mode_and_configured_model(self, groq_settings):
        provider, client, _ = make_provider(groq_settings, VALID_REVIEW)
        review = provider.generate_structured(REQUEST, EditorReview)
        assert review.approved
        call = client.calls[0]
        assert call["model"] == "test-model"
        assert call["response_format"] == {"type": "json_object"}
        assert call["timeout"] == groq_settings.llm_timeout_seconds
        assert "JSON" in call["messages"][1]["content"]

    def test_authentication_error_is_permanent_and_not_retried(self, groq_settings):
        provider, client, sleeps = make_provider(
            groq_settings, status_error(groq.AuthenticationError, 401, "Invalid API Key")
        )
        with pytest.raises(AuthenticationFailedError, match="GROQ_API_KEY"):
            provider.generate_structured(REQUEST, EditorReview)
        assert len(client.calls) == 1
        assert sleeps == []

    def test_groq_403_is_an_access_problem(self, groq_settings):
        provider, _, _ = make_provider(
            groq_settings, status_error(groq.PermissionDeniedError, 403, "Organization restricted")
        )
        with pytest.raises(AuthenticationFailedError, match="Organization restricted"):
            provider.generate_structured(REQUEST, EditorReview)

    def test_proxy_403_is_reported_as_network_block_not_groq(self, groq_settings):
        proxy_block = groq.PermissionDeniedError(
            "Host not in allowlist: api.groq.com",
            response=http_response(403, {"x-deny-reason": "host_not_allowed"}),
            body="Host not in allowlist: api.groq.com",
        )
        provider, client, _ = make_provider(groq_settings, proxy_block)
        with pytest.raises(NetworkBlockedError, match="host_not_allowed") as error:
            provider.generate_structured(REQUEST, EditorReview)
        assert "Groq denied" not in str(error.value)
        assert len(client.calls) == 1

    def test_unknown_model_gives_actionable_error(self, groq_settings):
        provider, _, _ = make_provider(
            groq_settings, status_error(groq.NotFoundError, 404, "model not found")
        )
        with pytest.raises(ModelUnavailableError, match="GROQ_MODEL"):
            provider.generate_structured(REQUEST, EditorReview)

    def test_short_rate_limit_waits_then_succeeds(self, groq_settings):
        provider, client, sleeps = make_provider(
            groq_settings,
            status_error(groq.RateLimitError, 429, "Rate limit on tokens per minute", {"retry-after": "4"}),
            VALID_REVIEW,
        )
        assert provider.generate_structured(REQUEST, EditorReview).approved
        assert sleeps == [4.0]
        assert len(client.calls) == 2

    def test_daily_quota_stops_immediately(self, groq_settings):
        message = (
            "Rate limit reached for model `test-model` on tokens per day (TPD): Limit 100000. "
            "Please try again in 7m12.5s."
        )
        provider, client, sleeps = make_provider(
            groq_settings, status_error(groq.RateLimitError, 429, message)
        )
        with pytest.raises(QuotaExhaustedError, match="daily limit"):
            provider.generate_structured(REQUEST, EditorReview)
        assert len(client.calls) == 1
        assert sleeps == []

    def test_long_requested_wait_is_treated_as_quota_exhaustion(self, groq_settings):
        provider, _, sleeps = make_provider(
            groq_settings,
            status_error(groq.RateLimitError, 429, "slow down", {"retry-after": "600"}),
        )
        with pytest.raises(QuotaExhaustedError, match="600 seconds"):
            provider.generate_structured(REQUEST, EditorReview)
        assert sleeps == []

    def test_timeouts_retry_with_backoff_up_to_the_limit(self, groq_settings):
        timeout = groq.APITimeoutError(request=httpx.Request("POST", GROQ_URL))
        provider, client, sleeps = make_provider(groq_settings, timeout, timeout, timeout)
        with pytest.raises(TransientProviderError, match="3 attempts"):
            provider.generate_structured(REQUEST, EditorReview)
        assert len(client.calls) == groq_settings.llm_max_retries + 1
        assert sleeps == [2.0, 4.0]

    def test_connection_and_server_errors_recover(self, groq_settings):
        provider, _, sleeps = make_provider(
            groq_settings,
            groq.APIConnectionError(request=httpx.Request("POST", GROQ_URL)),
            status_error(groq.InternalServerError, 503, "unavailable"),
            VALID_REVIEW,
        )
        assert provider.generate_structured(REQUEST, EditorReview).approved
        assert len(sleeps) == 2

    def test_malformed_json_gets_one_repair_attempt(self, groq_settings):
        provider, client, _ = make_provider(groq_settings, "not json at all", VALID_REVIEW)
        assert provider.generate_structured(REQUEST, EditorReview).approved
        repair_message = client.calls[1]["messages"][-1]["content"]
        assert "rejected" in repair_message and "invalid JSON" in repair_message

    def test_schema_violation_twice_raises_invalid_response(self, groq_settings):
        bad = json.dumps({"approved": False, "summary": "Needs work."})
        provider, client, _ = make_provider(groq_settings, bad, bad)
        with pytest.raises(InvalidResponseError, match="EditorReview"):
            provider.generate_structured(REQUEST, EditorReview)
        assert len(client.calls) == 2

    def test_groq_json_validation_failure_is_repaired(self, groq_settings):
        provider, _, _ = make_provider(
            groq_settings,
            status_error(groq.BadRequestError, 400, "Failed to generate JSON", code="json_validate_failed"),
            VALID_REVIEW,
        )
        assert provider.generate_structured(REQUEST, EditorReview).approved

    def test_other_bad_requests_are_permanent(self, groq_settings):
        provider, _, _ = make_provider(
            groq_settings, status_error(groq.BadRequestError, 400, "model is decommissioned")
        )
        with pytest.raises(PermanentProviderError, match="decommissioned"):
            provider.generate_structured(REQUEST, EditorReview)

    def test_markdown_fenced_json_is_accepted(self, groq_settings):
        provider, _, _ = make_provider(groq_settings, f"```json\n{VALID_REVIEW}\n```")
        assert provider.generate_structured(REQUEST, EditorReview).approved

    def test_api_key_never_reaches_logs(self, groq_settings, tmp_path):
        configure_logging("DEBUG")
        handler = attach_run_log_file(tmp_path / "run.log")
        try:
            provider, _, _ = make_provider(
                groq_settings,
                status_error(groq.AuthenticationError, 401, f"Invalid API Key {FAKE_KEY}"),
            )
            with pytest.raises(AuthenticationFailedError) as error:
                provider.generate_structured(REQUEST, EditorReview)
            logging.getLogger("ink_evidence.test").exception("auth failed", exc_info=error.value)
        finally:
            detach_run_log_file(handler)
        content = (tmp_path / "run.log").read_text()
        assert "auth failed" in content
        assert FAKE_KEY not in content


class TestMockProvider:
    def test_is_labelled_as_test_provider(self):
        provider = MockProvider()
        assert provider.is_test_provider
        assert provider.name == "mock"

    def test_outline_matches_requested_chapter_count(self):
        request = StructuredRequest(
            agent_name="planner", system_prompt="s", task_instructions="t",
            input_payload={"brief": {"title": "A title", "audience": "readers", "chapter_count": 5}},
        )
        outline = MockProvider().generate_structured(request, BookOutline)
        assert [c.chapter_number for c in outline.chapters] == [1, 2, 3, 4, 5]

    def test_override_with_malformed_output_raises_after_repair(self):
        provider = MockProvider(response_overrides={"editor": lambda payload: "{broken"})
        with pytest.raises(InvalidResponseError):
            provider.generate_structured(REQUEST, EditorReview)
        assert len(provider.calls_for("editor")) == 2
        assert "repair_feedback" in provider.calls_for("editor")[1]


class TestReasoningModelTokenBudget:
    """Reasoning models spend completion tokens on hidden reasoning, so an unbounded or small
    default budget yields finish_reason=length with no JSON. The provider sets an explicit budget."""

    def settings(self, model):
        return Settings(
            _env_file=None, llm_provider="groq", groq_api_key=FAKE_KEY, groq_model=model,
            llm_max_retries=0,
        )

    def test_request_sets_token_budget_and_low_reasoning_for_gpt_oss(self):
        provider, client, _ = make_provider(self.settings("openai/gpt-oss-20b"), VALID_REVIEW)
        provider.generate_structured(REQUEST, EditorReview)
        call = client.calls[0]
        assert call["max_completion_tokens"] == 3000 and call["reasoning_effort"] == "low"

    def test_reasoning_effort_not_sent_to_other_models(self):
        provider, client, _ = make_provider(self.settings("llama-3.3-70b-versatile"), VALID_REVIEW)
        provider.generate_structured(REQUEST, EditorReview)
        assert "reasoning_effort" not in client.calls[0]

    def test_truncated_output_is_logged_and_not_accepted(self, caplog):
        truncated = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"approved": tr'), finish_reason="length")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        provider, client, _ = make_provider(self.settings("openai/gpt-oss-20b"), "x", "x")
        client._create = lambda **kw: truncated
        client.chat.completions.create = client._create
        with caplog.at_level(logging.INFO, logger="ink_evidence"), pytest.raises(InvalidResponseError):
            provider.generate_structured(REQUEST, EditorReview)
        assert "event=llm_output_truncated" in caplog.text

    def test_json_validate_failed_logs_likely_cause(self, caplog):
        error = status_error(groq.BadRequestError, 400, "Failed to validate JSON", code="json_validate_failed")
        provider, _, _ = make_provider(self.settings("openai/gpt-oss-20b"), error, error)
        with caplog.at_level(logging.INFO, logger="ink_evidence"), pytest.raises(InvalidResponseError):
            provider.generate_structured(REQUEST, EditorReview)
        assert "event=llm_json_validate_failed" in caplog.text and "token budget" in caplog.text

    def test_413_request_too_large_is_a_clear_provider_error_not_a_crash(self):
        error = status_error(groq.APIStatusError, 413, "Request too large ... Requested 9501")
        provider, _, _ = make_provider(self.settings("openai/gpt-oss-20b"), error)
        with pytest.raises(QuotaExhaustedError, match="LLM_MAX_COMPLETION_TOKENS"):
            provider.generate_structured(REQUEST, EditorReview)


class TestRateLimitPolicy:
    def limit_error(self, message, retry_after=None):
        headers = {"retry-after": str(retry_after)} if retry_after else None
        return status_error(groq.RateLimitError, 429, message, headers=headers)

    def settings(self, retries=2):
        return Settings(_env_file=None, llm_provider="groq", groq_api_key=FAKE_KEY,
                        groq_model="m", llm_max_retries=retries)

    def test_default_cap_covers_a_per_minute_window_and_server_delay_is_respected(self):
        error = self.limit_error("tokens per minute (TPM). Please try again in 37s", retry_after=37)
        provider, client, sleeps = make_provider(self.settings(), error, VALID_REVIEW)
        assert provider.generate_structured(REQUEST, EditorReview).approved
        assert sleeps == [37.0] and len(client.calls) == 2

    def test_retries_are_bounded_and_then_stop_clearly(self):
        error = self.limit_error("tokens per minute (TPM). Please try again in 5s", retry_after=5)
        provider, client, sleeps = make_provider(self.settings(retries=1), error, error, error)
        with pytest.raises(TransientProviderError):
            provider.generate_structured(REQUEST, EditorReview)
        assert len(client.calls) == 2  # never retries indefinitely

    def test_daily_quota_or_long_wait_stops_without_waiting(self):
        for message in ("tokens per day (TPD). Please try again in 20m", "TPM. Please try again in 5m"):
            provider, client, sleeps = make_provider(self.settings(), self.limit_error(message))
            with pytest.raises(QuotaExhaustedError):
                provider.generate_structured(REQUEST, EditorReview)
            assert sleeps == [] and len(client.calls) == 1


class TestTokenBudgetAgainstLimit:
    def settings(self, **kw):
        return Settings(_env_file=None, llm_provider="groq", groq_api_key=FAKE_KEY,
                        groq_model="openai/gpt-oss-20b", llm_max_retries=0, **kw)

    def big_request(self, chars):
        return StructuredRequest(agent_name="writer", system_prompt="s", task_instructions="x" * chars)

    def test_budget_shrinks_so_prompt_plus_completion_fit_the_limit(self):
        provider, client, _ = make_provider(self.settings(), VALID_REVIEW)
        provider.generate_structured(self.big_request(20000), EditorReview)  # ~5k tokens
        budget = client.calls[0]["max_completion_tokens"]
        assert 1500 <= budget < 3000 and 20000 / 4.0 + budget < 8000

    def test_prompt_too_large_stops_before_calling_groq(self):
        provider, client, _ = make_provider(self.settings(), VALID_REVIEW)
        with pytest.raises(QuotaExhaustedError, match="tokens-per-minute"):
            provider.generate_structured(self.big_request(26000), EditorReview)
        assert client.calls == []

    def test_small_prompt_keeps_configured_budget(self):
        provider, client, _ = make_provider(self.settings(), VALID_REVIEW)
        provider.generate_structured(REQUEST, EditorReview)
        assert client.calls[0]["max_completion_tokens"] == 3000
