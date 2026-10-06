"""Groq implementation of ``LLMProvider`` using the official ``groq`` SDK.

The SDK's own retries are disabled (``max_retries=0``) so that every retry is
decided, bounded and logged here.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

import groq

from config import GROQ_MODELS_URL, Settings
from providers.llm_provider import (
    AuthenticationFailedError,
    LLMProvider,
    MalformedOutputError,
    ModelUnavailableError,
    NetworkBlockedError,
    PermanentProviderError,
    QuotaExhaustedError,
    StructuredRequest,
    TransientProviderError,
)
from utils.logging_config import get_logger, log_event, register_secret_for_redaction

logger = get_logger("providers.groq")

BASE_BACKOFF_SECONDS = 2.0
CHARS_PER_TOKEN_ESTIMATE = 4.0  # compact JSON prompts measured at about 4.2 chars/token
TOKEN_SAFETY_MARGIN = 300
MIN_USEFUL_COMPLETION_TOKENS = 1500
TRY_AGAIN_PATTERN = re.compile(
    r"try again in\s+(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>[\d.]+)s)?",
    re.IGNORECASE,
)
DAILY_LIMIT_MARKERS = ("per day", "(tpd)", "(rpd)")


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        settings.require_live_provider_settings()
        assert settings.groq_api_key is not None and settings.groq_model is not None
        api_key = settings.groq_api_key.get_secret_value()
        register_secret_for_redaction(api_key)
        self.model_name = settings.groq_model
        self._settings = settings
        self._sleep = sleep
        self._client = client or groq.Groq(
            api_key=api_key, timeout=settings.llm_timeout_seconds, max_retries=0
        )

    def _request_json(
        self,
        request: StructuredRequest,
        response_schema: dict[str, Any],
        repair_feedback: str | None,
    ) -> str:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": _render_user_prompt(request, response_schema)},
        ]
        if repair_feedback:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was rejected: "
                        f"{repair_feedback}\nReturn a corrected JSON object only."
                    ),
                }
            )

        completion_budget = self._completion_budget(request, messages)
        attempts_allowed = self._settings.llm_max_retries + 1
        for attempt in range(1, attempts_allowed + 1):
            started = time.monotonic()
            try:
                completion = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=self._settings.llm_temperature,
                    timeout=self._settings.llm_timeout_seconds,
                    max_completion_tokens=completion_budget,
                    **self._reasoning_kwargs(),
                )
            except groq.AuthenticationError as error:
                raise AuthenticationFailedError(
                    "Groq rejected the API key (HTTP 401). Check GROQ_API_KEY in your .env "
                    "file or create a new key at https://console.groq.com/keys."
                ) from error
            except groq.PermissionDeniedError as error:
                raise _permission_denied_error(error) from error
            except groq.NotFoundError as error:
                raise ModelUnavailableError(
                    f"Groq does not recognise model '{self.model_name}'. Choose a current "
                    f"model from {GROQ_MODELS_URL} and update GROQ_MODEL."
                ) from error
            except groq.RateLimitError as error:
                wait_seconds = self._rate_limit_wait_or_raise(error)
                self._wait_before_retry(request, attempt, attempts_allowed, wait_seconds, error)
                continue
            except groq.BadRequestError as error:
                if _is_json_generation_failure(error):
                    failed = _failed_generation(error)
                    log_event(
                        logger, "llm_json_validate_failed", agent=request.agent_name,
                        model=self.model_name, failed_generation_chars=len(failed),
                        max_completion_tokens=self._settings.llm_max_completion_tokens,
                        likely_cause=(
                            "output truncated or empty (reasoning consumed the token budget)"
                            if not failed else "model emitted malformed JSON"
                        ),
                    )
                    raise MalformedOutputError("Groq reported that the JSON output was invalid")
                raise PermanentProviderError(
                    f"Groq rejected the request (HTTP 400): {_error_message(error)}. If the "
                    f"model was retired or does not support JSON mode, choose another from "
                    f"{GROQ_MODELS_URL}."
                ) from error
            except (groq.APITimeoutError, groq.APIConnectionError, groq.InternalServerError) as error:
                backoff = BASE_BACKOFF_SECONDS * 2 ** (attempt - 1)
                self._wait_before_retry(request, attempt, attempts_allowed, backoff, error)
                continue
            except groq.APIStatusError as error:
                if error.status_code == 413:
                    raise QuotaExhaustedError(
                        "Groq refused the request as too large for the per-minute token limit "
                        "(prompt plus LLM_MAX_COMPLETION_TOKENS). Lower LLM_MAX_COMPLETION_TOKENS, "
                        f"use fewer sources per chapter, or choose a model with a larger limit. "
                        f"Groq said: {_error_message(error)}"
                    ) from error
                raise PermanentProviderError(
                    f"Groq returned HTTP {error.status_code}: {_error_message(error)}"
                ) from error

            log_event(
                logger,
                "llm_call_succeeded",
                agent=request.agent_name,
                model=self.model_name,
                attempt=attempt,
                seconds=round(time.monotonic() - started, 2),
                prompt_tokens=getattr(completion.usage, "prompt_tokens", None),
                completion_tokens=getattr(completion.usage, "completion_tokens", None),
            )
            choice = completion.choices[0]
            content = choice.message.content
            if getattr(choice, "finish_reason", None) == "length":
                log_event(
                    logger, "llm_output_truncated", agent=request.agent_name,
                    model=self.model_name, chars=len(content or ""),
                    max_completion_tokens=self._settings.llm_max_completion_tokens,
                )
                raise MalformedOutputError(
                    "Groq stopped at the completion token limit before finishing the JSON "
                    "(raise LLM_MAX_COMPLETION_TOKENS)"
                )
            if not content:
                raise MalformedOutputError("Groq returned an empty message")
            return content

        raise AssertionError("unreachable: retry loop always returns or raises")

    def _completion_budget(self, request: StructuredRequest, messages: list[dict]) -> int:
        """Largest completion budget that keeps prompt + budget inside the per-minute limit."""
        budget = self._settings.llm_max_completion_tokens
        limit = self._settings.llm_tokens_per_minute_limit
        if limit is None:
            return budget
        prompt_chars = sum(len(message["content"]) for message in messages)
        estimated_prompt = int(prompt_chars / CHARS_PER_TOKEN_ESTIMATE)
        available = limit - estimated_prompt - TOKEN_SAFETY_MARGIN
        if available < MIN_USEFUL_COMPLETION_TOKENS:
            raise QuotaExhaustedError(
                f"The {request.agent_name} request is too large for the {limit} tokens-per-minute "
                f"limit (about {estimated_prompt} prompt tokens, leaving {max(available, 0)} for "
                "the reply). Use fewer sources per chapter (MAX_SOURCES_PER_CHAPTER), a shorter "
                "word range, or a model/tier with a higher limit."
            )
        return min(budget, available)

    def _reasoning_kwargs(self) -> dict[str, Any]:
        effort = self._settings.groq_reasoning_effort
        # reasoning_effort is only accepted by reasoning models; others reject it.
        if effort and "gpt-oss" in self.model_name.lower():
            return {"reasoning_effort": effort}
        return {}

    def _rate_limit_wait_or_raise(self, error: groq.RateLimitError) -> float:
        message = _error_message(error)
        wait_seconds = _requested_wait_seconds(error, message)
        is_daily_limit = any(marker in message.lower() for marker in DAILY_LIMIT_MARKERS)
        if is_daily_limit or wait_seconds > self._settings.llm_max_rate_limit_wait_seconds:
            wait_text = f"about {wait_seconds:.0f} seconds" if wait_seconds else "an unknown time"
            raise QuotaExhaustedError(
                f"Groq rate limit or quota reached; Groq asks to wait {wait_text}. "
                f"{'This is a daily limit. ' if is_daily_limit else ''}Wait and rerun, choose "
                "a model with more free-tier headroom, or use LLM_PROVIDER=mock for an "
                f"offline demo. Groq said: {message}"
            ) from error
        return max(wait_seconds, BASE_BACKOFF_SECONDS)

    def _wait_before_retry(
        self,
        request: StructuredRequest,
        attempt: int,
        attempts_allowed: int,
        wait_seconds: float,
        error: Exception,
    ) -> None:
        if attempt >= attempts_allowed:
            raise TransientProviderError(
                f"Groq call for the {request.agent_name} agent failed after {attempts_allowed} "
                f"attempts ({type(error).__name__}). Check your connection and Groq's status, "
                "then rerun."
            ) from error
        log_event(
            logger,
            "llm_call_retrying",
            agent=request.agent_name,
            attempt=attempt,
            error_type=type(error).__name__,
            wait_seconds=round(wait_seconds, 1),
        )
        self._sleep(wait_seconds)


def _render_user_prompt(request: StructuredRequest, response_schema: dict[str, Any]) -> str:
    return (
        f"{request.task_instructions}\n\n"
        f"INPUT (JSON):\n{json.dumps(request.input_payload, ensure_ascii=False, separators=(",", ":"))}\n\n"
        "Respond with a single JSON object that conforms to this JSON Schema. "
        "Do not add commentary or Markdown fences.\n"
        f"{json.dumps(response_schema, ensure_ascii=False)}"
    )


def _permission_denied_error(error: groq.PermissionDeniedError) -> PermanentProviderError:
    """Separate a 403 from Groq (JSON error body) from one sent by a proxy or firewall."""
    if _is_groq_error_body(error.body):
        return AuthenticationFailedError(
            "Groq denied access (HTTP 403). The key may lack access to this model or the "
            f"account may be restricted. Groq said: {_error_message(error)}"
        )
    deny_reason = error.response.headers.get("x-deny-reason") if error.response else None
    return NetworkBlockedError(
        "The request to api.groq.com was blocked before reaching Groq (HTTP 403 from a "
        f"proxy or firewall{f', reason: {deny_reason}' if deny_reason else ''}). Allow "
        "outbound HTTPS to api.groq.com on this network, or use LLM_PROVIDER=mock offline."
    )


def _is_groq_error_body(body: object) -> bool:
    return isinstance(body, dict) and isinstance(body.get("error"), dict)


def _error_message(error: groq.APIStatusError) -> str:
    body = error.body
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
    return str(error)


def _is_json_generation_failure(error: groq.BadRequestError) -> bool:
    body = error.body
    inner = body.get("error", body) if isinstance(body, dict) else {}
    code = inner.get("code") if isinstance(inner, dict) else None
    return code == "json_validate_failed" or "json_validate_failed" in str(error)


def _failed_generation(error: groq.BadRequestError) -> str:
    body = error.body
    inner = body.get("error", body) if isinstance(body, dict) else {}
    value = inner.get("failed_generation") if isinstance(inner, dict) else None
    return value if isinstance(value, str) else ""


def _requested_wait_seconds(error: groq.APIStatusError, message: str) -> float:
    header_value = error.response.headers.get("retry-after") if error.response else None
    if header_value:
        try:
            return float(header_value)
        except ValueError:
            pass
    match = TRY_AGAIN_PATTERN.search(message)
    if match and any(match.groupdict().values()):
        return (
            int(match["hours"] or 0) * 3600
            + int(match["minutes"] or 0) * 60
            + float(match["seconds"] or 0)
        )
    return 0.0
