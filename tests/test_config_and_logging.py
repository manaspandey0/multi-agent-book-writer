import io
import logging

import pytest
from pydantic import ValidationError

from config import ConfigurationError, Settings
from utils.logging_config import (
    REDACTED,
    RedactingFormatter,
    attach_run_log_file,
    configure_logging,
    detach_run_log_file,
    get_logger,
    log_event,
    redact_secrets,
    register_secret_for_redaction,
)

FAKE_KEY = "gsk_" + "A1b2C3d4" * 4


class TestSettings:
    def test_defaults_without_env_file(self):
        settings = Settings(_env_file=None)
        assert settings.llm_provider == "groq"
        assert settings.groq_api_key is None
        assert settings.max_revision_rounds == 2

    def test_environment_variables_are_read(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "MOCK")
        monkeypatch.setenv("MAX_REVISION_ROUNDS", "1")
        settings = Settings(_env_file=None)
        assert settings.is_mock_mode
        assert settings.max_revision_rounds == 1

    def test_env_file_is_read(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(f"GROQ_API_KEY={FAKE_KEY}\nGROQ_MODEL=some-model\n")
        settings = Settings(_env_file=env_file)
        assert settings.groq_api_key.get_secret_value() == FAKE_KEY
        assert settings.groq_model == "some-model"

    @pytest.mark.parametrize("value", ["", "   ", "your-groq-api-key-here"])
    def test_blank_or_placeholder_key_counts_as_missing(self, value):
        assert Settings(_env_file=None, groq_api_key=value).groq_api_key is None

    def test_more_than_two_revision_rounds_is_rejected(self):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, max_revision_rounds=3)

    def test_live_mode_without_credentials_gives_actionable_error(self):
        with pytest.raises(ConfigurationError) as error:
            Settings(_env_file=None).require_live_provider_settings()
        message = str(error.value)
        assert "GROQ_API_KEY" in message
        assert "GROQ_MODEL" in message
        assert "LLM_PROVIDER=mock" in message

    def test_mock_mode_needs_no_credentials(self, mock_settings):
        mock_settings.require_live_provider_settings()
        assert mock_settings.active_model_name == "mock-deterministic"

    def test_secret_never_appears_in_safe_summary_or_repr(self):
        settings = Settings(_env_file=None, groq_api_key=FAKE_KEY, groq_model="m")
        assert FAKE_KEY not in str(settings.describe_safely())
        assert FAKE_KEY not in repr(settings)
        assert FAKE_KEY not in settings.model_dump_json()
        assert settings.describe_safely()["groq_api_key_configured"] is True


class TestLogging:
    def test_groq_shaped_keys_are_redacted(self):
        assert FAKE_KEY not in redact_secrets(f"calling api with {FAKE_KEY}")

    def test_bearer_tokens_and_key_assignments_are_redacted(self):
        text = redact_secrets("Authorization: Bearer abcdef123456 api_key='zyxw98765432'")
        assert "abcdef123456" not in text
        assert "zyxw98765432" not in text
        assert text.count(REDACTED) == 2

    def test_registered_secret_of_any_shape_is_redacted(self):
        register_secret_for_redaction("plain-secret-value-42")
        assert "plain-secret-value-42" not in redact_secrets("value=plain-secret-value-42")

    def test_log_event_renders_key_value_pairs(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(RedactingFormatter("%(message)s"))
        logger = get_logger("test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            log_event(logger, "chapter_drafted", chapter=1, words=742, title="Two words")
        finally:
            logger.removeHandler(handler)
        assert stream.getvalue().strip() == (
            'event=chapter_drafted chapter=1 words=742 title="Two words"'
        )

    def test_configure_logging_does_not_duplicate_handlers(self):
        app_logger = configure_logging("INFO")
        handler_count = len(app_logger.handlers)
        configure_logging("INFO")
        assert len(app_logger.handlers) == handler_count

    def test_run_log_file_receives_redacted_messages_and_tracebacks(self, tmp_path):
        configure_logging("INFO")
        log_path = tmp_path / "run" / "run.log"
        handler = attach_run_log_file(log_path)
        try:
            logger = get_logger("test")
            log_event(logger, "provider_call", key=FAKE_KEY)
            try:
                raise RuntimeError(f"request failed for key {FAKE_KEY}")
            except RuntimeError:
                logger.exception("provider error")
        finally:
            detach_run_log_file(handler)
        content = log_path.read_text(encoding="utf-8")
        assert "event=provider_call" in content
        assert "Traceback" in content
        assert FAKE_KEY not in content
