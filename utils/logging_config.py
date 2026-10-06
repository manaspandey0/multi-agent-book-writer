"""Application logging: structured ``key=value`` events with secret redaction.

Redaction happens in the formatter, after the message is fully rendered, so a
secret is removed no matter whether it arrived in the message, an argument,
an exception traceback or an event field.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

APP_LOGGER_NAME = "ink_evidence"
REDACTED = "[REDACTED]"

SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"gsk_[A-Za-z0-9]{8,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[^\s\"',]{8,}"),
)

_known_secret_values: set[str] = set()


def register_secret_for_redaction(secret: str | None) -> None:
    """Redact this exact value from all future log output."""
    if secret and len(secret) >= 8:
        _known_secret_values.add(secret)


def redact_secrets(text: str) -> str:
    for secret in _known_secret_values:
        text = text.replace(secret, REDACTED)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(
            lambda match: (match.group(1) if match.groups() else "") + REDACTED, text
        )
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(super().format(record))


def _format_field_value(value: Any) -> str:
    text = str(value)
    return f'"{text}"' if (" " in text or not text) else text


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log a named event with fields, e.g. ``event=chapter_drafted chapter=1 words=742``."""
    rendered_fields = " ".join(f"{key}={_format_field_value(val)}" for key, val in fields.items())
    logger.log(level, f"event={event} {rendered_fields}".rstrip())


def get_logger(module_name: str) -> logging.Logger:
    """Return a child of the application logger, e.g. ``ink_evidence.workflow``."""
    return logging.getLogger(f"{APP_LOGGER_NAME}.{module_name}")


def _build_formatter() -> RedactingFormatter:
    return RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure console logging once; safe to call again (e.g. on Streamlit reruns)."""
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(level)
    app_logger.propagate = False
    if not any(getattr(h, "is_console_handler", False) for h in app_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(_build_formatter())
        console_handler.is_console_handler = True  # type: ignore[attr-defined]
        app_logger.addHandler(console_handler)
    return app_logger


def attach_run_log_file(log_path: Path) -> logging.Handler:
    """Start copying application logs to ``log_path``. Detach with ``detach_run_log_file``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(_build_formatter())
    logging.getLogger(APP_LOGGER_NAME).addHandler(file_handler)
    return file_handler


def detach_run_log_file(handler: logging.Handler) -> None:
    logging.getLogger(APP_LOGGER_NAME).removeHandler(handler)
    handler.close()
