"""Centralized application settings loaded from environment variables and ``.env``.

Precedence (highest first): explicit keyword arguments, real environment
variables, the project ``.env`` file, then the defaults below. Every other
module receives a ``Settings`` instance instead of reading the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent
GROQ_CONSOLE_URL = "https://console.groq.com/keys"
GROQ_MODELS_URL = "https://console.groq.com/docs/models"
PLACEHOLDER_PREFIX = "your-"


class ConfigurationError(RuntimeError):
    """Raised when settings are present but unusable for the requested run."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["groq", "mock"] = "groq"
    groq_api_key: SecretStr | None = None
    groq_model: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_max_rate_limit_wait_seconds: float = Field(default=65.0, ge=0, le=300)
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    # Reasoning models (gpt-oss) spend completion tokens on hidden reasoning. Without an
    # explicit budget and low effort the reply is cut off before any JSON is emitted.
    # Groq counts prompt + this budget against the per-minute token limit (8000 on the
    # free tier), so keep it as small as the longest chapter allows.
    llm_max_completion_tokens: int = Field(default=3000, ge=256, le=32000)
    # Per-minute token limit of the Groq tier/model (8000 for gpt-oss-20b on the free tier).
    # Each request's completion budget is shrunk so prompt + budget fit inside it.
    llm_tokens_per_minute_limit: int | None = Field(default=8000, ge=1000)
    groq_reasoning_effort: Literal["low", "medium", "high"] | None = "low"

    max_revision_rounds: int = Field(default=2, ge=0, le=2)

    search_results_per_query: int = Field(default=6, ge=1, le=20)
    max_sources_per_chapter: int = Field(default=6, ge=1, le=15)
    fetch_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    per_domain_delay_seconds: float = Field(default=1.5, ge=0, le=30)
    max_page_bytes: int = Field(default=3_000_000, ge=10_000)
    http_user_agent: str = (
        "InkAndEvidenceResearchBot/0.1 (educational book-research prototype)"
    )

    outputs_dir: Path = PROJECT_ROOT / "outputs"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("groq_api_key", mode="after")
    @classmethod
    def treat_blank_key_as_missing(cls, key: SecretStr | None) -> SecretStr | None:
        if key is None:
            return None
        value = key.get_secret_value().strip()
        if not value or value.startswith(PLACEHOLDER_PREFIX):
            return None
        return SecretStr(value)

    @field_validator("groq_model", mode="after")
    @classmethod
    def treat_blank_model_as_missing(cls, model: str | None) -> str | None:
        if model is None:
            return None
        model = model.strip()
        if not model or model.startswith(PLACEHOLDER_PREFIX):
            return None
        return model

    @field_validator("llm_provider", mode="before")
    @classmethod
    def lowercase_provider(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def uppercase_log_level(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @property
    def is_mock_mode(self) -> bool:
        return self.llm_provider == "mock"

    @property
    def active_model_name(self) -> str:
        if self.is_mock_mode:
            return "mock-deterministic"
        return self.groq_model or "(not configured)"

    def require_live_provider_settings(self) -> None:
        """Raise ``ConfigurationError`` with setup steps if Groq cannot be used."""
        if self.is_mock_mode:
            return
        problems: list[str] = []
        if self.groq_api_key is None:
            problems.append(
                f"GROQ_API_KEY is not set (or is still the .env.example placeholder). Create a key at {GROQ_CONSOLE_URL} and add "
                "it to your .env file."
            )
        if self.groq_model is None:
            problems.append(
                "GROQ_MODEL is not set (or is still the placeholder). Choose a model that supports JSON mode from "
                f"{GROQ_MODELS_URL} and add it to your .env file."
            )
        if problems:
            problems.append("To run without credentials, set LLM_PROVIDER=mock.")
            raise ConfigurationError(" ".join(problems))

    def describe_safely(self) -> dict[str, Any]:
        """Settings summary safe to show in the UI or write to a run report."""
        return {
            "llm_provider": self.llm_provider,
            "model": self.active_model_name,
            "groq_api_key_configured": self.groq_api_key is not None,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "llm_max_retries": self.llm_max_retries,
            "max_revision_rounds": self.max_revision_rounds,
            "max_sources_per_chapter": self.max_sources_per_chapter,
        }


def load_settings(**overrides: Any) -> Settings:
    """Build settings from the environment, applying any explicit overrides."""
    return Settings(**overrides)
