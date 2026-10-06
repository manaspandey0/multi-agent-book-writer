"""Shared fixtures. Every test runs without network access or real credentials."""

import pytest

from config import Settings

APP_ENV_VARS = (
    "LLM_PROVIDER",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "LOG_LEVEL",
    "MAX_REVISION_ROUNDS",
    "OUTPUTS_DIR",
)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent a developer's real environment from leaking into tests."""
    for name in APP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def mock_settings(tmp_path) -> Settings:
    return Settings(_env_file=None, llm_provider="mock", outputs_dir=tmp_path / "outputs")
