"""Startup tests for the Streamlit app and the CLI, plus an import check of every module."""

import importlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from main import run_cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(PROJECT_ROOT / "app.py")
MODULES = [
    "config", "main",
    "agents.prompts", "agents.planner_agent", "agents.researcher_agent", "agents.writer_agent",
    "agents.editor_agent", "agents.fact_checker_agent",
    "models.book_models", "models.research_models",
    "providers.llm_provider", "providers.groq_provider", "providers.mock_provider",
    "tools.web_search", "tools.source_fetcher", "tools.citation_validator", "tools.fixture_research",
    "workflow.book_workflow", "workflow.revision_manager", "workflow.run_artifacts",
    "utils.logging_config",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_every_module_imports(module_name):
    importlib.import_module(module_name)


@pytest.fixture
def offline_app_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("GROQ_MODEL", "")
    return tmp_path


class TestStreamlitApp:
    def test_app_starts_without_credentials(self, offline_app_env):
        app = AppTest.from_file(APP_PATH, default_timeout=60).run()
        assert not app.exception
        assert app.title[0].value == "INK & EVIDENCE — Multi-Agent Book Studio"
        assert [tab.label for tab in app.tabs] == ["Brief", "Workflow", "Research & quality", "Book"]
        assert any("Groq not configured" in warning.value for warning in app.sidebar.warning)

    def test_offline_demo_run_completes_and_offers_downloads(self, offline_app_env):
        app = AppTest.from_file(APP_PATH, default_timeout=60).run()
        app.radio[0].set_value("Offline demo (mock provider + fictional fixtures)")
        app.button[0].click().run()
        assert not app.exception
        assert app.session_state["run_active"] is False
        assert app.session_state["last_result"].publication_status == "published"
        assert any("NOT REAL RESEARCH" in w.value for w in app.warning)
        assert not any("Live run:" in info.value for info in app.info)
        assert len([b for b in app.get("download_button") if not (b.key or "").startswith("dl_")]) == 3
        assert any(metric.value == "3 / 3" for metric in app.metric)

    def test_live_mode_without_key_shows_setup_instructions(self, offline_app_env):
        app = AppTest.from_file(APP_PATH, default_timeout=60).run()
        app.radio[0].set_value("Live research (Groq + web sources)")
        app.button[0].click().run()
        assert not app.exception
        assert app.session_state["last_result"] is None
        assert any("GROQ_API_KEY" in error.value for error in app.error)

    def test_invalid_brief_is_rejected_before_running(self, offline_app_env):
        app = AppTest.from_file(APP_PATH, default_timeout=60).run()
        app.number_input[1].set_value(900)
        app.number_input[2].set_value(600)
        app.button[0].click().run()
        assert app.session_state["run_active"] is False
        assert any("Please fix the brief" in error.value for error in app.error)


class TestCli:
    def test_mock_run_publishes_and_exits_zero(self, offline_app_env, capsys):
        assert run_cli(["--mock", "--chapters", "2"]) == 0
        output = capsys.readouterr().out
        assert "PUBLISHED" in output and "Mock mode" in output

    def test_missing_credentials_exit_one_with_instructions(self, offline_app_env, monkeypatch, capsys):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        assert run_cli([]) == 1
        assert "GROQ_API_KEY" in capsys.readouterr().err

    def test_invalid_brief_exits_one(self, offline_app_env, capsys):
        assert run_cli(["--mock", "--min-words", "900", "--max-words", "600"]) == 1
        assert "min_words_per_chapter" in capsys.readouterr().err
