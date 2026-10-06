"""Workflow tests: full offline runs with scripted agent behaviour."""

import json

import pytest

from agents.researcher_agent import ResearcherAgent
from models.book_models import ChapterStatus, EvidenceBasis, Severity, ValidationIssue, WorkflowStage
from providers.llm_provider import QuotaExhaustedError
from providers.mock_provider import MockProvider, _build_chapter, _build_fact_check
from tools.fixture_research import FixturePageFetcher, FixtureSearchAdapter
from workflow.book_workflow import BookWorkflow, create_book_workflow
from workflow.revision_manager import NextStep, RevisionManager

BRIEF = {"title": "Pay Me on UPI: How Digital Payments Changed Small Business in India",
         "audience": "First-time small-business owners in India"}


def build_workflow(settings, provider=None, events=None):
    researcher = ResearcherAgent(FixtureSearchAdapter(), FixturePageFetcher(), settings)
    return BookWorkflow(
        settings, provider or MockProvider(), researcher, uses_test_fixtures=True,
        progress_callback=(events.append if events is not None else None),
    )


def short_chapter(payload):
    return {"title": payload["chapter"]["title"],
            "body": "Too short to pass [S01].\n\nTakeaway: This draft will never be long enough."}


class TestEndToEndMockRun:
    def test_publishes_three_chapters_and_saves_all_artifacts(self, mock_settings):
        result = create_book_workflow(mock_settings).generate_book(BRIEF)
        assert result.publication_status == "published"
        assert result.state.stage == WorkflowStage.COMPLETED
        assert [c.status for c in result.state.chapters] == [ChapterStatus.PASSED] * 3
        for chapter in result.state.chapters:
            assert 600 <= chapter.word_count <= 900
            assert chapter.fact_check is not None and chapter.fact_check.claims
        assert set(result.saved_files) == {"book", "sources", "report", "log"}
        assert all(path.exists() for path in result.saved_files.values())
        assert "event=run_finished" in result.saved_files["log"].read_text()

    def test_book_citations_match_registered_sources(self, mock_settings):
        result = create_book_workflow(mock_settings).generate_book(BRIEF)
        for chapter in result.state.chapters:
            for source_id in chapter.source_ids:
                assert result.registry.get_source(source_id).can_support_claims
            prose, references = chapter.rendered_markdown.split("### References")
            assert "[S0" not in prose
            reference_lines = [line for line in references.splitlines() if line.startswith("[")]
            assert len(reference_lines) == len(chapter.source_ids)
            assert prose.strip().splitlines()[-1].startswith("Takeaway:")

    def test_mock_output_is_labelled_and_never_called_real_research(self, mock_settings):
        result = create_book_workflow(mock_settings).generate_book(BRIEF)
        report = json.loads(result.saved_files["report"].read_text())
        assert report["evidence_basis"] == EvidenceBasis.TEST_FIXTURES.value
        assert report["verification_scope"].startswith("fixture_consistency_only")
        assert report["sources"]["real_world_evidence_sources"] == 0
        assert "not a researched book" in result.book_markdown
        assert "TEST FIXTURE: not a real source" in result.book_markdown

    def test_progress_events_follow_real_stage_order(self, mock_settings):
        events = []
        build_workflow(mock_settings, events=events).generate_book(BRIEF)
        stages = [event.stage for event in events]
        assert stages[:3] == [WorkflowStage.VALIDATING_BRIEF, WorkflowStage.PLANNING, WorkflowStage.RESEARCHING]
        first_chapter = stages[2:stages.index(WorkflowStage.RESEARCHING, 3)]
        assert first_chapter == [
            WorkflowStage.RESEARCHING, WorkflowStage.WRITING, WorkflowStage.VALIDATING,
            WorkflowStage.EDITING, WorkflowStage.FACT_CHECKING,
        ]
        assert stages[-1] == WorkflowStage.COMPLETED

    def test_each_run_gets_its_own_directory(self, mock_settings):
        workflow = create_book_workflow(mock_settings)
        first, second = workflow.generate_book(BRIEF), workflow.generate_book(BRIEF)
        assert first.run_dir != second.run_dir
        assert first.saved_files["book"].exists() and second.saved_files["book"].exists()


class TestRevisionLoop:
    def test_always_failing_writer_stops_at_the_limit_and_blocks_publication(self, mock_settings):
        provider = MockProvider({"writer": short_chapter})
        result = build_workflow(mock_settings, provider).generate_book({**BRIEF, "chapter_count": 1})
        chapter = result.state.chapters[0]
        assert chapter.status == ChapterStatus.FAILED
        assert chapter.revision_count == 2
        assert len(provider.calls_for("writer")) == 3
        assert result.publication_status == "blocked"
        assert result.book_markdown is None and "book" not in result.saved_files
        assert "word_count_out_of_range" in result.failure_report

    def test_feedback_reaches_the_writer_and_a_fixed_draft_passes(self, mock_settings):
        drafts = iter([short_chapter, _build_chapter])
        provider = MockProvider({"writer": lambda payload: next(drafts)(payload)})
        result = build_workflow(mock_settings, provider).generate_book({**BRIEF, "chapter_count": 1})
        chapter = result.state.chapters[0]
        assert chapter.status == ChapterStatus.PASSED and chapter.revision_count == 1
        feedback = provider.calls_for("writer")[1]["revision_feedback"]
        assert any("words" in item for item in feedback)
        assert provider.calls_for("writer")[1]["previous_draft"] is not None

    def test_zero_revision_rounds_means_one_draft_only(self, mock_settings):
        settings = mock_settings.model_copy(update={"max_revision_rounds": 0})
        provider = MockProvider({"writer": short_chapter})
        build_workflow(settings, provider).generate_book({**BRIEF, "chapter_count": 1})
        assert len(provider.calls_for("writer")) == 1

    def test_revision_manager_never_allows_a_third_revision(self):
        critical = [ValidationIssue(code="x", severity=Severity.CRITICAL, message="broken")]
        manager = RevisionManager(2)
        assert [manager.decide_next_step(n, critical, None).next_step for n in range(4)] == [
            NextStep.REVISE, NextStep.REVISE, NextStep.FAIL, NextStep.FAIL,
        ]
        with pytest.raises(ValueError):
            RevisionManager(3)


class TestPublicationGate:
    def test_contradicted_claim_blocks_publication(self, mock_settings):
        def contradict(payload):
            report = _build_fact_check(payload)
            report["claims"][0]["verdict"] = "contradicted"
            return report

        provider = MockProvider({"fact_checker": contradict})
        result = build_workflow(mock_settings, provider).generate_book({**BRIEF, "chapter_count": 1})
        assert result.state.chapters[0].status == ChapterStatus.FAILED
        assert result.publication_status == "blocked"
        assert "claim_contradicted" in result.failure_report
        assert len(provider.calls_for("writer")) == 3

    def test_uncertain_claim_publishes_with_warnings(self, mock_settings):
        def uncertain(payload):
            report = _build_fact_check(payload)
            report["claims"][0].update(verdict="uncertain", evidence_excerpt_ids=[])
            return report

        result = build_workflow(mock_settings, MockProvider({"fact_checker": uncertain})).generate_book(
            {**BRIEF, "chapter_count": 1}
        )
        assert result.state.chapters[0].status == ChapterStatus.PASSED_WITH_WARNINGS
        assert result.publication_status == "published"
        assert "flagged for human review" in result.book_markdown

    def test_empty_fact_check_is_not_a_pass(self, mock_settings):
        provider = MockProvider({"fact_checker": lambda payload: {"claims": []}})
        result = build_workflow(mock_settings, provider).generate_book({**BRIEF, "chapter_count": 1})
        assert result.state.chapters[0].status == ChapterStatus.FAILED
        assert "fact_check_empty" in result.failure_report

    def test_invalid_llm_output_fails_the_chapter_not_the_process(self, mock_settings):
        provider = MockProvider({"editor": lambda payload: "not json"})
        result = build_workflow(mock_settings, provider).generate_book({**BRIEF, "chapter_count": 2})
        assert [c.status for c in result.state.chapters] == [ChapterStatus.FAILED] * 2
        assert "InvalidResponseError" in result.state.chapters[0].failure_reason
        assert result.publication_status == "blocked"

    def test_quota_exhaustion_aborts_run_and_still_saves_a_report(self, mock_settings):
        def quota(payload):
            raise QuotaExhaustedError("Groq rate limit or quota reached")

        result = build_workflow(mock_settings, MockProvider({"writer": quota})).generate_book(BRIEF)
        assert result.publication_status == "failed"
        assert result.state.stage == WorkflowStage.FAILED
        assert any("QuotaExhaustedError" in error for error in result.state.errors)
        report = json.loads(result.saved_files["report"].read_text())
        assert report["publication_status"] == "failed" and report["errors"]
        assert "book" not in result.saved_files

    def test_report_marks_fact_check_as_not_run_when_structure_fails(self, mock_settings):
        result = build_workflow(mock_settings, MockProvider({"writer": short_chapter})).generate_book(
            {**BRIEF, "chapter_count": 1}
        )
        chapter_report = result.report["chapters"][0]
        assert chapter_report["fact_check_performed"] is False
        assert chapter_report["fact_check_claims"] == []


class TestPublishedArtifactStructure:
    def test_each_chapter_has_takeaway_and_its_own_numbered_references(self, mock_settings):
        result = create_book_workflow(mock_settings).generate_book(BRIEF)
        book = result.book_markdown
        assert book.count("## Chapter ") == 3 and book.count("### References") == 3
        for chapter in result.state.chapters:
            prose, references = chapter.rendered_markdown.split("### References")
            assert "[1]" in prose and references.strip().startswith("[1]")
            assert prose.strip().splitlines()[-1].startswith("Takeaway:")


class TestAuditRegressions:
    def test_drafts_are_preserved_even_when_rejected(self, mock_settings):
        class Short(MockProvider):
            def generate_structured(self, request, response_model):
                if request.agent_name == "writer":
                    return response_model.model_validate(short_chapter(request.input_payload))
                return super().generate_structured(request, response_model)

        result = build_workflow(mock_settings, provider=Short()).generate_book(BRIEF)
        drafts = sorted(p.name for p in (result.run_dir / "drafts").glob("*.md"))
        assert result.publication_status == "blocked" and "chapter1_rev0.md" in drafts
        assert "word_count_out_of_range" in (result.run_dir / "drafts" / "chapter1_rev0.md").read_text()

    def test_quota_abort_never_leaves_a_chapter_in_progress(self, mock_settings):
        class OutOfQuota(MockProvider):
            def generate_structured(self, request, response_model):
                if request.agent_name == "writer":
                    raise QuotaExhaustedError("daily limit")
                return super().generate_structured(request, response_model)

        result = build_workflow(mock_settings, provider=OutOfQuota()).generate_book(BRIEF)
        statuses = [c.status for c in result.state.chapters]
        assert result.publication_status == "failed" and result.book_markdown is None
        assert statuses[0] == ChapterStatus.FAILED and ChapterStatus.IN_PROGRESS not in statuses
        assert (result.run_dir / "failure_report.md").exists()


class TestSourceTiersAndSavedRuns:
    def test_official_vs_secondary_classification_rejects_lookalikes(self):
        from models.research_models import source_tier
        assert source_tier("https://www.rbi.org.in/x") == "official"
        assert source_tier("https://pib.gov.in/x") == "official"
        assert source_tier("https://rbi.org.in.evil.com/x") == "secondary"
        assert source_tier("https://fakerbi.org.in/x") == "secondary"
        assert source_tier("https://blog.example.com/x") == "secondary"

    def test_saved_runs_listing_reads_reports_and_flags_demo(self, mock_settings):
        from workflow.run_artifacts import list_saved_runs
        result = create_book_workflow(mock_settings).generate_book(BRIEF)
        (mock_settings.outputs_dir / "run_broken").mkdir()
        runs = {r["run_id"]: r for r in list_saved_runs(mock_settings.outputs_dir)}
        assert runs[result.run_dir.name]["status"] == "published"
        assert runs[result.run_dir.name]["evidence_basis"] == "test_fixtures"
        assert runs[result.run_dir.name]["has_book"] and runs["run_broken"]["status"] == "unreadable"
