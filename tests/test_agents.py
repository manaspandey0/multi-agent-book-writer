"""Each agent tested on its own, with the mock provider standing in for the LLM."""

from datetime import UTC, datetime

import pytest

from agents.editor_agent import EditorAgent
from agents.fact_checker_agent import FactCheckerAgent, ground_claim_verdicts
from agents.planner_agent import PlannerAgent, PlanningError
from agents.writer_agent import WriterAgent
from models.book_models import (
    BookBrief,
    ClaimCheck,
    ClaimVerdict,
    FactCheckReport,
    Severity,
    ValidationIssue,
    WrittenChapter,
)
from models.research_models import EvidenceExcerpt, RetrievalStatus, SourceRecord, SourceRegistry
from providers.mock_provider import MockProvider, _build_outline

BRIEF = BookBrief(title="Pay Me on UPI", audience="First-time small-business owners")
EXCERPT_TEXT = "TEST FIXTURE: in this fixture the stall records 40 digital payments in a week."


def fixture_registry() -> SourceRegistry:
    registry = SourceRegistry()
    for number in (1, 2):
        source_id = f"S0{number}"
        registry.add_source(
            SourceRecord(
                source_id=source_id, title=f"Fixture {number}", publisher="Test suite",
                url=f"https://fixtures.invalid/{number}", retrieval_status=RetrievalStatus.TEST_FIXTURE,
                retrieved_at=datetime.now(UTC),
                excerpts=[EvidenceExcerpt(excerpt_id=f"{source_id}-E1", source_id=source_id,
                                          text=EXCERPT_TEXT, char_start=0, char_end=len(EXCERPT_TEXT))],
            )
        )
    return registry


class TestPlanner:
    def test_outline_uses_brief_title_and_count(self):
        outline = PlannerAgent(MockProvider()).create_book_outline(BRIEF)
        assert outline.book_title == BRIEF.title
        assert len(outline.chapters) == 3
        assert all(chapter.search_queries for chapter in outline.chapters)

    def test_wrong_chapter_count_is_rejected(self):
        def two_chapters(payload):
            return _build_outline({"brief": {**payload["brief"], "chapter_count": 2}})

        with pytest.raises(PlanningError, match="2 chapters"):
            PlannerAgent(MockProvider({"planner": two_chapters})).create_book_outline(BRIEF)


class TestWriter:
    def test_payload_contains_only_supplied_evidence_and_feedback(self):
        provider = MockProvider()
        outline = PlannerAgent(provider).create_book_outline(BRIEF)
        registry = fixture_registry()
        sources = [registry.get_source("S01")]
        previous = WrittenChapter(title="Old", body="Old body text that needs a revision. " * 5)
        WriterAgent(provider).write_chapter(
            BRIEF, outline, outline.chapters[0], sources, ["Add a takeaway."], previous
        )
        payload = provider.calls_for("writer")[-1]
        assert payload["allowed_source_ids"] == ["S01"]
        assert [s["source_id"] for s in payload["evidence"]] == ["S01"]
        assert payload["revision_feedback"] == ["Add a takeaway."]
        assert payload["previous_draft"]["title"] == "Old"
        assert "url" not in payload["evidence"][0]

    def test_refuses_to_write_without_evidence(self):
        provider = MockProvider()
        outline = PlannerAgent(provider).create_book_outline(BRIEF)
        with pytest.raises(ValueError, match="evidence"):
            WriterAgent(provider).write_chapter(BRIEF, outline, outline.chapters[0], [])


class TestEditor:
    def test_deterministic_findings_reach_the_editor(self):
        provider = MockProvider()
        outline = PlannerAgent(provider).create_book_outline(BRIEF)
        issue = ValidationIssue(code="takeaway_missing", severity=Severity.CRITICAL, message="Add a takeaway.")
        review = EditorAgent(provider).review_chapter(
            BRIEF, outline.chapters[0], WrittenChapter(title="Chapter title", body="Some body text here. " * 5), [issue]
        )
        assert not review.approved
        assert provider.calls_for("editor")[-1]["deterministic_findings"] == ["Add a takeaway."]


class TestFactCheckGrounding:
    def claim(self, **overrides) -> ClaimCheck:
        data = {
            "claim_text": "The fixture stall records 40 digital payments in a week.",
            "cited_source_ids": ["S01"], "verdict": "supported",
            "evidence_excerpt_ids": ["S01-E1"], "explanation": "Excerpt states it.",
        }
        data.update(overrides)
        return ClaimCheck.model_validate(data)

    def ground(self, claim) -> ClaimCheck:
        return ground_claim_verdicts(FactCheckReport(claims=[claim]), fixture_registry()).claims[0]

    def test_well_grounded_support_is_kept(self):
        assert self.ground(self.claim()).verdict == ClaimVerdict.SUPPORTED

    def test_hallucinated_excerpt_becomes_unsupported(self):
        result = self.ground(self.claim(evidence_excerpt_ids=["S01-E9"]))
        assert result.verdict == ClaimVerdict.UNSUPPORTED
        assert "does not exist" in result.explanation

    def test_unregistered_source_becomes_unsupported(self):
        result = self.ground(self.claim(cited_source_ids=["S07"], evidence_excerpt_ids=["S07-E1"]))
        assert result.verdict == ClaimVerdict.UNSUPPORTED

    def test_evidence_from_uncited_source_becomes_uncertain(self):
        result = self.ground(self.claim(evidence_excerpt_ids=["S02-E1"]))
        assert result.verdict == ClaimVerdict.UNCERTAIN and result.needs_human_review

    def test_number_missing_from_evidence_becomes_uncertain(self):
        result = self.ground(self.claim(claim_text="The fixture stall records 400 digital payments."))
        assert result.verdict == ClaimVerdict.UNCERTAIN
        assert "400" in result.explanation

    def test_agent_sends_only_cited_sources_and_grounds_the_result(self):
        provider = MockProvider()
        draft = WrittenChapter(
            title="Chapter title", body=f"{EXCERPT_TEXT.rstrip('.')} [S01].\n\nTakeaway: Check every payment you receive."
        )
        report = FactCheckerAgent(provider).verify_chapter_citations(1, draft, fixture_registry())
        assert [s["source_id"] for s in provider.calls_for("fact_checker")[-1]["cited_sources"]] == ["S01"]
        assert [c.verdict for c in report.claims] == [ClaimVerdict.SUPPORTED]
