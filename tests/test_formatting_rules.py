"""Chapter bodies must be continuous prose: the prompts forbid lists, the validator rejects
them, and revision feedback quotes the offending lines."""

import pytest

from agents.prompts import EDITOR_TASK, WRITER_TASK
from models.book_models import BookBrief, Severity, WrittenChapter
from models.research_models import SourceRecord, SourceRegistry, RetrievalStatus
from datetime import UTC, datetime
from workflow.revision_manager import build_revision_feedback
from tools.citation_validator import validate_chapter_draft

TAKEAWAY = "Takeaway: Confirm every payment before you hand over the goods."
PROSE = "Link your bank account first [S01]; then print the QR code and place it at the counter [S01]. "
BRIEF = BookBrief(title="UPI", audience="owners", min_words_per_chapter=100, max_words_per_chapter=900)


def registry():
    reg = SourceRegistry()
    from models.research_models import EvidenceExcerpt
    reg.add_source(SourceRecord(
        source_id="S01", title="T", publisher="P", url="https://example.org/a",
        retrieval_status=RetrievalStatus.RETRIEVED, retrieved_at=datetime.now(UTC), content_hash="h",
        excerpts=[EvidenceExcerpt(excerpt_id="S01-E1", source_id="S01", text="x" * 30, char_start=0, char_end=30)],
    ))
    return reg


def codes(body):
    draft = WrittenChapter(title="Chapter", body=body)
    return {i.code: i for i in validate_chapter_draft(draft, BRIEF, registry(), ["S01"])}


FILLER = (PROSE * 12).strip()


class TestPrompts:
    def test_writer_prompt_forbids_lists_with_bad_and_good_example(self):
        assert "BAD:" in WRITER_TASK and "GOOD:" in WRITER_TASK
        assert "'1.'" in WRITER_TASK and "First," in WRITER_TASK
        assert "Takeaway:" in WRITER_TASK

    def test_editor_prompt_never_asks_for_lists(self):
        assert "Never suggest bullet points, numbered lists" in EDITOR_TASK
        assert "Takeaway:" in EDITOR_TASK


class TestValidatorStillStrict:
    @pytest.mark.parametrize("marker", ["1. ", "2) ", "- ", "* ", "• "])
    def test_list_lines_are_critical(self, marker):
        found = codes(f"{FILLER}\n{marker}Link your bank account [S01].\n{TAKEAWAY}")
        assert found["bullet_list_in_prose"].severity == Severity.CRITICAL

    def test_connected_prose_with_sequence_words_and_numbers_passes_format_rules(self):
        body = (
            f"{FILLER}\n\nFirst, link your account, and next, print the QR code. "
            f"In 2024 UPI handled 131 billion transactions [S01].\n{TAKEAWAY}"
        )
        found = codes(body)
        assert "bullet_list_in_prose" not in found and "uncited_numeric_claim" not in found
        assert "takeaway_missing" not in found

    def test_takeaway_is_required_last_and_unformatted(self):
        assert "takeaway_missing" in codes(FILLER)
        assert "takeaway_formatting" in codes(f"{FILLER}\n**Takeaway:** Do the right thing every day.")
        assert "takeaway_misplaced" in codes(f"{FILLER}\n{TAKEAWAY}\nOne more sentence follows [S01].")

    def test_reference_list_in_draft_is_rejected(self):
        assert "references_in_draft" in codes(f"{FILLER}\n## References\n{TAKEAWAY}")


class TestFeedback:
    def test_feedback_quotes_violation_and_asks_for_prose(self):
        found = codes(f"{FILLER}\n1. Link your bank account [S01].\n2. Print the QR code [S01].\n{TAKEAWAY}")
        feedback = build_revision_feedback([found["bullet_list_in_prose"]], None)[0]
        assert "1. Link your bank account" in feedback and "2. Print the QR code" in feedback
        assert "connected prose" in feedback and "do not just delete" in feedback


class TestUncitedFigureFeedback:
    def test_feedback_offers_cite_or_reword_never_invent(self):
        found = codes(f"{FILLER}\nThe RBI guidelines of March 2020 apply to aggregators.\n{TAKEAWAY}")
        message = found["uncited_numeric_claim"].message
        assert found["uncited_numeric_claim"].severity == Severity.CRITICAL
        assert "'2020'" in message and "rewrite the sentence" in message and "never invent" in message


def test_editor_is_told_not_to_count_words():
    assert "Do not judge or count words" in EDITOR_TASK
