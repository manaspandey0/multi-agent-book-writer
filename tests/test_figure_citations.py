"""Figures (circular identifiers, years, settlement cycles, percentages, amounts) need citations.

Validation stays strict, its feedback names the figure and the sentence, and a citation to
a source whose excerpts lack the figure is flagged."""

from datetime import UTC, datetime

import pytest

from agents.prompts import CITATION_RULES, WRITER_TASK
from models.book_models import BookBrief, Severity, WrittenChapter
from models.research_models import EvidenceExcerpt, RetrievalStatus, SourceRecord, SourceRegistry
from tools.citation_validator import describe_figures, find_uncited_numeric_sentences, validate_chapter_draft

BRIEF = BookBrief(title="UPI", audience="owners", min_words_per_chapter=100, max_words_per_chapter=900)
TAKEAWAY = "Takeaway: Confirm each settlement cycle with your bank before relying on it."
FILLER = ("Merchants link a bank account and display a QR code at the counter [S01]. " * 14).strip()
EXCERPT = (
    "Under circular DPSS.CO.PD.No.1164/02.14.006/2017-18 banks issue UPI handles. "
    "A study found 76.2% of respondents said UPI helped and 15.6% said sometimes. "
    "Settlement moved from T+1 to T+0 in 2021."
)


def registry(excerpt=EXCERPT):
    reg = SourceRegistry()
    reg.add_source(SourceRecord(
        source_id="S01", title="T", publisher="P", url="https://example.org/a",
        retrieval_status=RetrievalStatus.RETRIEVED, retrieved_at=datetime.now(UTC), content_hash="h",
        excerpts=[EvidenceExcerpt(excerpt_id="S01-E1", source_id="S01", text=excerpt, char_start=0, char_end=len(excerpt))],
    ))
    return reg


def issues_for(sentence, reg=None):
    body = f"{FILLER}\n{sentence}\n{TAKEAWAY}"
    draft = WrittenChapter(title="Chapter", body=body)
    return validate_chapter_draft(draft, BRIEF, reg or registry(), ["S01"])


UNCITED = {
    "circular id": ("Follow RBI circular DPSS.CO.PD.No.1164/02.14.006/2017-18 when onboarding.", "circular"),
    "year": ("The RBI Master Direction on payment security was issued in 2021.", "year"),
    "settlement": ("Confirm whether settlement is T+0, T+1 or batched.", "settlement"),
    "percentage": ("76.2% of respondents found UPI helpful.", "percentage"),
    "amount": ("Fees are capped at ₹300 per transaction.", "amount"),
}


class TestUncitedFiguresStayCritical:
    @pytest.mark.parametrize("name", UNCITED)
    def test_each_kind_is_critical_and_feedback_names_sentence_and_fix(self, name):
        sentence, kind = UNCITED[name]
        found = [i for i in issues_for(sentence) if i.code == "uncited_numeric_claim"]
        assert found and found[0].severity == Severity.CRITICAL
        message = found[0].message
        assert kind in message and sentence[:40] in message
        assert "rewrite the sentence in general terms" in message and "never invent" in message

    def test_describe_figures_lists_every_kind(self):
        text = describe_figures("In 2021 RBI said T+1 applies to 76.2% of flows.")
        assert "year" in text and "settlement cycle" in text and "percentage" in text


class TestCitedFiguresMustAppearInEvidence:
    def test_supported_figures_raise_nothing(self):
        for sentence in (
            "A study found 76.2% of respondents said UPI helped [S01].",
            "Settlement moved from T+1 to T+0 in 2021 [S01].",
        ):
            assert not [i for i in issues_for(sentence) if i.code == "figure_not_in_cited_excerpts"]

    def test_citation_to_source_lacking_the_figure_is_flagged(self):
        found = [i for i in issues_for("A study found 91.4% of shops use UPI [S01].")
                 if i.code == "figure_not_in_cited_excerpts"]
        assert found and "91.4" in found[0].message and found[0].is_heuristic

    def test_unknown_or_unusable_sources_still_critical(self):
        codes = {i.code for i in issues_for("Rates changed in 2021 [S09].")}
        assert "unknown_citation_id" in codes

    def test_adding_a_citation_does_not_hide_a_fabricated_number(self):
        reg = registry("Banks issue UPI handles to merchants after onboarding checks are done.")
        found = [i for i in issues_for("Circular 1164/2017 applies [S01].", reg)]
        assert any(i.code == "figure_not_in_cited_excerpts" for i in found)


def test_cited_numeric_sentence_is_not_reported_as_uncited():
    assert find_uncited_numeric_sentences("Settlement moved to T+0 in 2021 [S01].") == []


def test_prompts_require_evidence_for_every_figure_kind():
    for term in ("circular", "T+0", "percentages", "never recall figures from memory"):
        assert term.lower() in CITATION_RULES.lower()
    assert "rather than a circular number or year" in WRITER_TASK


class TestSpeedAndStatisticClaims:
    def test_uncited_speed_claim_with_number_is_critical(self):
        found = [i for i in issues_for("Payments settle in under 15 seconds.") if i.code == "uncited_numeric_claim"]
        assert found and found[0].severity == Severity.CRITICAL

    def test_uncited_speed_claim_without_number_is_flagged(self):
        found = [i for i in issues_for("Money arrives within seconds of the payment.") if i.code == "uncited_speed_claim"]
        assert found and found[0].severity == Severity.WARNING

    def test_cited_speed_claim_is_not_flagged(self):
        assert not [i for i in issues_for("Money arrives instantly [S01].") if i.code == "uncited_speed_claim"]

    def test_cited_statistic_missing_from_evidence_is_critical(self):
        found = [i for i in issues_for("UPI handled 91.4% of retail payments in 2026 [S01].")
                 if i.code == "figure_not_in_cited_excerpts"]
        assert found and found[0].severity == Severity.CRITICAL


def test_revision_payload_asks_for_minimal_edit_of_quoted_sentences():
    from agents.prompts import WRITER_TASK as task
    assert "MINIMAL EDIT" in task and "copy every other sentence exactly" in task
