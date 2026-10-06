from datetime import UTC, datetime

import pytest

from models.book_models import BookBrief, ClaimCheck, ClaimVerdict, FactCheckReport, Severity, WrittenChapter
from models.research_models import (
    EvidenceExcerpt,
    RetrievalStatus,
    SourceRecord,
    SourceRegistry,
)
from tools.citation_validator import (
    CitationRenderingError,
    count_prose_words,
    extract_cited_source_ids,
    find_uncited_numeric_sentences,
    issues_from_fact_check,
    render_chapter_citations,
    validate_chapter_draft,
    validate_rendered_chapter,
)

BRIEF = BookBrief(title="Pay Me on UPI", audience="owners", min_words_per_chapter=600, max_words_per_chapter=900)
TAKEAWAY = "Takeaway: Start with one QR code and check every payment carefully."
FILLER_WORD = "steady"


def make_source(source_id: str, url: str, status=RetrievalStatus.RETRIEVED) -> SourceRecord:
    text = f"Evidence excerpt text for {source_id}: UPI launched in 2016, with enough length to be valid."
    retrieved = status == RetrievalStatus.RETRIEVED
    return SourceRecord(
        source_id=source_id, title=f"Title {source_id}", publisher="Publisher", url=url,
        retrieved_at=datetime.now(UTC) if retrieved else None,
        retrieval_status=status, content_hash="hash" if retrieved else None,
        failure_reason=None if retrieved else "HTTP 403",
        excerpts=(
            [EvidenceExcerpt(excerpt_id=f"{source_id}-E1", source_id=source_id, text=text,
                             char_start=0, char_end=len(text))]
            if retrieved else []
        ),
    )


@pytest.fixture
def registry() -> SourceRegistry:
    reg = SourceRegistry()
    reg.add_source(make_source("S01", "https://example.org/one"))
    reg.add_source(make_source("S02", "https://example.org/two"))
    reg.add_source(make_source("S03", "https://example.org/three", RetrievalStatus.INACCESSIBLE))
    return reg


def body_with_words(total_words: int, cited_sentence: str = "UPI launched in 2016 [S01].") -> str:
    """Build a body whose prose word count is exactly ``total_words``."""
    fixed = f"{cited_sentence}\n\n{TAKEAWAY}"
    padding = total_words - count_prose_words(fixed)
    assert padding >= 0
    return f"{cited_sentence} {' '.join([FILLER_WORD] * padding)}\n\n{TAKEAWAY}"


def codes(issues) -> set[str]:
    return {issue.code for issue in issues}


class TestWordCount:
    def test_citation_tokens_and_heading_marks_are_not_words(self):
        assert count_prose_words("## Heading here\n\nOne two [S01][S02] three.") == 5

    @pytest.mark.parametrize("words", [600, 750, 900])
    def test_in_range_passes(self, registry, words):
        draft = WrittenChapter(title="Chapter", body=body_with_words(words))
        assert "word_count_out_of_range" not in codes(validate_chapter_draft(draft, BRIEF, registry, ["S01"]))

    @pytest.mark.parametrize("words", [599, 901])
    def test_boundaries_fail(self, registry, words):
        draft = WrittenChapter(title="Chapter", body=body_with_words(words))
        issues = validate_chapter_draft(draft, BRIEF, registry, ["S01"])
        assert "word_count_out_of_range" in codes(issues)
        assert all(i.severity == Severity.CRITICAL for i in issues if i.code == "word_count_out_of_range")


class TestStructure:
    def test_clean_draft_has_no_issues(self, registry):
        draft = WrittenChapter(title="Chapter", body=body_with_words(700))
        assert validate_chapter_draft(draft, BRIEF, registry, ["S01"]) == []

    def test_missing_takeaway(self, registry):
        body = body_with_words(700).replace(TAKEAWAY, "In short, start small and check payments.")
        issues = validate_chapter_draft(WrittenChapter(title="Chapter title", body=body), BRIEF, registry, ["S01"])
        assert "takeaway_missing" in codes(issues)

    def test_takeaway_not_last(self, registry):
        body = body_with_words(700) + "\n\nOne more paragraph after the takeaway."
        issues = validate_chapter_draft(WrittenChapter(title="Chapter title", body=body), BRIEF, registry, ["S01"])
        assert "takeaway_misplaced" in codes(issues)

    def test_bold_takeaway_is_rejected(self, registry):
        body = body_with_words(700).replace("Takeaway:", "**Takeaway:**")
        issues = validate_chapter_draft(WrittenChapter(title="Chapter title", body=body), BRIEF, registry, ["S01"])
        assert "takeaway_formatting" in codes(issues)

    @pytest.mark.parametrize("bullet", ["- a point", "* a point", "• a point", "1. a point", "2) a point"])
    def test_bullet_lines_are_rejected(self, registry, bullet):
        body = body_with_words(700).replace(TAKEAWAY, f"{bullet}\n\n{TAKEAWAY}")
        issues = validate_chapter_draft(WrittenChapter(title="Chapter title", body=body), BRIEF, registry, ["S01"])
        assert "bullet_list_in_prose" in codes(issues)

    def test_urls_numeric_citations_and_reference_lists_are_rejected(self, registry):
        body = body_with_words(700).replace(
            TAKEAWAY, f"See https://example.org for more [1].\n\n## References\n\n{TAKEAWAY}"
        )
        issues = codes(validate_chapter_draft(WrittenChapter(title="Chapter title", body=body), BRIEF, registry, ["S01"]))
        assert {"url_in_prose", "numbered_citation_in_draft", "references_in_draft"} <= issues


class TestCitationIds:
    def test_unknown_id_is_critical(self, registry):
        body = body_with_words(700, "UPI launched in 2016 [S09].")
        issues = validate_chapter_draft(WrittenChapter(title="Chapter title", body=body), BRIEF, registry, ["S01"])
        assert "unknown_citation_id" in codes(issues)

    def test_inaccessible_source_cannot_be_cited(self, registry):
        body = body_with_words(700, "UPI launched in 2016 [S03].")
        issues = validate_chapter_draft(WrittenChapter(title="Chapter title", body=body), BRIEF, registry, ["S03"])
        assert "cites_unusable_source" in codes(issues)

    def test_source_outside_chapter_evidence_is_a_warning(self, registry):
        body = body_with_words(700, "UPI launched in 2016 [S02].")
        issues = validate_chapter_draft(WrittenChapter(title="Chapter title", body=body), BRIEF, registry, ["S01"])
        outside = [i for i in issues if i.code == "cites_source_outside_chapter_evidence"]
        assert outside and outside[0].severity == Severity.WARNING

    def test_chapter_without_citations_is_critical(self, registry):
        body = body_with_words(700, "UPI changed how shops are paid.")
        assert "no_citations" in codes(
            validate_chapter_draft(WrittenChapter(title="Chapter title", body=body), BRIEF, registry, ["S01"])
        )

    def test_extracts_ids_in_first_appearance_order(self):
        assert extract_cited_source_ids("A [S02]. B [S01][S02]. C [S10].") == ["S02", "S01", "S10"]


class TestUncitedNumbers:
    def test_number_without_citation_is_flagged_as_heuristic(self, registry):
        body = body_with_words(700).replace(TAKEAWAY, f"About 40 shops joined last year.\n\n{TAKEAWAY}")
        issues = [i for i in validate_chapter_draft(WrittenChapter(title="Chapter title", body=body), BRIEF, registry, ["S01"])
                  if i.code == "uncited_numeric_claim"]
        assert issues and issues[0].is_heuristic and issues[0].severity == Severity.CRITICAL

    def test_citation_after_full_stop_counts_for_that_sentence(self):
        assert find_uncited_numeric_sentences("UPI launched in 2016. [S01] It grew quickly.") == []

    def test_sentences_without_digits_are_ignored(self):
        assert find_uncited_numeric_sentences("Customers like paying by phone.") == []


class TestRendering:
    def test_ids_become_chapter_local_numbers_with_matching_references(self, registry):
        draft = WrittenChapter(title="Chapter title", body="First [S02]. Second [S01][S02].\n\n" + TAKEAWAY)
        rendered = render_chapter_citations(2, draft, registry)
        assert rendered.citation_numbers == {"S02": 1, "S01": 2}
        assert rendered.prose_markdown.startswith("First [1]. Second [2][1].")
        assert rendered.references_markdown.startswith("[1] Publisher. *Title S02*. <https://example.org/two>")
        assert "[2] Publisher. *Title S01*" in rendered.references_markdown
        assert rendered.markdown.startswith("## Chapter 2: Chapter title")
        assert validate_rendered_chapter(rendered, registry) == []

    def test_takeaway_is_immediately_before_references(self, registry):
        rendered = render_chapter_citations(
            1, WrittenChapter(title="Chapter title", body=body_with_words(700)), registry
        )
        before_references = rendered.markdown.split("### References")[0].strip().splitlines()
        assert before_references[-1].startswith("Takeaway:")

    def test_unknown_ids_cannot_be_rendered(self, registry):
        with pytest.raises(CitationRenderingError, match="S42"):
            render_chapter_citations(1, WrittenChapter(title="Chapter title", body="Claim [S42]. " * 10), registry)

    def test_numbering_mismatch_is_detected(self, registry):
        rendered = render_chapter_citations(
            1, WrittenChapter(title="Chapter title", body="Claim [S01]. Other [S02]. " * 5), registry
        )
        tampered = rendered.model_copy(update={"prose_markdown": "Claim [1]. Other [3]."})
        found = codes(validate_rendered_chapter(tampered, registry))
        assert {"missing_reference_entry", "reference_never_cited"} <= found

    def test_duplicate_pages_are_detected(self):
        reg = SourceRegistry()
        reg.add_source(make_source("S01", "https://example.org/page"))
        reg.sources["S02"] = make_source("S02", "https://EXAMPLE.org/page/")
        rendered = render_chapter_citations(
            1, WrittenChapter(title="Chapter title", body="Claim [S01]. Again [S02]. " * 5), reg
        )
        assert "duplicate_reference" in codes(validate_rendered_chapter(rendered, reg))

    def test_fixture_sources_are_labelled_in_references(self):
        reg = SourceRegistry()
        text = "Fixture excerpt text that is long enough to be valid."
        reg.add_source(SourceRecord(
            source_id="S01", title="Fixture", publisher="Test suite", url="https://fixtures.invalid/a",
            retrieval_status=RetrievalStatus.TEST_FIXTURE,
            excerpts=[EvidenceExcerpt(excerpt_id="S01-E1", source_id="S01", text=text, char_start=0, char_end=len(text))],
        ))
        rendered = render_chapter_citations(1, WrittenChapter(title="Chapter title", body="Claim [S01]. " * 10), reg)
        assert "TEST FIXTURE: not a real source" in rendered.references_markdown


class TestFactCheckIssues:
    def claim(self, verdict, evidence=("S01-E1",)) -> ClaimCheck:
        return ClaimCheck(
            claim_text="UPI launched in 2016.", cited_source_ids=["S01"], verdict=verdict,
            evidence_excerpt_ids=list(evidence) if verdict in {ClaimVerdict.SUPPORTED, ClaimVerdict.CONTRADICTED} else [],
            explanation="Reason given.",
        )

    @pytest.mark.parametrize("verdict", [ClaimVerdict.UNSUPPORTED, ClaimVerdict.CONTRADICTED])
    def test_unsupported_and_contradicted_are_critical(self, verdict):
        issues = issues_from_fact_check(FactCheckReport(claims=[self.claim(verdict)]), ["S01"])
        assert issues[0].code == f"claim_{verdict.value}"
        assert issues[0].severity == Severity.CRITICAL

    def test_uncertain_is_a_warning(self):
        issues = issues_from_fact_check(FactCheckReport(claims=[self.claim(ClaimVerdict.UNCERTAIN)]), ["S01"])
        assert [(i.code, i.severity) for i in issues] == [("claim_uncertain", Severity.WARNING)]

    def test_empty_fact_check_is_not_treated_as_a_pass(self):
        issues = issues_from_fact_check(FactCheckReport(claims=[]), ["S01"])
        assert "fact_check_empty" in codes(issues)

    def test_cited_source_never_checked_is_reported(self):
        issues = issues_from_fact_check(FactCheckReport(claims=[self.claim(ClaimVerdict.SUPPORTED)]), ["S01", "S02"])
        assert "citation_not_fact_checked" in codes(issues)
