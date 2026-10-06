"""Excerpt-level citations such as [S02-E1] are normalised to source IDs before validation.

Without normalisation the validator sees no citations and counts the excerpt digit as an
uncited number. Normalisation changes only the token's form, never which source is cited."""

from agents.writer_agent import normalize_excerpt_citations
from models.book_models import BookBrief, WrittenChapter
from models.research_models import SourceRegistry
from tools.citation_validator import (
    extract_cited_source_ids,
    find_uncited_numeric_sentences,
    validate_chapter_draft,
)

EXCERPT_STYLE = "Merchants settle quickly [S02-E2][S02-E3]. Fees are lower than cards [S01-E1]."


class TestNormalization:
    def test_excerpt_ids_become_source_ids_and_repeats_collapse(self):
        assert normalize_excerpt_citations(EXCERPT_STYLE) == (
            "Merchants settle quickly [S02]. Fees are lower than cards [S01]."
        )

    def test_plain_source_ids_and_distinct_sources_are_untouched(self):
        text = "Claim [S01][S03]. Another [S02]."
        assert normalize_excerpt_citations(text) == text

    def test_does_not_add_citations_to_uncited_sentences(self):
        assert normalize_excerpt_citations("Settlement takes 2 days.") == "Settlement takes 2 days."

    def test_validator_sees_citations_only_after_normalization(self):
        assert extract_cited_source_ids(EXCERPT_STYLE) == []
        assert find_uncited_numeric_sentences(EXCERPT_STYLE)  # the old false positive
        normalized = normalize_excerpt_citations(EXCERPT_STYLE)
        assert extract_cited_source_ids(normalized) == ["S02", "S01"]
        assert find_uncited_numeric_sentences(normalized) == []

    def test_unknown_source_still_fails_validation_after_normalization(self):
        body = normalize_excerpt_citations("A fact about UPI settlement [S09-E1]. " * 40 + "\nTakeaway: Check your settlement account daily.")
        draft = WrittenChapter(title="Chapter", body=body)
        brief = BookBrief(title="UPI", audience="owners", min_words_per_chapter=100, max_words_per_chapter=900)
        issues = validate_chapter_draft(draft, brief, SourceRegistry(), ["S01"])
        assert any(i.code.startswith("unknown") or "source" in i.code for i in issues)


class TestListMarkers:
    def test_bare_list_markers_are_not_numeric_claims_but_list_is_still_flagged(self):
        body = "Steps follow.\n1. Set up the QR code at the counter [S01].\n2. Check each payment [S01]."
        assert find_uncited_numeric_sentences(body) == []
        draft = WrittenChapter(title="Chapter", body=body + "\nTakeaway: Always confirm each payment before handing over goods.")
        brief = BookBrief(title="UPI", audience="owners", min_words_per_chapter=100, max_words_per_chapter=900)
        issues = validate_chapter_draft(draft, brief, SourceRegistry(), ["S01"])
        assert any(i.code == "bullet_list_in_prose" for i in issues)

    def test_a_real_uncited_figure_is_still_flagged(self):
        assert find_uncited_numeric_sentences("Settlement takes 2 days.")
