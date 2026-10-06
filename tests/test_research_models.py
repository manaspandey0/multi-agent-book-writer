from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from models.research_models import (
    DuplicateSourceError,
    EvidenceExcerpt,
    RetrievalStatus,
    SearchResult,
    SourceRecord,
    SourceRegistry,
    UnknownSourceError,
    canonicalize_url,
)

EXCERPT_TEXT = "TEST FIXTURE: this sentence is fixture text, not real research evidence."


def make_excerpt(source_id: str, number: int = 1) -> EvidenceExcerpt:
    return EvidenceExcerpt(
        excerpt_id=f"{source_id}-E{number}",
        source_id=source_id,
        text=EXCERPT_TEXT,
        char_start=0,
        char_end=len(EXCERPT_TEXT),
    )


def make_retrieved(source_id: str, url: str) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        title="Example page",
        publisher="Example publisher",
        url=url,
        retrieved_at=datetime.now(UTC),
        retrieval_status=RetrievalStatus.RETRIEVED,
        content_hash="abc123",
        excerpts=[make_excerpt(source_id)],
    )


def make_inaccessible(source_id: str, url: str) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        title="Unreachable page",
        publisher="Example publisher",
        url=url,
        retrieval_status=RetrievalStatus.INACCESSIBLE,
        failure_reason="HTTP 403",
    )


class TestCanonicalizeUrl:
    def test_ignores_case_fragment_trailing_slash_and_tracking(self):
        assert canonicalize_url(
            "HTTPS://Example.ORG/report/?utm_source=x&id=7#section"
        ) == canonicalize_url("https://example.org/report?id=7")

    def test_keeps_meaningful_query_parameters(self):
        assert canonicalize_url("https://example.org/a?id=1") != canonicalize_url(
            "https://example.org/a?id=2"
        )


class TestSearchResult:
    def test_relative_url_is_rejected(self):
        with pytest.raises(ValidationError, match="absolute http"):
            SearchResult(title="t", url="/page", search_query="q")


class TestSourceRecord:
    def test_retrieved_source_requires_hash_and_timestamp(self):
        with pytest.raises(ValidationError, match="content_hash"):
            SourceRecord(
                source_id="S01",
                title="t",
                publisher="p",
                url="https://example.org",
                retrieval_status=RetrievalStatus.RETRIEVED,
            )

    def test_inaccessible_source_cannot_carry_excerpts(self):
        with pytest.raises(ValidationError, match="cannot carry excerpts"):
            SourceRecord(
                source_id="S01",
                title="t",
                publisher="p",
                url="https://example.org",
                retrieval_status=RetrievalStatus.INACCESSIBLE,
                failure_reason="timeout",
                excerpts=[make_excerpt("S01")],
            )

    def test_inaccessible_source_must_explain_failure(self):
        with pytest.raises(ValidationError, match="failure_reason"):
            SourceRecord(
                source_id="S01",
                title="t",
                publisher="p",
                url="https://example.org",
                retrieval_status=RetrievalStatus.INACCESSIBLE,
            )

    def test_excerpt_from_another_source_is_rejected(self):
        with pytest.raises(ValidationError, match="belongs to S02"):
            SourceRecord(
                source_id="S01",
                title="t",
                publisher="p",
                url="https://example.org",
                retrieved_at=datetime.now(UTC),
                retrieval_status=RetrievalStatus.RETRIEVED,
                content_hash="h",
                excerpts=[make_excerpt("S02")],
            )

    def test_excerpt_location_must_be_a_forward_range(self):
        with pytest.raises(ValidationError, match="char_end"):
            EvidenceExcerpt(
                excerpt_id="S01-E1", source_id="S01", text=EXCERPT_TEXT, char_start=50, char_end=10
            )

    def test_fixture_source_supports_claims_but_is_not_real_world_evidence(self):
        fixture = SourceRecord(
            source_id="S01",
            title="Fixture",
            publisher="Test suite",
            url="https://example.invalid/fixture",
            retrieval_status=RetrievalStatus.TEST_FIXTURE,
            excerpts=[make_excerpt("S01")],
        )
        assert fixture.can_support_claims
        assert not fixture.is_real_world_evidence

    def test_inaccessible_source_supports_nothing(self):
        source = make_inaccessible("S01", "https://example.org")
        assert not source.can_support_claims
        assert not source.is_real_world_evidence


class TestSourceRegistry:
    def test_ids_are_sequential_and_zero_padded(self):
        registry = SourceRegistry()
        assert registry.next_source_id() == "S01"
        registry.add_source(make_retrieved("S01", "https://example.org/a"))
        assert registry.next_source_id() == "S02"

    def test_ids_are_not_reused_after_gaps(self):
        registry = SourceRegistry()
        registry.add_source(make_retrieved("S05", "https://example.org/a"))
        assert registry.next_source_id() == "S06"

    def test_same_page_with_different_url_form_is_rejected(self):
        registry = SourceRegistry()
        registry.add_source(make_retrieved("S01", "https://example.org/report"))
        with pytest.raises(DuplicateSourceError, match="already registered as S01"):
            registry.add_source(
                make_retrieved("S02", "https://EXAMPLE.org/report/?utm_campaign=x")
            )

    def test_duplicate_id_is_rejected(self):
        registry = SourceRegistry()
        registry.add_source(make_retrieved("S01", "https://example.org/a"))
        with pytest.raises(DuplicateSourceError, match="S01"):
            registry.add_source(make_retrieved("S01", "https://example.org/b"))

    def test_unknown_source_raises_specific_error(self):
        with pytest.raises(UnknownSourceError, match="S09"):
            SourceRegistry().get_source("S09")

    def test_excerpt_lookup(self):
        registry = SourceRegistry()
        registry.add_source(make_retrieved("S01", "https://example.org/a"))
        assert registry.get_excerpt("S01-E1").text == EXCERPT_TEXT
        assert registry.get_excerpt("S01-E9") is None
        assert registry.get_excerpt("S07-E1") is None

    def test_supporting_sources_exclude_inaccessible_and_are_ordered(self):
        registry = SourceRegistry()
        registry.add_source(make_retrieved("S02", "https://example.org/b"))
        registry.add_source(make_inaccessible("S03", "https://example.org/c"))
        registry.add_source(make_retrieved("S01", "https://example.org/a"))
        assert [s.source_id for s in registry.sources_supporting_claims()] == ["S01", "S02"]
        assert [s.source_id for s in registry.sources_supporting_claims(["S03", "S02"])] == [
            "S02"
        ]

    def test_registry_reports_fixture_content(self):
        registry = SourceRegistry()
        registry.add_source(make_retrieved("S01", "https://example.org/a"))
        assert not registry.contains_test_fixtures
        registry.add_source(
            SourceRecord(
                source_id="S02",
                title="Fixture",
                publisher="Test suite",
                url="https://example.invalid/f",
                retrieval_status=RetrievalStatus.TEST_FIXTURE,
                excerpts=[make_excerpt("S02")],
            )
        )
        assert registry.contains_test_fixtures

    def test_registry_round_trips_through_json(self):
        registry = SourceRegistry()
        registry.add_source(make_retrieved("S01", "https://example.org/a"))
        restored = SourceRegistry.model_validate_json(registry.model_dump_json())
        assert restored == registry
