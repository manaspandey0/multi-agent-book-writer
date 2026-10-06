"""Researcher agent: turns a chapter plan into registered, retrieved evidence.

This agent is tool-driven rather than LLM-driven. Evidence excerpts are exact
character slices of fetched page text, chosen by keyword relevance, so an
excerpt can never contain words the page did not contain.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from config import Settings
from models.book_models import BookBrief, ChapterPlan
from models.research_models import (
    ChapterEvidence,
    EvidenceExcerpt,
    RetrievalStatus,
    SearchResult,
    SourceRecord,
    SourceRegistry,
    canonicalize_url,
)
from tools.source_fetcher import FetchedPage, FetchFailure, PageFetcher
from tools.web_search import SearchAdapter, SearchError, rank_search_results
from utils.logging_config import get_logger, log_event

logger = get_logger("agents.researcher")

MAX_EXCERPTS_PER_SOURCE = 4
MAX_EXCERPT_CHARS = 700
MIN_EXCERPT_CHARS = 60
SITE_RESTRICTED_DOMAINS_PER_CHAPTER = 2
# Robots-blocked URLs cost no page download, so they do not use the fetch budget,
# but total URLs examined per chapter is still capped.
MAX_URLS_EXAMINED_FACTOR = 5
STOPWORDS = frozenset(
    "about after also been being from have into more most much over such than that their "
    "them then there these they this what when where which while with will would your "
    "does small india indian".split()
)


class InsufficientEvidenceError(RuntimeError):
    """Raised when no usable source could be retrieved for a chapter."""


def _keywords(texts: list[str]) -> set[str]:
    words = re.findall(r"[a-z]{4,}", " ".join(texts).lower())
    return {word for word in words if word not in STOPWORDS}


def _trim_to_sentence(paragraph: str) -> str:
    if len(paragraph) <= MAX_EXCERPT_CHARS:
        return paragraph
    cut = paragraph[:MAX_EXCERPT_CHARS]
    last_stop = cut.rfind(". ")
    return cut[: last_stop + 1] if last_stop >= MIN_EXCERPT_CHARS else cut


def select_evidence_excerpts(
    source_id: str, page_text: str, keywords: set[str], max_excerpts: int = MAX_EXCERPTS_PER_SOURCE
) -> list[EvidenceExcerpt]:
    """Pick the most relevant paragraphs as verbatim excerpts with character offsets."""
    scored: list[tuple[int, int, int, str]] = []
    position = 0
    for paragraph in page_text.split("\n"):
        start = position
        position += len(paragraph) + 1
        excerpt_text = _trim_to_sentence(paragraph.strip())
        if len(excerpt_text) < MIN_EXCERPT_CHARS:
            continue
        lowered = excerpt_text.lower()
        relevance = sum(1 for keyword in keywords if keyword in lowered)
        if relevance == 0:
            continue
        has_figures = 1 if re.search(r"\d", excerpt_text) else 0
        char_start = page_text.index(excerpt_text, start)
        scored.append((relevance + has_figures, -char_start, char_start, excerpt_text))

    best = sorted(scored, reverse=True)[:max_excerpts]
    best.sort(key=lambda item: item[2])
    return [
        EvidenceExcerpt(
            excerpt_id=f"{source_id}-E{index}",
            source_id=source_id,
            text=text,
            char_start=char_start,
            char_end=char_start + len(text),
        )
        for index, (_, _, char_start, text) in enumerate(best, start=1)
    ]


class ResearcherAgent:
    """Searches, fetches and registers sources for one chapter at a time.

    The registry is passed in explicitly and is the only state this agent
    changes, so sources found for chapter 1 are reused, not refetched, later.
    """

    def __init__(self, search_adapter: SearchAdapter, page_fetcher: PageFetcher, settings: Settings) -> None:
        self._search = search_adapter
        self._fetcher = page_fetcher
        self._settings = settings
        self._requested_to_source: dict[str, str] = {}

    def collect_chapter_evidence(
        self, chapter_plan: ChapterPlan, brief: BookBrief, registry: SourceRegistry
    ) -> ChapterEvidence:
        evidence = ChapterEvidence(chapter_number=chapter_plan.chapter_number)
        candidates = self._discover_candidates(chapter_plan, brief, evidence)
        keywords = _keywords(
            [chapter_plan.title, *chapter_plan.key_points, *chapter_plan.research_questions,
             *chapter_plan.search_queries]
        )
        max_usable = self._settings.max_sources_per_chapter
        max_attempts = max_usable * 2
        max_examined = max_usable * MAX_URLS_EXAMINED_FACTOR
        fetch_attempts = 0

        for candidate in candidates:
            if (
                len(evidence.usable_source_ids) >= max_usable
                or fetch_attempts >= max_attempts
                or evidence.urls_considered >= max_examined
            ):
                break
            evidence.urls_considered += 1
            existing = self._already_registered(candidate.url, registry)
            record = existing or self._retrieve_and_register(candidate, keywords, registry)
            if existing is None and record.retrieval_status != RetrievalStatus.BLOCKED_BY_ROBOTS:
                fetch_attempts += 1
            log_event(
                logger, "candidate_evaluated", chapter=chapter_plan.chapter_number,
                source=record.source_id, status=record.retrieval_status.value,
                usable=record.can_support_claims, excerpts=len(record.excerpts),
                reason=record.failure_reason or "",
            )
            target = evidence.usable_source_ids if record.can_support_claims else evidence.unusable_source_ids
            if record.source_id not in target:
                target.append(record.source_id)

        log_event(
            logger,
            "chapter_research_completed",
            chapter=chapter_plan.chapter_number,
            usable=len(evidence.usable_source_ids),
            unusable=len(evidence.unusable_source_ids),
            search_errors=len(evidence.search_errors),
        )
        if not evidence.usable_source_ids:
            raise InsufficientEvidenceError(
                f"No usable sources were retrieved for chapter {chapter_plan.chapter_number} "
                f"({evidence.urls_considered} URLs tried, {len(evidence.search_errors)} search "
                "errors). The chapter cannot be written without evidence."
            )
        return evidence

    def _discover_candidates(
        self, chapter_plan: ChapterPlan, brief: BookBrief, evidence: ChapterEvidence
    ) -> list[SearchResult]:
        queries = list(chapter_plan.search_queries)
        for domain in brief.preferred_source_domains[:SITE_RESTRICTED_DOMAINS_PER_CHAPTER]:
            queries.append(f"{chapter_plan.search_queries[0]} site:{domain}")

        results: list[SearchResult] = []
        for query in queries:
            evidence.queries_run.append(query)
            try:
                results.extend(
                    self._search.search(query, self._settings.search_results_per_query)
                )
            except SearchError as error:
                evidence.search_errors.append(str(error))
        return rank_search_results(results, brief.preferred_source_domains)

    def _retrieve_and_register(
        self, candidate: SearchResult, keywords: set[str], registry: SourceRegistry
    ) -> SourceRecord:
        source_id = registry.next_source_id()
        try:
            page = self._fetcher.fetch_page(candidate.url)
        except FetchFailure as failure:
            return registry.add_source(
                SourceRecord(
                    source_id=source_id,
                    title=candidate.title,
                    publisher=_host_label(candidate.url),
                    url=candidate.url,
                    retrieval_status=failure.status,
                    failure_reason=failure.reason,
                    discovered_by_query=candidate.search_query,
                )
            )
        final_match = registry.find_by_url(page.final_url)
        if final_match is not None:  # a different search URL redirected to a known page
            self._requested_to_source[canonicalize_url(candidate.url)] = final_match.source_id
            return final_match
        record = registry.add_source(self._record_from_page(source_id, page, candidate, keywords))
        self._requested_to_source[canonicalize_url(candidate.url)] = record.source_id
        return record

    def _already_registered(self, url: str, registry: SourceRegistry) -> SourceRecord | None:
        known = registry.find_by_url(url)
        if known is not None:
            return known
        source_id = self._requested_to_source.get(canonicalize_url(url))
        return registry.sources.get(source_id) if source_id else None

    @staticmethod
    def _record_from_page(
        source_id: str, page: FetchedPage, candidate: SearchResult, keywords: set[str]
    ) -> SourceRecord:
        excerpts = select_evidence_excerpts(source_id, page.text, keywords)
        if not excerpts and not page.is_test_fixture:
            return SourceRecord(
                source_id=source_id,
                title=page.title,
                publisher=page.publisher,
                url=page.final_url,
                retrieval_status=RetrievalStatus.IRRELEVANT,
                failure_reason="page was retrieved but no passage matches the chapter topic",
                discovered_by_query=candidate.search_query,
            )
        return SourceRecord(
            source_id=source_id,
            title=page.title,
            publisher=page.publisher,
            url=page.final_url,
            publication_date=page.publication_date,
            retrieved_at=page.retrieved_at,
            retrieval_status=(
                RetrievalStatus.TEST_FIXTURE if page.is_test_fixture else RetrievalStatus.RETRIEVED
            ),
            content_hash=page.content_hash,
            excerpts=excerpts,
            discovered_by_query=candidate.search_query,
        )


def _host_label(url: str) -> str:
    return urlsplit(url).netloc.removeprefix("www.") or "unknown publisher"
