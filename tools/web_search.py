"""Source discovery. Search results are leads to fetch, never evidence.

``DuckDuckGoSearchAdapter`` uses the free ``ddgs`` package, which queries public
search-engine result pages rather than an official API. It needs no key, but it
can be rate-limited or break when those pages change, so it sits behind the
small ``SearchAdapter`` protocol and can be replaced without touching agents.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol
from urllib.parse import urlsplit

from pydantic import ValidationError

from models.research_models import SearchResult, canonicalize_url
from utils.logging_config import get_logger, log_event

logger = get_logger("tools.search")

LOW_VALUE_DOMAINS = (
    "youtube.com", "facebook.com", "instagram.com", "x.com", "twitter.com",
    "pinterest.com", "quora.com", "reddit.com", "linkedin.com", "tiktok.com",
)
OFFICIAL_DOMAINS = ("gov.in", "nic.in", "npci.org.in", "rbi.org.in")
# Hosts that publish jobs or site-health mirrors, not research content.
NON_CONTENT_HOST_PREFIXES = ("careers.", "jobs.", "systemhealth.", "opportunities.")


class SearchError(RuntimeError):
    """Raised when the search backend fails after its retries."""


class SearchAdapter(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[SearchResult]: ...


class DuckDuckGoSearchAdapter:
    name = "ddgs"

    def __init__(
        self,
        region: str = "in-en",
        max_attempts: int = 2,
        backoff_seconds: float = 3.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._region = region
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException

        for attempt in range(1, self._max_attempts + 1):
            try:
                raw_results = DDGS().text(
                    query, region=self._region, safesearch="moderate", max_results=max_results
                )
                break
            except DDGSException as error:
                if "no results found" in str(error).lower():
                    # An empty result page is an answer, not a backend failure.
                    log_event(logger, "search_no_results", query=query)
                    return []
                log_event(
                    logger, "search_failed", query=query, attempt=attempt,
                    error_type=type(error).__name__,
                )
                if attempt == self._max_attempts:
                    raise SearchError(
                        f"Search for {query!r} failed after {attempt} attempts: {error}"
                    ) from error
                self._sleep(self._backoff_seconds * attempt)

        results = []
        for item in raw_results or []:
            try:
                results.append(
                    SearchResult(
                        title=str(item.get("title") or "Untitled"),
                        url=str(item.get("href") or ""),
                        snippet=str(item.get("body") or ""),
                        search_query=query,
                    )
                )
            except ValidationError:
                continue
        log_event(logger, "search_completed", query=query, results=len(results))
        return results


def _host(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def _matches_domain(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def rank_search_results(
    results: list[SearchResult], preferred_domains: list[str]
) -> list[SearchResult]:
    """Deduplicate, drop low-value platforms and put authoritative domains first.

    Order: the user's preferred domains, then other official Indian government
    domains, then everything else, keeping search order within each group.
    """
    seen: set[str] = set()
    ranked: list[tuple[int, int, SearchResult]] = []
    for position, result in enumerate(results):
        canonical = canonicalize_url(result.url)
        host = _host(result.url)
        if (
            canonical in seen
            or any(_matches_domain(host, d) for d in LOW_VALUE_DOMAINS)
            or host.startswith(NON_CONTENT_HOST_PREFIXES)
        ):
            continue
        seen.add(canonical)
        if any(_matches_domain(host, d) for d in preferred_domains):
            priority = 0
        elif any(_matches_domain(host, d) for d in OFFICIAL_DOMAINS):
            priority = 1
        else:
            priority = 2
        ranked.append((priority, position, result))
    return [result for _, _, result in sorted(ranked, key=lambda item: item[:2])]
