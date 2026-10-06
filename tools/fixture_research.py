"""Offline research backend used in mock mode and tests.

The pages below are fictional. Their URLs use the reserved ``.invalid`` top-level
domain (RFC 2606), so they cannot resolve to a real website, and every
paragraph says it is a test fixture. Sources built from them carry
``RetrievalStatus.TEST_FIXTURE`` and are never reported as real research.
"""

from __future__ import annotations

from datetime import UTC, datetime

from models.research_models import RetrievalStatus, SearchResult
from tools.source_fetcher import FetchedPage, FetchFailure, hash_text

FIXTURE_HOST = "https://fixtures.invalid"
FIXTURE_PUBLISHER = "Ink & Evidence test fixtures"
UNREACHABLE_FIXTURE_URL = f"{FIXTURE_HOST}/simulated-access-denied"

FIXTURE_PAGES: dict[str, tuple[str, list[str]]] = {
    f"{FIXTURE_HOST}/getting-started-with-qr-payments": (
        "Fixture: getting started with QR code payments",
        [
            "TEST FIXTURE, fictional example: a tea stall owner in this fixture displays a "
            "printed UPI QR code so customers can pay from any UPI app on their phone.",
            "TEST FIXTURE, fictional example: in this fixture the stall records 40 digital "
            "payments during its first week of accepting UPI at the counter.",
            "TEST FIXTURE, fictional example: the fixture merchant links the QR code to a "
            "bank account and checks each payment confirmation before handing over goods.",
        ],
    ),
    f"{FIXTURE_HOST}/daily-operations-for-small-shops": (
        "Fixture: daily operations for small shops using UPI",
        [
            "TEST FIXTURE, fictional example: a fixture grocery shop reconciles UPI payments "
            "at closing time by comparing the payment app history with its sales notebook.",
            "TEST FIXTURE, fictional example: the fixture shop reports that 3 out of 5 "
            "customers chose UPI over cash on a typical fixture weekday.",
            "TEST FIXTURE, fictional example: fixture merchants are advised to watch for "
            "payment request scams and never share a UPI PIN to receive money.",
        ],
    ),
    f"{FIXTURE_HOST}/growth-and-records-for-merchants": (
        "Fixture: growth and digital records for merchants",
        [
            "TEST FIXTURE, fictional example: a fixture tailoring business uses its UPI "
            "transaction history as a simple record of income for planning purchases.",
            "TEST FIXTURE, fictional example: in this fixture a merchant saves 2 hours each "
            "week by no longer counting and depositing as much cash.",
            "TEST FIXTURE, fictional example: the fixture business owner explains digital "
            "payment options to new staff so that every customer can pay small business "
            "counters the way they prefer.",
        ],
    ),
}


class FixtureSearchAdapter:
    """Returns every fixture page plus one simulated inaccessible page for any query."""

    name = "fixture-search"

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        urls = [*FIXTURE_PAGES, UNREACHABLE_FIXTURE_URL]
        return [
            SearchResult(
                title=FIXTURE_PAGES.get(url, ("Fixture: simulated inaccessible page", []))[0],
                url=url,
                snippet="Test fixture search result; not evidence.",
                search_query=query,
            )
            for url in urls[:max_results]
        ]


class FixturePageFetcher:
    """Serves fixture pages and simulates one access-denied page."""

    def fetch_page(self, url: str) -> FetchedPage:
        if url not in FIXTURE_PAGES:
            raise FetchFailure(
                RetrievalStatus.INACCESSIBLE, "HTTP 403 (simulated by the test fixture backend)"
            )
        title, paragraphs = FIXTURE_PAGES[url]
        text = "\n".join(paragraphs)
        return FetchedPage(
            requested_url=url,
            final_url=url,
            title=title,
            publisher=FIXTURE_PUBLISHER,
            text=text,
            publication_date=None,
            content_hash=hash_text(text),
            retrieved_at=datetime.now(UTC),
            is_test_fixture=True,
        )
