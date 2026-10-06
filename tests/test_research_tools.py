"""Research tools tested offline with fake HTTP sessions and fake search backends."""

import pytest

from agents.researcher_agent import (
    InsufficientEvidenceError,
    ResearcherAgent,
    select_evidence_excerpts,
)
from config import Settings
from models.book_models import BookBrief, ChapterPlan
from models.research_models import RetrievalStatus, SearchResult, SourceRegistry
from tools.fixture_research import FixturePageFetcher, FixtureSearchAdapter
from tools.source_fetcher import FetchFailure, WebPageFetcher, extract_page_content
from tools.web_search import SearchError, rank_search_results

ARTICLE_PARAGRAPH = (
    "Test page paragraph about UPI merchant payments and QR codes for small shops, "
    "repeated to give the extractor enough readable words to accept the page."
)
ARTICLE_HTML = f"""
<html><head><title>Test UPI article</title>
<meta property="og:site_name" content="Test News">
<meta property="article:published_time" content="2024-03-05T10:00:00Z"></head>
<body><nav>Menu Home About Contact Login Subscribe Newsletter</nav>
<article>{"".join(f"<p>{ARTICLE_PARAGRAPH} Paragraph {i}.</p>" for i in range(8))}</article>
<footer>Copyright footer text that should never become evidence at all.</footer>
</body></html>
"""


class FakeResponse:
    def __init__(self, status=200, text="", content_type="text/html; charset=utf-8", url=None):
        self.status_code = status
        self.text = text
        self.headers = {"Content-Type": content_type}
        self.url = url
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"

    def iter_content(self, chunk_size):
        data = self.text.encode()
        for start in range(0, len(data), chunk_size):
            yield data[start:start + chunk_size]

    def close(self):
        pass


class FakeSession:
    def __init__(self, responses: dict[str, FakeResponse]):
        self.responses = responses
        self.headers: dict[str, str] = {}
        self.requested: list[str] = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        assert kwargs["timeout"] > 0
        return self.responses.get(url, FakeResponse(status=404))


def make_fetcher(responses, delay=0.0, sleeps=None, clock=lambda: 0.0):
    settings = Settings(_env_file=None, llm_provider="mock", per_domain_delay_seconds=delay)
    return WebPageFetcher(settings, session=FakeSession(responses), sleep=(sleeps if sleeps is not None else []).append, clock=clock)


class TestRanking:
    def results(self, *urls):
        return [SearchResult(title=u, url=u, search_query="q") for u in urls]

    def test_preferred_then_official_then_others_and_low_value_removed(self):
        ranked = rank_search_results(
            self.results(
                "https://news.example.com/upi",
                "https://www.youtube.com/watch?v=1",
                "https://pib.gov.in/release",
                "https://www.npci.org.in/stats",
                "https://news.example.com/upi/",
            ),
            preferred_domains=["npci.org.in"],
        )
        assert [r.url for r in ranked] == [
            "https://www.npci.org.in/stats", "https://pib.gov.in/release", "https://news.example.com/upi",
        ]

    def test_lookalike_domains_are_not_preferred(self):
        ranked = rank_search_results(
            self.results("https://fakerbi.org.in/a", "https://www.rbi.org.in/b"), ["rbi.org.in"]
        )
        assert ranked[0].url == "https://www.rbi.org.in/b"


class TestFetcher:
    URL = "https://news.example.com/upi"

    def test_extracts_article_text_metadata_and_hash(self):
        fetcher = make_fetcher({self.URL: FakeResponse(text=ARTICLE_HTML, url=self.URL)})
        page = fetcher.fetch_page(self.URL)
        assert page.title == "Test UPI article"
        assert page.publisher == "Test News"
        assert page.publication_date.isoformat() == "2024-03-05"
        assert "Copyright footer" not in page.text and "Menu Home" not in page.text
        assert len(page.content_hash) == 64 and not page.is_test_fixture

    def test_robots_disallow_is_respected(self):
        robots = FakeResponse(text="User-agent: *\nDisallow: /", content_type="text/plain")
        fetcher = make_fetcher({"https://news.example.com/robots.txt": robots,
                                self.URL: FakeResponse(text=ARTICLE_HTML)})
        with pytest.raises(FetchFailure) as failure:
            fetcher.fetch_page(self.URL)
        assert failure.value.status == RetrievalStatus.BLOCKED_BY_ROBOTS

    @pytest.mark.parametrize("status", [401, 403, 451])
    def test_access_restrictions_are_not_bypassed(self, status):
        with pytest.raises(FetchFailure, match="not bypassed") as failure:
            make_fetcher({self.URL: FakeResponse(status=status)}).fetch_page(self.URL)
        assert failure.value.status == RetrievalStatus.INACCESSIBLE

    def test_pdf_is_unsupported_content(self):
        fetcher = make_fetcher({self.URL: FakeResponse(text="%PDF", content_type="application/pdf")})
        with pytest.raises(FetchFailure) as failure:
            fetcher.fetch_page(self.URL)
        assert failure.value.status == RetrievalStatus.UNSUPPORTED_CONTENT

    def test_thin_pages_are_inaccessible(self):
        fetcher = make_fetcher({self.URL: FakeResponse(text="<html><body><p>Enable JavaScript.</p></body></html>")})
        with pytest.raises(FetchFailure, match="readable words"):
            fetcher.fetch_page(self.URL)

    def test_requests_to_same_host_are_spaced_out(self):
        times = iter([0.0, 0.5, 0.5, 0.5])
        sleeps: list[float] = []
        fetcher = make_fetcher(
            {self.URL: FakeResponse(text=ARTICLE_HTML)}, delay=1.5, sleeps=sleeps, clock=lambda: next(times)
        )
        fetcher.fetch_page(self.URL)
        assert sleeps == [1.0]

    def test_publication_date_is_never_guessed_from_prose(self):
        html = "<html><body><p>" + "Published on 2020-01-01 in the text only. " * 30 + "</p></body></html>"
        assert extract_page_content(html, self.URL, self.URL)["publication_date"] is None


class TestEvidenceSelection:
    def test_excerpts_are_verbatim_slices_with_offsets(self):
        text = (
            "Unrelated paragraph about weather patterns in the mountains and rivers.\n"
            "UPI merchant payments grew as shops adopted QR codes for everyday sales in 2023.\n"
            "Another merchant paragraph explaining how payment confirmations reach the shopkeeper."
        )
        excerpts = select_evidence_excerpts("S07", text, {"merchant", "payments", "shops"})
        assert [e.excerpt_id for e in excerpts] == ["S07-E1", "S07-E2"]
        for excerpt in excerpts:
            assert text[excerpt.char_start:excerpt.char_end] == excerpt.text
        assert "weather" not in " ".join(e.text for e in excerpts)


PLAN = ChapterPlan(
    chapter_number=1, title="Accepting UPI", purpose="Show how merchants start.",
    key_points=["qr codes", "payment confirmation"], research_questions=["How do merchants accept UPI payments?"],
    search_queries=["upi merchant payments"],
)
BRIEF = BookBrief(title="Pay Me on UPI", audience="owners", preferred_source_domains=["npci.org.in"])


class FailingSearch:
    name = "failing"

    def search(self, query, max_results):
        raise SearchError(f"backend down for {query}")


class TestResearcherAgent:
    def researcher(self, search=None):
        settings = Settings(_env_file=None, llm_provider="mock", max_sources_per_chapter=6)
        return ResearcherAgent(search or FixtureSearchAdapter(), FixturePageFetcher(), settings)

    def test_registers_usable_and_inaccessible_sources_with_provenance(self):
        registry = SourceRegistry()
        evidence = self.researcher().collect_chapter_evidence(PLAN, BRIEF, registry)
        assert evidence.usable_source_ids == ["S01", "S02", "S03"]
        assert evidence.unusable_source_ids == ["S04"]
        assert registry.get_source("S04").retrieval_status == RetrievalStatus.INACCESSIBLE
        assert registry.get_source("S01").discovered_by_query == "upi merchant payments"
        assert "upi merchant payments site:npci.org.in" in evidence.queries_run

    def test_fixture_sources_are_never_real_world_evidence(self):
        registry = SourceRegistry()
        self.researcher().collect_chapter_evidence(PLAN, BRIEF, registry)
        assert registry.contains_test_fixtures
        assert not any(source.is_real_world_evidence for source in registry.sources.values())

    def test_second_chapter_reuses_registered_sources_without_new_ids(self):
        registry = SourceRegistry()
        researcher = self.researcher()
        researcher.collect_chapter_evidence(PLAN, BRIEF, registry)
        second = researcher.collect_chapter_evidence(PLAN.model_copy(update={"chapter_number": 2}), BRIEF, registry)
        assert len(registry.sources) == 4
        assert second.usable_source_ids == ["S01", "S02", "S03"]

    def test_search_outage_raises_insufficient_evidence_with_errors_recorded(self):
        with pytest.raises(InsufficientEvidenceError, match="search errors"):
            self.researcher(FailingSearch()).collect_chapter_evidence(PLAN, BRIEF, SourceRegistry())

    def test_search_snippets_are_never_stored_as_evidence(self):
        registry = SourceRegistry()
        self.researcher().collect_chapter_evidence(PLAN, BRIEF, registry)
        all_excerpts = " ".join(e.text for s in registry.sources.values() for e in s.excerpts)
        assert "Test fixture search result; not evidence." not in all_excerpts
