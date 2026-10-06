"""Research pipeline behaviour: extraction, redirects, robots, ranking and source status.

Covers ASP.NET ``<form>``-wrapped pages, robots-blocked URLs not consuming the fetch budget,
non-content hosts, and retrieved-but-irrelevant pages. All tests are offline.
"""

import io
import logging

import pytest
import requests

from agents.researcher_agent import InsufficientEvidenceError, ResearcherAgent
from config import Settings
from models.book_models import BookBrief, ChapterPlan
from models.research_models import RetrievalStatus, SearchResult, SourceRegistry
from tools.source_fetcher import (
    FetchedPage,
    FetchFailure,
    WebPageFetcher,
    extract_page_content,
    hash_text,
)
from tools.web_search import rank_search_results

SENTENCE = "UPI merchant payments settle through the bank account linked to the QR code"
URL = "https://www.rbi.org.in/Scripts/Page.aspx"


def paragraphs(n=8, topic=SENTENCE):
    return "".join(f"<p>{topic}, point number {i} explained in plain words.</p>" for i in range(n))


FORM_WRAPPED_HTML = (
    "<html><head><title>Payment systems</title></head><body>"
    f"<form id='form1' action='x'><input type='hidden' value='viewstate'>{paragraphs()}"
    "<button>Submit</button></form></body></html>"
)


class FakeResponse:
    def __init__(self, status=200, text="", content_type="text/html", url=None, body=None):
        self.status_code = status
        self.text = text
        self.headers = {"Content-Type": content_type}
        self.url = url
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"
        self._body = body if body is not None else text.encode()

    def iter_content(self, chunk_size):
        yield self._body

    def close(self):
        pass


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.headers = {}
        self.requested = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        outcome = self.responses.get(url, FakeResponse(status=404))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_fetcher(responses):
    settings = Settings(_env_file=None, llm_provider="mock", per_domain_delay_seconds=0.0)
    return WebPageFetcher(settings, session=FakeSession(responses), sleep=lambda s: None)


class TestExtraction:
    def test_form_wrapped_page_is_not_emptied(self):
        page = make_fetcher({URL: FakeResponse(text=FORM_WRAPPED_HTML, url=URL)}).fetch_page(URL)
        assert "settle through the bank account" in page.text
        assert "viewstate" not in page.text and "Submit" not in page.text

    def test_nested_table_cells_are_not_duplicated(self):
        cell = "A long enough sentence about UPI settlement timing for merchants."
        html = f"<html><body><table><tr><td><table><tr><td>{cell}</td></tr></table></td></tr></table></body></html>"
        text = extract_page_content(html, URL, URL)["text"]
        assert text.count(cell) == 1

    def test_empty_extraction_is_rejected_with_reason(self):
        html = "<html><body><script>render()</script><p>Loading</p></body></html>"
        with pytest.raises(FetchFailure, match="readable words") as failure:
            make_fetcher({URL: FakeResponse(text=html, url=URL)}).fetch_page(URL)
        assert failure.value.status == RetrievalStatus.INACCESSIBLE

    def test_text_pdf_is_extracted(self):
        pypdf = pytest.importorskip("pypdf")
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=300, height=300)
        buffer = io.BytesIO()
        writer.write(buffer)
        pdf_url = "https://www.rbi.org.in/doc.pdf"
        fetcher = make_fetcher(
            {pdf_url: FakeResponse(content_type="application/pdf", body=buffer.getvalue(), url=pdf_url)}
        )
        with pytest.raises(FetchFailure, match="0 readable words"):  # blank page: no text, no evidence
            fetcher.fetch_page(pdf_url)

    def test_corrupt_pdf_is_unsupported_not_a_crash(self):
        pytest.importorskip("pypdf")
        pdf_url = "https://www.rbi.org.in/doc.pdf"
        fetcher = make_fetcher(
            {pdf_url: FakeResponse(content_type="application/pdf", body=b"not a pdf", url=pdf_url)}
        )
        with pytest.raises(FetchFailure) as failure:
            fetcher.fetch_page(pdf_url)
        assert failure.value.status == RetrievalStatus.UNSUPPORTED_CONTENT


class TestRedirectsAndFailures:
    def test_redirect_keeps_final_url(self):
        final = "https://www.rbi.org.in/new/Page.aspx"
        page = make_fetcher({URL: FakeResponse(text=FORM_WRAPPED_HTML, url=final)}).fetch_page(URL)
        assert page.requested_url == URL and page.final_url == final

    def test_redirect_to_origin_that_disallows_us_is_blocked(self):
        final = "https://other.example.org/page"
        fetcher = make_fetcher(
            {
                URL: FakeResponse(text=FORM_WRAPPED_HTML, url=final),
                "https://other.example.org/robots.txt": FakeResponse(
                    text="User-agent: *\nDisallow: /", content_type="text/plain"
                ),
            }
        )
        with pytest.raises(FetchFailure) as failure:
            fetcher.fetch_page(URL)
        assert failure.value.status == RetrievalStatus.BLOCKED_BY_ROBOTS

    def test_robots_block_makes_no_page_request(self):
        robots = FakeResponse(text="User-agent: *\nDisallow: /", content_type="text/plain")
        fetcher = make_fetcher({"https://www.rbi.org.in/robots.txt": robots})
        with pytest.raises(FetchFailure):
            fetcher.fetch_page(URL)
        assert fetcher._session.requested == ["https://www.rbi.org.in/robots.txt"]

    def test_timeout_is_inaccessible(self):
        fetcher = make_fetcher({URL: requests.Timeout()})
        with pytest.raises(FetchFailure, match="timed out") as failure:
            fetcher.fetch_page(URL)
        assert failure.value.status == RetrievalStatus.INACCESSIBLE

    def test_tls_failure_is_reported_and_not_bypassed(self):
        fetcher = make_fetcher({URL: requests.exceptions.SSLError()})
        with pytest.raises(FetchFailure, match="not bypassed"):
            fetcher.fetch_page(URL)

    def test_diagnostics_log_status_and_drop_query_strings(self, caplog):
        secret_url = URL + "?token=abc123secret"
        fetcher = make_fetcher({secret_url: FakeResponse(status=403, url=secret_url)})
        with caplog.at_level(logging.INFO, logger="ink_evidence"), pytest.raises(FetchFailure):
            fetcher.fetch_page(secret_url)
        logged = " ".join(r.getMessage() for r in caplog.records)
        assert "http_status=403" in logged and "event=page_rejected" in logged
        assert "abc123secret" not in logged


class TestRankingQuality:
    def test_non_content_hosts_are_dropped_and_official_first(self):
        results = [
            SearchResult(title=u, url=u, search_query="q")
            for u in (
                "https://careers.npci.org.in/jobs",
                "https://systemhealth.rbi.org.in/a.html",
                "https://blog.example.com/upi",
                "https://www.rbi.org.in/x",
            )
        ]
        assert [r.url for r in rank_search_results(results, ["npci.org.in"])] == [
            "https://www.rbi.org.in/x", "https://blog.example.com/upi",
        ]


PLAN = ChapterPlan(
    chapter_number=1, title="Settlement", purpose="Explain settlement.",
    key_points=["merchant settlement", "bank account"], research_questions=["How do merchant payments settle?"],
    search_queries=["upi merchant settlement"],
)
BRIEF = BookBrief(title="UPI", audience="owners", preferred_source_domains=["npci.org.in"])


class StaticSearch:
    name = "static"

    def __init__(self, urls):
        self.urls = urls

    def search(self, query, max_results):
        return [SearchResult(title=u, url=u, search_query=query) for u in self.urls]


class ScriptedFetcher:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def fetch_page(self, url):
        self.calls.append(url)
        outcome = self.outcomes[url]
        if isinstance(outcome, FetchFailure):
            raise outcome
        return outcome


def page(url, text, final_url=None):
    from datetime import UTC, datetime

    return FetchedPage(
        requested_url=url, final_url=final_url or url, title="Official page",
        publisher="Reserve Bank of India (RBI)", text=text, publication_date=None,
        content_hash=hash_text(text), retrieved_at=datetime.now(UTC),
    )


RELEVANT_TEXT = "Merchant settlement of UPI payments reaches the bank account on the same day for most merchants."
IRRELEVANT_TEXT = "Cricket tournament schedules and stadium seating arrangements for the coming winter season."


def researcher(fetcher, urls, max_sources=3):
    settings = Settings(_env_file=None, llm_provider="mock", max_sources_per_chapter=max_sources)
    return ResearcherAgent(StaticSearch(urls), fetcher, settings)


class TestResearcherPipeline:
    def test_valid_page_is_stored_with_final_url_and_verbatim_excerpt(self):
        a, final = "https://www.rbi.org.in/a", "https://www.rbi.org.in/canonical/a"
        fetcher = ScriptedFetcher({a: page(a, RELEVANT_TEXT, final)})
        registry = SourceRegistry()
        evidence = researcher(fetcher, [a]).collect_chapter_evidence(PLAN, BRIEF, registry)
        source = registry.get_source(evidence.usable_source_ids[0])
        assert source.retrieval_status == RetrievalStatus.RETRIEVED
        assert source.url == final and source.content_hash and source.retrieved_at
        assert all(e.text in RELEVANT_TEXT for e in source.excerpts)

    def test_irrelevant_page_is_rejected_not_cited(self):
        a, b = "https://www.rbi.org.in/a", "https://www.rbi.org.in/b"
        fetcher = ScriptedFetcher({a: page(a, IRRELEVANT_TEXT), b: page(b, RELEVANT_TEXT)})
        registry = SourceRegistry()
        evidence = researcher(fetcher, [a, b]).collect_chapter_evidence(PLAN, BRIEF, registry)
        assert registry.get_source("S01").retrieval_status == RetrievalStatus.IRRELEVANT
        assert not registry.get_source("S01").excerpts
        assert evidence.usable_source_ids == ["S02"]

    def test_robots_blocked_urls_do_not_exhaust_fetch_budget(self):
        blocked = [f"https://www.npci.org.in/p{i}" for i in range(8)]
        good = "https://www.rbi.org.in/good"
        outcomes = {u: FetchFailure(RetrievalStatus.BLOCKED_BY_ROBOTS, "robots") for u in blocked}
        outcomes[good] = page(good, RELEVANT_TEXT)
        evidence = researcher(ScriptedFetcher(outcomes), [*blocked, good], max_sources=2)\
            .collect_chapter_evidence(PLAN, BRIEF, SourceRegistry())
        assert evidence.usable_source_ids == ["S09"]

    def test_alternative_source_used_when_first_fails(self):
        a, b = "https://www.rbi.org.in/a", "https://pib.gov.in/b"
        fetcher = ScriptedFetcher({
            a: FetchFailure(RetrievalStatus.INACCESSIBLE, "HTTP 403"), b: page(b, RELEVANT_TEXT),
        })
        evidence = researcher(fetcher, [a, b]).collect_chapter_evidence(PLAN, BRIEF, SourceRegistry())
        assert evidence.usable_source_ids == ["S02"] and evidence.unusable_source_ids == ["S01"]

    def test_two_urls_redirecting_to_same_page_register_once(self):
        a, b, final = "https://www.rbi.org.in/a", "https://www.rbi.org.in/b", "https://www.rbi.org.in/c"
        fetcher = ScriptedFetcher({a: page(a, RELEVANT_TEXT, final), b: page(b, RELEVANT_TEXT, final)})
        registry = SourceRegistry()
        researcher(fetcher, [a, b]).collect_chapter_evidence(PLAN, BRIEF, registry)
        assert len(registry.sources) == 1

    def test_zero_usable_sources_still_blocks_the_chapter(self):
        a = "https://www.npci.org.in/a"
        fetcher = ScriptedFetcher({a: FetchFailure(RetrievalStatus.BLOCKED_BY_ROBOTS, "robots")})
        with pytest.raises(InsufficientEvidenceError, match="cannot be written without evidence"):
            researcher(fetcher, [a]).collect_chapter_evidence(PLAN, BRIEF, SourceRegistry())


class TestSearchNoResults:
    def test_no_results_is_empty_not_a_search_error(self, monkeypatch):
        import ddgs
        from ddgs.exceptions import DDGSException

        from tools.web_search import DuckDuckGoSearchAdapter

        class FakeDDGS:
            def text(self, *args, **kwargs):
                raise DDGSException("No results found.")

        monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
        adapter = DuckDuckGoSearchAdapter(sleep=lambda s: None)
        assert adapter.search("obscure query", 6) == []
