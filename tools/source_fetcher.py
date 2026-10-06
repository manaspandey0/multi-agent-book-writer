"""Retrieve public web pages and extract readable text and metadata.

Access rules: robots.txt is honoured, every request has a timeout, requests to
the same host are spaced out, and pages behind authentication, paywalls or
anti-bot responses are recorded as inaccessible rather than worked around.
HTML and text-based PDFs are parsed; other content types are recorded as
unsupported. Each fetch logs status, redirect target, content type, extraction
size and the rejection reason (never headers, cookies or keys).
"""

from __future__ import annotations

import hashlib
import io
import re
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Protocol
from urllib import robotparser
from urllib.parse import unquote, urlsplit

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

from config import Settings
from models.research_models import RetrievalStatus
from utils.logging_config import get_logger, log_event

logger = get_logger("tools.fetcher")

MIN_READABLE_WORDS = 150
MIN_PARAGRAPH_CHARS = 40
ACCESS_DENIED_CODES = {401, 402, 403, 407, 451}
# ``form`` is deliberately not noise: ASP.NET sites (e.g. rbi.org.in) wrap the
# whole page, content included, in a single <form>. Form controls are dropped instead.
NOISE_TAGS = (
    "script", "style", "noscript", "nav", "header", "footer", "aside", "svg",
    "input", "button", "select", "textarea",
)
BLOCK_TAGS = ("p", "li", "h2", "h3", "td", "div")
MAX_PDF_PAGES = 60
KNOWN_PUBLISHERS = {
    "npci.org.in": "National Payments Corporation of India (NPCI)",
    "rbi.org.in": "Reserve Bank of India (RBI)",
    "pib.gov.in": "Press Information Bureau, Government of India",
    "meity.gov.in": "Ministry of Electronics and Information Technology",
    "finmin.nic.in": "Ministry of Finance, Government of India",
}
DATE_META_KEYS = (
    "article:published_time", "datepublished", "publishdate", "publish-date",
    "date", "dc.date", "dc.date.issued", "pubdate",
)


class FetchFailure(Exception):
    """A page could not be used as evidence. ``status`` is recorded in the registry."""

    def __init__(self, status: RetrievalStatus, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


class FetchedPage(BaseModel):
    requested_url: str
    final_url: str
    title: str
    publisher: str
    text: str
    publication_date: date | None
    content_hash: str
    retrieved_at: datetime
    is_test_fixture: bool = False


class PageFetcher(Protocol):
    def fetch_page(self, url: str) -> FetchedPage: ...


def hash_text(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


class WebPageFetcher:
    def __init__(
        self,
        settings: Settings,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": settings.http_user_agent,
             "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.8"}
        )
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: dict[str, float] = {}
        self._robots_cache: dict[str, robotparser.RobotFileParser | None] = {}

    def fetch_page(self, url: str) -> FetchedPage:
        try:
            return self._fetch_page(url)
        except FetchFailure as failure:
            log_event(
                logger, "page_rejected", url=_loggable_url(url),
                status=failure.status.value, reason=failure.reason,
            )
            raise

    def _fetch_page(self, url: str) -> FetchedPage:
        if not self._is_allowed_by_robots(url):
            raise FetchFailure(
                RetrievalStatus.BLOCKED_BY_ROBOTS, "robots.txt disallows this URL for our agent"
            )
        response = self._polite_get(url, stream=True)
        try:
            final_url = response.url or url
            content_type = response.headers.get("Content-Type", "").lower()
            log_event(
                logger, "page_response", url=_loggable_url(url), http_status=response.status_code,
                final_url=_loggable_url(final_url) if final_url != url else "same",
                content_type=content_type.split(";")[0] or "unknown",
            )
            if response.status_code in ACCESS_DENIED_CODES:
                raise FetchFailure(
                    RetrievalStatus.INACCESSIBLE,
                    f"HTTP {response.status_code}: access restricted (not bypassed)",
                )
            if response.status_code == 429:
                raise FetchFailure(RetrievalStatus.INACCESSIBLE, "HTTP 429: site rate-limited us")
            if response.status_code >= 400:
                raise FetchFailure(RetrievalStatus.INACCESSIBLE, f"HTTP {response.status_code}")
            # A redirect to another origin needs that origin's robots.txt too.
            if _origin(final_url) != _origin(url) and not self._is_allowed_by_robots(final_url):
                raise FetchFailure(
                    RetrievalStatus.BLOCKED_BY_ROBOTS,
                    "robots.txt of the redirect destination disallows this URL",
                )
            is_pdf = "pdf" in content_type
            if "html" not in content_type and not is_pdf:
                raise FetchFailure(
                    RetrievalStatus.UNSUPPORTED_CONTENT,
                    f"content type {content_type or 'unknown'} is not parsed (HTML or PDF only)",
                )
            body = self._read_limited(response)
        finally:
            response.close()

        if is_pdf:
            page = extract_pdf_content(body, url)
        else:
            page = extract_page_content(self._decode(response, body), url, final_url)
        word_count = len(page["text"].split())
        if word_count < MIN_READABLE_WORDS:
            raise FetchFailure(
                RetrievalStatus.INACCESSIBLE,
                f"only {word_count} readable words (script-rendered page, paywall, "
                "block page or stub)",
            )
        fetched = FetchedPage(
            requested_url=url,
            final_url=final_url,
            title=page["title"],
            publisher=page["publisher"],
            text=page["text"],
            publication_date=page["publication_date"],
            content_hash=hash_text(page["text"]),
            retrieved_at=datetime.now(UTC),
        )
        log_event(
            logger, "page_retrieved", url=_loggable_url(url), words=word_count,
            kind="pdf" if is_pdf else "html", has_date=page["publication_date"] is not None,
        )
        return fetched

    def _polite_get(self, url: str, stream: bool = False) -> requests.Response:
        host = urlsplit(url).netloc.lower()
        last = self._last_request_at.get(host)
        if last is not None:
            remaining = self._settings.per_domain_delay_seconds - (self._clock() - last)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at[host] = self._clock()
        try:
            return self._session.get(
                url, timeout=self._settings.fetch_timeout_seconds, stream=stream,
                allow_redirects=True,
            )
        except requests.Timeout as error:
            raise FetchFailure(RetrievalStatus.INACCESSIBLE, "request timed out") from error
        except requests.exceptions.SSLError as error:
            raise FetchFailure(
                RetrievalStatus.INACCESSIBLE,
                "TLS certificate verification failed (not bypassed)",
            ) from error
        except requests.RequestException as error:
            raise FetchFailure(
                RetrievalStatus.INACCESSIBLE, f"network error: {type(error).__name__}"
            ) from error

    def _read_limited(self, response: requests.Response) -> bytes:
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > self._settings.max_page_bytes:
                raise FetchFailure(RetrievalStatus.UNSUPPORTED_CONTENT, "page exceeds size limit")
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _decode(response: requests.Response, body: bytes) -> str:
        encoding = response.encoding or response.apparent_encoding or "utf-8"
        return body.decode(encoding, errors="replace")

    def _is_allowed_by_robots(self, url: str) -> bool:
        """Honour robots.txt. A missing (4xx) file allows access; an unreachable one is logged."""
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots_cache:
            parser: robotparser.RobotFileParser | None = robotparser.RobotFileParser()
            try:
                response = self._polite_get(f"{origin}/robots.txt")
                log_event(
                    logger, "robots_checked", origin=origin, http_status=response.status_code
                )
                if response.status_code >= 400:
                    parser = None
                else:
                    parser.parse(response.text.splitlines())
            except FetchFailure as error:
                log_event(logger, "robots_unreachable", origin=origin, reason=error.reason)
                parser = None
            self._robots_cache[origin] = parser
        parser = self._robots_cache[origin]
        return True if parser is None else parser.can_fetch(self._settings.http_user_agent, url)


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower()


def _loggable_url(url: str) -> str:
    """Scheme, host and path only: query strings can carry tokens."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"[:200]


def _meta_content(soup: BeautifulSoup, *keys: str) -> str | None:
    for tag in soup.find_all("meta"):
        key = (tag.get("property") or tag.get("name") or tag.get("itemprop") or "").lower()
        if key in keys and tag.get("content"):
            return str(tag["content"]).strip()
    return None


def _parse_publication_date(soup: BeautifulSoup) -> date | None:
    """Read a publication date only from explicit metadata; never guess from prose."""
    candidates = [_meta_content(soup, key) for key in DATE_META_KEYS]
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag is not None:
        candidates.append(str(time_tag["datetime"]))
    for value in candidates:
        if value and (match := re.match(r"(\d{4})-(\d{2})-(\d{2})", value)):
            try:
                return date(int(match[1]), int(match[2]), int(match[3]))
            except ValueError:
                continue
    return None


def _publisher_for(soup: BeautifulSoup, url: str) -> str:
    host = urlsplit(url).netloc.lower().removeprefix("www.")
    for domain, name in KNOWN_PUBLISHERS.items():
        if host == domain or host.endswith("." + domain):
            return name
    return _meta_content(soup, "og:site_name") or host


def extract_page_content(html: str, requested_url: str, final_url: str) -> dict:
    """Return title, publisher, publication date and readable paragraph text."""
    soup = BeautifulSoup(html, "html.parser")
    title = (
        _meta_content(soup, "og:title")
        or (soup.title.get_text(" ", strip=True) if soup.title else "")
        or requested_url
    )
    publication_date = _parse_publication_date(soup)
    publisher = _publisher_for(soup, final_url)

    for tag in soup(NOISE_TAGS):
        tag.decompose()
    container = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs: list[str] = []
    seen: set[str] = set()
    for element in container.find_all(BLOCK_TAGS):
        # Only leaf blocks: a <td> or <div> wrapping other blocks would repeat their text.
        if element.find([*BLOCK_TAGS, "table"]):
            continue
        text = " ".join(element.get_text(" ", strip=True).split())
        if len(text) >= MIN_PARAGRAPH_CHARS and text not in seen:
            seen.add(text)
            paragraphs.append(text)
    return {
        "title": " ".join(title.split())[:300],
        "publisher": publisher,
        "publication_date": publication_date,
        "text": "\n".join(paragraphs),
    }


def extract_pdf_content(data: bytes, url: str) -> dict:
    """Extract text from a text-based PDF. Scanned (image-only) PDFs yield no text."""
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError as error:
        raise FetchFailure(
            RetrievalStatus.UNSUPPORTED_CONTENT, "PDF support needs the pypdf package"
        ) from error
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise FetchFailure(RetrievalStatus.INACCESSIBLE, "PDF is encrypted (not bypassed)")
        pages = [page.extract_text() or "" for page in reader.pages[:MAX_PDF_PAGES]]
        metadata_title = (reader.metadata.title if reader.metadata else None) or ""
    except FetchFailure:
        raise
    except (PyPdfError, ValueError, OSError, KeyError) as error:
        raise FetchFailure(
            RetrievalStatus.UNSUPPORTED_CONTENT, f"PDF could not be parsed: {type(error).__name__}"
        ) from error
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", "\n\n".join(pages)):
        text = " ".join(block.split())
        if len(text) >= MIN_PARAGRAPH_CHARS and text not in paragraphs:
            paragraphs.append(text)
    filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1]) or url
    return {
        "title": " ".join((metadata_title.strip() or filename).split())[:300],
        "publisher": _publisher_for(BeautifulSoup("", "html.parser"), url),
        "publication_date": None,  # PDF creation metadata is not a publication date
        "text": "\n".join(paragraphs),
    }
