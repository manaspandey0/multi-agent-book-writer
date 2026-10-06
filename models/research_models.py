"""Schemas for source discovery, retrieved evidence and the source registry.

The central distinction in this module is between *discovering* a URL
(``SearchResult``) and *retrieving evidence* from it (``SourceRecord`` with
excerpts). Only retrieved page text can support a claim.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SOURCE_ID_PATTERN = r"^S\d{2,}$"
EXCERPT_ID_PATTERN = r"^S\d{2,}-E\d+$"
TRACKING_QUERY_PREFIXES = ("utm_", "fbclid", "gclid")


class DuplicateSourceError(ValueError):
    """Raised when a source with the same ID or canonical URL is registered twice."""


class UnknownSourceError(KeyError):
    """Raised when a source ID is not present in the registry."""


class RetrievalStatus(StrEnum):
    """Outcome of trying to fetch a discovered URL."""

    RETRIEVED = "retrieved"
    INACCESSIBLE = "inaccessible"
    BLOCKED_BY_ROBOTS = "blocked_by_robots"
    UNSUPPORTED_CONTENT = "unsupported_content"
    IRRELEVANT = "irrelevant"  # fetched fine, but no passage matches the chapter topic
    TEST_FIXTURE = "test_fixture"


def canonicalize_url(url: str) -> str:
    """Normalize a URL so trivially different links to one page compare equal.

    Lower-cases the scheme and host, drops fragments, tracking parameters and
    trailing slashes. It does not follow redirects, so two URLs that redirect
    to the same page can still be registered separately.
    """
    parts = urlsplit(url.strip())
    query_pairs = [
        pair
        for pair in parts.query.split("&")
        if pair and not pair.lower().startswith(TRACKING_QUERY_PREFIXES)
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, "&".join(query_pairs), "")
    )


def _require_http_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"URL must be an absolute http(s) URL, got {url!r}")
    return url.strip()


OFFICIAL_HOST_SUFFIXES = ("gov.in", "nic.in", "rbi.org.in", "npci.org.in")


def source_tier(url: str) -> str:
    """'official' for government, regulator and NPCI hosts; otherwise 'secondary'."""
    host = urlsplit(url).netloc.lower().removeprefix("www.").split(":")[0]
    official = any(host == s or host.endswith("." + s) for s in OFFICIAL_HOST_SUFFIXES)
    return "official" if official else "secondary"


class SearchResult(BaseModel):
    """A search hit. This is a lead to investigate, never evidence."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    snippet: str = ""
    search_query: str

    @field_validator("url")
    @classmethod
    def check_url(cls, url: str) -> str:
        return _require_http_url(url)


class EvidenceExcerpt(BaseModel):
    """A passage copied verbatim from a retrieved page, with its location."""

    model_config = ConfigDict(frozen=True)

    excerpt_id: str = Field(pattern=EXCERPT_ID_PATTERN)
    source_id: str = Field(pattern=SOURCE_ID_PATTERN)
    text: str = Field(min_length=20)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)

    @model_validator(mode="after")
    def check_location_and_ownership(self) -> EvidenceExcerpt:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if not self.excerpt_id.startswith(f"{self.source_id}-"):
            raise ValueError(
                f"excerpt {self.excerpt_id} does not belong to source {self.source_id}"
            )
        return self


class SourceRecord(BaseModel):
    """Everything known about one source, including whether it was retrieved."""

    source_id: str = Field(pattern=SOURCE_ID_PATTERN)
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    url: str
    publication_date: date | None = None
    retrieved_at: datetime | None = None
    retrieval_status: RetrievalStatus
    failure_reason: str | None = None
    content_hash: str | None = None
    excerpts: list[EvidenceExcerpt] = Field(default_factory=list)
    discovered_by_query: str | None = None

    @field_validator("url")
    @classmethod
    def check_url(cls, url: str) -> str:
        return _require_http_url(url)

    @model_validator(mode="after")
    def check_status_consistency(self) -> SourceRecord:
        if self.retrieval_status == RetrievalStatus.RETRIEVED:
            if self.retrieved_at is None or not self.content_hash:
                raise ValueError("retrieved sources need retrieved_at and content_hash")
        elif self.retrieval_status != RetrievalStatus.TEST_FIXTURE:
            if self.excerpts:
                raise ValueError(
                    f"a source with status {self.retrieval_status} cannot carry excerpts"
                )
            if not self.failure_reason:
                raise ValueError("non-retrieved sources must record a failure_reason")
        for excerpt in self.excerpts:
            if excerpt.source_id != self.source_id:
                raise ValueError(
                    f"excerpt {excerpt.excerpt_id} belongs to {excerpt.source_id}, "
                    f"not {self.source_id}"
                )
        return self

    @property
    def can_support_claims(self) -> bool:
        """True when the source has excerpts a claim could be checked against."""
        return (
            self.retrieval_status in {RetrievalStatus.RETRIEVED, RetrievalStatus.TEST_FIXTURE}
            and bool(self.excerpts)
        )

    @property
    def tier(self) -> str:
        return source_tier(self.url)

    @property
    def is_real_world_evidence(self) -> bool:
        """True only for pages actually fetched from the internet."""
        return self.retrieval_status == RetrievalStatus.RETRIEVED and bool(self.excerpts)

    def as_prompt_evidence(self) -> dict:
        """The subset of this record an LLM agent sees: IDs, labels and excerpt text."""
        return {
            "source_id": self.source_id,
            "title": self.title,
            "publisher": self.publisher,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "excerpts": [{"excerpt_id": e.excerpt_id, "text": e.text} for e in self.excerpts],
        }


class SourceRegistry(BaseModel):
    """The single list of sources shared by research, writing and verification.

    IDs are assigned in registration order (``S01``, ``S02``...) and never reused,
    so a citation written by the Writer always points at the same record.
    """

    sources: dict[str, SourceRecord] = Field(default_factory=dict)

    def next_source_id(self) -> str:
        highest = max((int(sid[1:]) for sid in self.sources), default=0)
        return f"S{highest + 1:02d}"

    def find_by_url(self, url: str) -> SourceRecord | None:
        target = canonicalize_url(url)
        return next(
            (s for s in self.sources.values() if canonicalize_url(s.url) == target), None
        )

    def add_source(self, record: SourceRecord) -> SourceRecord:
        if record.source_id in self.sources:
            raise DuplicateSourceError(f"source ID {record.source_id} is already registered")
        existing = self.find_by_url(record.url)
        if existing is not None:
            raise DuplicateSourceError(
                f"{record.url} is already registered as {existing.source_id}"
            )
        self.sources[record.source_id] = record
        return record

    def get_source(self, source_id: str) -> SourceRecord:
        try:
            return self.sources[source_id]
        except KeyError:
            raise UnknownSourceError(f"source {source_id} is not in the registry") from None

    def get_excerpt(self, excerpt_id: str) -> EvidenceExcerpt | None:
        source_id = excerpt_id.split("-", 1)[0]
        source = self.sources.get(source_id)
        if source is None:
            return None
        return next((e for e in source.excerpts if e.excerpt_id == excerpt_id), None)

    def sources_supporting_claims(self, source_ids: list[str] | None = None) -> list[SourceRecord]:
        """Return usable sources, optionally limited to the given IDs, in ID order."""
        selected = (
            self.sources.values()
            if source_ids is None
            else (self.sources[sid] for sid in source_ids if sid in self.sources)
        )
        return sorted(
            (s for s in selected if s.can_support_claims), key=lambda s: int(s.source_id[1:])
        )

    @property
    def contains_test_fixtures(self) -> bool:
        return any(
            s.retrieval_status == RetrievalStatus.TEST_FIXTURE for s in self.sources.values()
        )



class ChapterEvidence(BaseModel):
    """What the Researcher found for one chapter, including what did not work."""

    chapter_number: int = Field(ge=1)
    usable_source_ids: list[str] = Field(default_factory=list)
    unusable_source_ids: list[str] = Field(default_factory=list)
    queries_run: list[str] = Field(default_factory=list)
    search_errors: list[str] = Field(default_factory=list)
    urls_considered: int = 0
