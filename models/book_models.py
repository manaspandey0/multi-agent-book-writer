"""Schemas for the book brief, agent outputs and workflow state.

Models that an LLM fills in (``BookOutline``, ``WrittenChapter``,
``EditorReview``, ``FactCheckReport``) contain only fields the LLM is allowed
to decide. Bookkeeping such as revision numbers and statuses lives in models
the workflow controls (``ChapterResult``, ``BookRunState``).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from models.research_models import EXCERPT_ID_PATTERN, SOURCE_ID_PATTERN

MAX_CHAPTERS = 10


class BookBrief(BaseModel):
    """The user's request. Validated before any agent runs."""

    title: str = Field(min_length=3, max_length=200)
    audience: str = Field(min_length=3, max_length=300)
    chapter_count: int = Field(default=3, ge=1, le=MAX_CHAPTERS)
    min_words_per_chapter: int = Field(default=600, ge=100, le=5000)
    max_words_per_chapter: int = Field(default=900, ge=100, le=5000)
    tone: str = Field(default="friendly, clear, encouraging and practical", min_length=3)
    preferred_source_domains: list[str] = Field(
        default_factory=lambda: ["npci.org.in", "rbi.org.in", "pib.gov.in", "gov.in"]
    )

    @field_validator("title", "audience", "tone")
    @classmethod
    def strip_and_require_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("preferred_source_domains")
    @classmethod
    def normalize_domains(cls, domains: list[str]) -> list[str]:
        cleaned = [d.strip().lower().removeprefix("www.") for d in domains if d.strip()]
        return list(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def check_word_range(self) -> BookBrief:
        if self.min_words_per_chapter >= self.max_words_per_chapter:
            raise ValueError("min_words_per_chapter must be less than max_words_per_chapter")
        return self


class ChapterPlan(BaseModel):
    """The Planner's specification for one chapter."""

    chapter_number: int = Field(ge=1, le=MAX_CHAPTERS)
    title: str = Field(min_length=3)
    purpose: str = Field(min_length=10)
    key_points: list[str] = Field(min_length=2, max_length=8)
    research_questions: list[str] = Field(min_length=1, max_length=6)
    search_queries: list[str] = Field(min_length=1, max_length=6)


class BookOutline(BaseModel):
    """The Planner's output: the argument of the book and its chapters."""

    book_title: str = Field(min_length=3)
    central_argument: str = Field(min_length=20)
    audience: str = Field(min_length=3)
    chapters: list[ChapterPlan] = Field(min_length=1, max_length=MAX_CHAPTERS)

    @model_validator(mode="after")
    def check_chapter_numbering(self) -> BookOutline:
        numbers = [chapter.chapter_number for chapter in self.chapters]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"chapters must be numbered 1..n in order, got {numbers}")
        return self

    def matches_brief(self, brief: BookBrief) -> bool:
        return len(self.chapters) == brief.chapter_count


class WrittenChapter(BaseModel):
    """The Writer's output. The body cites sources only as ``[Sxx]`` tokens."""

    title: str = Field(min_length=3)
    body: str = Field(min_length=50)


class Severity(StrEnum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    """A problem found by deterministic checks or derived from an agent verdict."""

    code: str = Field(pattern=r"^[a-z_]+$")
    severity: Severity
    message: str
    is_heuristic: bool = False


class ReviewCategory(StrEnum):
    GRAMMAR = "grammar"
    CLARITY = "clarity"
    TONE = "tone"
    REPETITION = "repetition"
    STRUCTURE = "structure"
    CITATION = "citation"
    LENGTH = "length"
    BRIEF_COMPLIANCE = "brief_compliance"


class EditorIssue(BaseModel):
    category: ReviewCategory
    description: str = Field(min_length=5)
    suggestion: str = Field(min_length=5)


class EditorReview(BaseModel):
    """The Editor's output."""

    approved: bool
    summary: str = Field(min_length=5)
    issues: list[EditorIssue] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def rejection_needs_instructions(self) -> EditorReview:
        if not self.approved and not self.revision_instructions:
            raise ValueError("a rejected chapter must include revision_instructions")
        return self


class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


CRITICAL_VERDICTS = frozenset({ClaimVerdict.UNSUPPORTED, ClaimVerdict.CONTRADICTED})


class ClaimCheck(BaseModel):
    """One factual claim and the Fact-checker's grounded verdict on it."""

    claim_text: str = Field(min_length=5)
    cited_source_ids: list[str] = Field(default_factory=list)
    verdict: ClaimVerdict
    evidence_excerpt_ids: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=5)
    needs_human_review: bool = False

    @field_validator("cited_source_ids")
    @classmethod
    def check_source_id_format(cls, ids: list[str]) -> list[str]:
        for source_id in ids:
            if not _matches(SOURCE_ID_PATTERN, source_id):
                raise ValueError(f"invalid source ID {source_id!r}")
        return ids

    @field_validator("evidence_excerpt_ids")
    @classmethod
    def check_excerpt_id_format(cls, ids: list[str]) -> list[str]:
        for excerpt_id in ids:
            if not _matches(EXCERPT_ID_PATTERN, excerpt_id):
                raise ValueError(f"invalid excerpt ID {excerpt_id!r}")
        return ids

    @model_validator(mode="after")
    def ground_the_verdict(self) -> ClaimCheck:
        if self.verdict in {ClaimVerdict.SUPPORTED, ClaimVerdict.CONTRADICTED}:
            if not self.evidence_excerpt_ids:
                raise ValueError(
                    f"a '{self.verdict}' verdict must name the excerpt IDs it relies on"
                )
        if self.verdict == ClaimVerdict.UNCERTAIN:
            self.needs_human_review = True
        return self


class FactCheckReport(BaseModel):
    """The Fact-checker's output for one chapter."""

    claims: list[ClaimCheck] = Field(default_factory=list)

    @property
    def critical_claims(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.verdict in CRITICAL_VERDICTS]

    @property
    def claims_needing_review(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.needs_human_review]


class ChapterStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class ChapterResult(BaseModel):
    """The workflow's record of one chapter across all revision rounds."""

    plan: ChapterPlan
    status: ChapterStatus = ChapterStatus.PENDING
    draft: WrittenChapter | None = None
    rendered_markdown: str | None = None
    word_count: int | None = None
    revision_count: int = Field(default=0, ge=0)
    source_ids: list[str] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    editor_review: EditorReview | None = None
    fact_check: FactCheckReport | None = None
    failure_reason: str | None = None

    @property
    def critical_issues(self) -> list[ValidationIssue]:
        return [i for i in self.validation_issues if i.severity == Severity.CRITICAL]


class WorkflowStage(StrEnum):
    NOT_STARTED = "not_started"
    VALIDATING_BRIEF = "validating_brief"
    PLANNING = "planning"
    RESEARCHING = "researching"
    WRITING = "writing"
    VALIDATING = "validating"
    EDITING = "editing"
    FACT_CHECKING = "fact_checking"
    REVISING = "revising"
    ASSEMBLING = "assembling"
    SAVING = "saving"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class EvidenceBasis(StrEnum):
    """Where the run's evidence came from. Reported prominently in every output."""

    REAL_RETRIEVED_SOURCES = "real_retrieved_sources"
    TEST_FIXTURES = "test_fixtures"


class BookRunState(BaseModel):
    """The single shared state object for one generation run.

    Agents never receive or mutate this object. The workflow passes each agent
    only the inputs it needs and writes the agent's validated output back here.
    """

    run_id: str
    brief: BookBrief
    provider_name: str
    model_name: str
    evidence_basis: EvidenceBasis
    stage: WorkflowStage = WorkflowStage.NOT_STARTED
    current_chapter: int | None = None
    current_revision: int = 0
    outline: BookOutline | None = None
    chapters: list[ChapterResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at or datetime.now(UTC)
        return round((end - self.started_at).total_seconds(), 2)

    @property
    def is_publishable(self) -> bool:
        return bool(self.chapters) and all(
            c.status in {ChapterStatus.PASSED, ChapterStatus.PASSED_WITH_WARNINGS}
            for c in self.chapters
        )


def _matches(pattern: str, value: str) -> bool:
    return re.fullmatch(pattern, value) is not None
