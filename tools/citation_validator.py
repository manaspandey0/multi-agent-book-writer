"""Deterministic chapter checks and citation rendering.

Everything here is plain code: the same input always gives the same result.
Checks marked ``is_heuristic=True`` (uncited numeric sentences) use pattern
matching and can produce false positives or miss claims written without digits.
Passing these checks shows the chapter is well-formed and consistently cited;
it does not show that every statement is true.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from models.book_models import (
    BookBrief,
    ClaimVerdict,
    FactCheckReport,
    Severity,
    ValidationIssue,
    WrittenChapter,
)
from models.research_models import RetrievalStatus, SourceRegistry, canonicalize_url

CITATION_TOKEN = re.compile(r"\[(S\d{2,})\]")
NUMERIC_CITATION = re.compile(r"\[\d+\]")
RENDERED_CITATION = re.compile(r"\[(\d+)\]")
WORD = re.compile(r"[A-Za-z0-9\u20b9][A-Za-z0-9\u20b9'’.,%-]*")
BULLET_LINE = re.compile(r"^\s*(?:[-*•+▪◦]|\d+[.)])\s+")
URL_IN_TEXT = re.compile(r"https?://|www\.", re.IGNORECASE)
REFERENCE_HEADING = re.compile(r"^\s*#*\s*(references|sources|bibliography)\b", re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
LEADING_CITATIONS = re.compile(r"^((?:\[S\d{2,}\]\s*)+)")
LIST_MARKER_ONLY = re.compile(r"^\s*\d+[.)]\s*$")
FIGURE_KINDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("circular or document identifier", re.compile(r"\b[A-Z][A-Za-z]*(?:\.[A-Za-z]+)*[.\s]*(?:No\.?\s*)?\d[\w]*(?:[/.-]\d[\w]*)+")),
    ("settlement cycle", re.compile(r"\bT\s*\+\s*\d\b")),
    ("percentage", re.compile(r"\d[\d,]*(?:\.\d+)?\s*%")),
    ("amount", re.compile(r"(?:₹|Rs\.?|INR)\s*\d[\d,]*(?:\.\d+)?")),
    ("year or date", re.compile(r"\b(?:19|20)\d{2}\b")),
)
SPEED_TERMS = re.compile(
    r"\b(?:instant(?:ly)?|real[- ]time|immediate(?:ly)?|within (?:a few )?(?:seconds|minutes|hours)|"
    r"in (?:a few )?(?:seconds|minutes)|same[- ]day|overnight|24/7|round[- ]the[- ]clock)\b",
    re.IGNORECASE,
)
FIGURE_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")
TAKEAWAY_PREFIX = "Takeaway:"
MIN_TAKEAWAY_WORDS = 5


class CitationRenderingError(ValueError):
    """Raised when a draft cannot be rendered because it cites unknown sources."""


class RenderedChapter(BaseModel):
    """A chapter with ``[Sxx]`` tokens replaced by chapter-local ``[n]`` numbers."""

    chapter_number: int
    title: str
    prose_markdown: str
    references_markdown: str
    citation_numbers: dict[str, int]

    @property
    def markdown(self) -> str:
        return (
            f"## Chapter {self.chapter_number}: {self.title}\n\n"
            f"{self.prose_markdown}\n\n### References\n\n{self.references_markdown}\n"
        )


def _issue(code: str, severity: Severity, message: str, heuristic: bool = False) -> ValidationIssue:
    return ValidationIssue(code=code, severity=severity, message=message, is_heuristic=heuristic)


def strip_citation_tokens(text: str) -> str:
    return CITATION_TOKEN.sub("", text)


def count_prose_words(body: str) -> int:
    """Count words in chapter prose, ignoring citation tokens and Markdown heading marks."""
    text = strip_citation_tokens(body).replace("#", " ")
    return len(WORD.findall(text))


def extract_cited_source_ids(body: str) -> list[str]:
    """Return cited source IDs in order of first appearance, without duplicates."""
    return list(dict.fromkeys(CITATION_TOKEN.findall(body)))


def _content_lines(body: str) -> list[str]:
    return [line.strip() for line in body.splitlines() if line.strip()]


def split_into_sentences(body: str) -> list[str]:
    """Split prose into sentences, keeping citations that follow a full stop attached."""
    sentences: list[str] = []
    for line in _content_lines(body):
        if line.startswith("#"):
            continue
        for piece in SENTENCE_SPLIT.split(line):
            leading = LEADING_CITATIONS.match(piece)
            if leading and sentences:
                sentences[-1] = f"{sentences[-1]} {leading.group(1).strip()}"
                piece = piece[leading.end():]
            if piece.strip():
                sentences.append(piece.strip())
    return sentences


def find_uncited_numeric_sentences(body: str) -> list[str]:
    """Sentences containing digits but no citation token (heuristic)."""
    return [
        sentence
        for sentence in split_into_sentences(body)
        if re.search(r"\d", strip_citation_tokens(sentence))
        and not CITATION_TOKEN.search(sentence)
        and not LIST_MARKER_ONLY.match(sentence)  # "1." from a list item is not a claim
    ]


def describe_figures(sentence: str) -> str:
    """Name the kinds of figures in a sentence, e.g. 'percentage 76.2%, year 2021'."""
    text = strip_citation_tokens(sentence)
    found = []
    for kind, pattern in FIGURE_KINDS:
        match = pattern.search(text)
        if match:
            found.append(f"{kind} {match.group(0).strip()!r}")
    if not found:
        digits = FIGURE_TOKEN.search(text)
        found.append(f"number {digits.group(0)!r}" if digits else "figure")
    return ", ".join(found)


def _unsupported_figures(sentence: str, cited_ids: list[str], registry: SourceRegistry) -> list[str]:
    """Multi-digit figures in a cited sentence that appear in none of the cited excerpts."""
    evidence = " ".join(
        excerpt.text
        for source_id in cited_ids
        if (source := registry.sources.get(source_id)) is not None and source.can_support_claims
        for excerpt in source.excerpts
    ).replace(",", "")
    if not evidence:
        return []
    figures = FIGURE_TOKEN.findall(strip_citation_tokens(sentence))
    return [
        figure for figure in dict.fromkeys(figures)
        if len(figure.replace(",", "").replace(".", "")) >= 2 and figure.replace(",", "") not in evidence
    ]


def _check_takeaway(lines: list[str]) -> list[ValidationIssue]:
    takeaway_positions = [i for i, line in enumerate(lines) if line.startswith(TAKEAWAY_PREFIX)]
    decorated = [
        line
        for line in lines
        if not line.startswith(TAKEAWAY_PREFIX)
        and line.lstrip("*_> ").lower().startswith("takeaway")
    ]
    if decorated:
        return [
            _issue(
                "takeaway_formatting",
                Severity.CRITICAL,
                f"The takeaway line must begin exactly with '{TAKEAWAY_PREFIX}' without "
                f"bold, quotes or other markup. Found: {decorated[0][:60]!r}",
            )
        ]
    if not takeaway_positions:
        return [
            _issue(
                "takeaway_missing",
                Severity.CRITICAL,
                f"The chapter must end with one line beginning '{TAKEAWAY_PREFIX}'.",
            )
        ]
    issues = []
    if len(takeaway_positions) > 1:
        issues.append(
            _issue("takeaway_repeated", Severity.CRITICAL, "Only one Takeaway line is allowed.")
        )
    if takeaway_positions[-1] != len(lines) - 1:
        issues.append(
            _issue(
                "takeaway_misplaced",
                Severity.CRITICAL,
                "The Takeaway line must be the final line of the chapter prose.",
            )
        )
    takeaway_text = lines[takeaway_positions[-1]][len(TAKEAWAY_PREFIX):]
    if len(takeaway_text.split()) < MIN_TAKEAWAY_WORDS:
        issues.append(
            _issue(
                "takeaway_too_short",
                Severity.CRITICAL,
                f"The Takeaway needs a full practical sentence ({MIN_TAKEAWAY_WORDS}+ words).",
            )
        )
    return issues


def validate_chapter_draft(
    draft: WrittenChapter,
    brief: BookBrief,
    registry: SourceRegistry,
    chapter_source_ids: list[str],
) -> list[ValidationIssue]:
    """Run every deterministic check on a draft before it is rendered."""
    body = draft.body
    lines = _content_lines(body)
    issues: list[ValidationIssue] = []

    words = count_prose_words(body)
    if not brief.min_words_per_chapter <= words <= brief.max_words_per_chapter:
        issues.append(
            _issue(
                "word_count_out_of_range",
                Severity.CRITICAL,
                f"The chapter has {words} words; it must have between "
                f"{brief.min_words_per_chapter} and {brief.max_words_per_chapter} "
                "(excluding references and citation markers).",
            )
        )

    bullet_lines = [line for line in lines if BULLET_LINE.match(line)]
    if bullet_lines:
        issues.append(
            _issue(
                "bullet_list_in_prose",
                Severity.CRITICAL,
                f"Found {len(bullet_lines)} bullet or numbered list line(s), for example "
                + "; ".join(repr(line[:90]) for line in bullet_lines[:3])
                + ". Rewrite that passage as connected prose paragraphs joined by transitions, "
                "keeping every [Sxx] citation; do not just delete the bullet or number marks.",
            )
        )

    issues.extend(_check_takeaway(lines))

    if any(REFERENCE_HEADING.match(line) for line in lines):
        issues.append(
            _issue(
                "references_in_draft",
                Severity.CRITICAL,
                "Remove the reference list from the draft; references are generated automatically.",
            )
        )
    if URL_IN_TEXT.search(body):
        issues.append(
            _issue("url_in_prose", Severity.CRITICAL, "Remove URLs from the prose; cite [Sxx] IDs.")
        )
    if NUMERIC_CITATION.search(body):
        issues.append(
            _issue(
                "numbered_citation_in_draft",
                Severity.CRITICAL,
                "Use source ID tokens such as [S01], not numbered citations such as [1].",
            )
        )

    cited_ids = extract_cited_source_ids(body)
    if not cited_ids:
        issues.append(
            _issue("no_citations", Severity.CRITICAL, "The chapter cites no sources.")
        )
    for source_id in cited_ids:
        if source_id not in registry.sources:
            issues.append(
                _issue(
                    "unknown_citation_id",
                    Severity.CRITICAL,
                    f"[{source_id}] is not a registered source. Cite only the supplied IDs.",
                )
            )
            continue
        source = registry.sources[source_id]
        if not source.can_support_claims:
            issues.append(
                _issue(
                    "cites_unusable_source",
                    Severity.CRITICAL,
                    f"[{source_id}] has status '{source.retrieval_status}' and no usable "
                    "evidence, so it cannot support a claim.",
                )
            )
        elif source_id not in chapter_source_ids:
            issues.append(
                _issue(
                    "cites_source_outside_chapter_evidence",
                    Severity.WARNING,
                    f"[{source_id}] was not part of this chapter's evidence set.",
                )
            )

    for sentence in find_uncited_numeric_sentences(body):
        issues.append(
            _issue(
                "uncited_numeric_claim",
                Severity.CRITICAL,
                f"This sentence contains {describe_figures(sentence)} but no citation: "
                f"{sentence[:200]!r}. Fix: (1) if a supplied evidence excerpt states this exact "
                "figure, attach that source's [Sxx] token before the final punctuation; "
                "(2) otherwise rewrite the sentence in general terms without the figure, "
                "identifier, date or term. Do not guess a source and never invent a citation.",
                heuristic=True,
            )
        )
    for sentence in split_into_sentences(body):
        sentence_ids = extract_cited_source_ids(sentence)
        if not sentence_ids:
            if (
                SPEED_TERMS.search(sentence)
                and not re.search(r"\d", sentence)
                and not LIST_MARKER_ONLY.match(sentence)
            ):
                issues.append(
                    _issue(
                        "uncited_speed_claim",
                        Severity.WARNING,
                        f"Processing-speed claim without a citation: {sentence[:160]!r}. Cite "
                        "an excerpt that states it, or reword without the speed claim.",
                        heuristic=True,
                    )
                )
            continue
        missing = _unsupported_figures(sentence, sentence_ids, registry)
        compact = re.sub(r"\s+%", "%", sentence)
        if missing:
            # Percentages and decimals match reliably, so a miss there is a real mismatch.
            exact = any("." in m or f"{m}%" in compact for m in missing)
            issues.append(
                _issue(
                    "figure_not_in_cited_excerpts",
                    Severity.CRITICAL if exact else Severity.WARNING,
                    f"Figure(s) {', '.join(missing)} do not appear in the evidence excerpts of "
                    f"{', '.join(sentence_ids)}: {sentence[:160]!r}. Check the figure against "
                    "the supplied excerpts or reword it.",
                    heuristic=True,
                )
            )
    return issues


def render_chapter_citations(
    chapter_number: int, draft: WrittenChapter, registry: SourceRegistry
) -> RenderedChapter:
    """Replace ``[Sxx]`` with chapter-local ``[n]`` and build the matching reference list."""
    cited_ids = extract_cited_source_ids(draft.body)
    unknown = [sid for sid in cited_ids if sid not in registry.sources]
    if unknown:
        raise CitationRenderingError(f"cannot render unknown source IDs: {', '.join(unknown)}")

    numbers = {source_id: index for index, source_id in enumerate(cited_ids, start=1)}
    prose = CITATION_TOKEN.sub(lambda match: f"[{numbers[match.group(1)]}]", draft.body.strip())

    reference_lines = []
    for source_id, number in numbers.items():
        source = registry.get_source(source_id)
        line = f"[{number}] {source.publisher}. *{source.title}*. <{source.url}>"
        if source.publication_date:
            line += f" Published {source.publication_date.isoformat()}."
        if source.retrieved_at:
            line += f" Retrieved {source.retrieved_at.date().isoformat()}."
        if source.retrieval_status == RetrievalStatus.RETRIEVED:
            line += f" ({source.tier.capitalize()} source.)"
        if source.retrieval_status == RetrievalStatus.TEST_FIXTURE:
            line += " TEST FIXTURE: not a real source."
        reference_lines.append(line)

    return RenderedChapter(
        chapter_number=chapter_number,
        title=draft.title.strip(),
        prose_markdown=prose,
        references_markdown="\n\n".join(reference_lines),
        citation_numbers=numbers,
    )


def validate_rendered_chapter(
    rendered: RenderedChapter, registry: SourceRegistry
) -> list[ValidationIssue]:
    """Confirm the in-text numbers and the reference list correspond exactly."""
    issues: list[ValidationIssue] = []
    used_numbers = {int(n) for n in RENDERED_CITATION.findall(rendered.prose_markdown)}
    listed_numbers = set(rendered.citation_numbers.values())
    expected = set(range(1, len(rendered.citation_numbers) + 1))

    if listed_numbers != expected:
        issues.append(
            _issue("reference_numbering_gap", Severity.CRITICAL, "References are not numbered 1..n.")
        )
    if missing := sorted(used_numbers - listed_numbers):
        issues.append(
            _issue(
                "missing_reference_entry",
                Severity.CRITICAL,
                f"Citations {missing} have no reference entry.",
            )
        )
    if uncited := sorted(listed_numbers - used_numbers):
        issues.append(
            _issue(
                "reference_never_cited",
                Severity.CRITICAL,
                f"References {uncited} are never cited in the prose.",
            )
        )

    seen_urls: dict[str, str] = {}
    for source_id in rendered.citation_numbers:
        source = registry.get_source(source_id)
        if not source.url:
            issues.append(
                _issue("reference_missing_url", Severity.CRITICAL, f"{source_id} has no URL.")
            )
        canonical = canonicalize_url(source.url)
        if canonical in seen_urls:
            issues.append(
                _issue(
                    "duplicate_reference",
                    Severity.CRITICAL,
                    f"{source_id} and {seen_urls[canonical]} point to the same page.",
                )
            )
        seen_urls[canonical] = source_id
    return issues


def issues_from_fact_check(report: FactCheckReport, cited_source_ids: list[str]) -> list[ValidationIssue]:
    """Translate Fact-checker verdicts into issues under the publication policy."""
    issues: list[ValidationIssue] = []
    if cited_source_ids and not report.claims:
        issues.append(
            _issue(
                "fact_check_empty",
                Severity.CRITICAL,
                "The Fact-checker returned no claims for a chapter that cites sources, so the "
                "chapter has not been verified.",
            )
        )
    for claim in report.claims:
        if claim.verdict in {ClaimVerdict.UNSUPPORTED, ClaimVerdict.CONTRADICTED}:
            issues.append(
                _issue(
                    f"claim_{claim.verdict.value}",
                    Severity.CRITICAL,
                    f"{claim.verdict.value.capitalize()} claim: {claim.claim_text[:150]!r}. "
                    f"{claim.explanation}",
                )
            )
        elif claim.verdict == ClaimVerdict.UNCERTAIN:
            issues.append(
                _issue(
                    "claim_uncertain",
                    Severity.WARNING,
                    f"Needs human review: {claim.claim_text[:150]!r}. {claim.explanation}",
                )
            )
    checked_ids = {sid for claim in report.claims for sid in claim.cited_source_ids}
    if unchecked := [sid for sid in cited_source_ids if sid not in checked_ids]:
        issues.append(
            _issue(
                "citation_not_fact_checked",
                Severity.WARNING,
                f"No checked claim relies on {', '.join(unchecked)}.",
            )
        )
    return issues
