"""Assemble the book, build reports and save each run to its own directory."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from models.book_models import BookRunState, ChapterStatus, EvidenceBasis, Severity
from models.research_models import ChapterEvidence, RetrievalStatus, SourceRegistry
from workflow.revision_manager import summarize_fact_check

BOOK_FILENAME = "book.md"
FAILURE_REPORT_FILENAME = "failure_report.md"
SOURCES_FILENAME = "sources.json"
REPORT_FILENAME = "run_report.json"
LOG_FILENAME = "run.log"

MOCK_BANNER = (
    "> **Test output — not a researched book.** This draft was produced by the offline mock "
    "provider from fictional test fixtures. Its citations point to fixture pages, and its "
    "fact-check results only show consistency with fixture text."
)
VERIFICATION_DISCLAIMER = (
    "Automated checks confirm structure, citation consistency and that cited excerpts appear "
    "to support each identified claim. They do not prove every statement is true; an LLM "
    "fact-checker can misjudge evidence. Review flagged claims and spot-check sources."
)


def create_run_directory(outputs_dir: Path) -> tuple[str, Path]:
    """Create a new, never-reused directory such as ``run_20260101_101500_a1b2c3``."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        run_id = f"run_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        run_dir = outputs_dir / run_id
        try:
            run_dir.mkdir()
            return run_id, run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a unique run directory in {outputs_dir}")


def assemble_book_markdown(state: BookRunState, registry: SourceRegistry | None = None) -> str:
    """Join passed chapters into the final book. Callers must check publishability first."""
    if not state.is_publishable:
        raise ValueError("refusing to assemble a book with unresolved critical failures")
    parts = [f"# {state.brief.title}", ""]
    if state.evidence_basis == EvidenceBasis.TEST_FIXTURES:
        parts += [MOCK_BANNER, ""]
    parts += [f"*Written for: {state.brief.audience}.*", ""]
    review_count = sum(
        len(c.fact_check.claims_needing_review) for c in state.chapters if c.fact_check
    )
    if review_count:
        parts += [
            f"> **{review_count} claim(s) are flagged for human review.** See run_report.json.",
            "",
        ]
    for chapter in state.chapters:
        parts += [chapter.rendered_markdown or "", ""]
    if registry is not None and state.evidence_basis != EvidenceBasis.TEST_FIXTURES:
        cited = {sid for c in state.chapters for sid in c.source_ids}
        secondary = sorted(sid for sid in cited if registry.sources[sid].tier == "secondary")
        parts += [
            f"> **Evidence limits:** {len(cited) - len(secondary)} of {len(cited)} cited "
            f"sources are official (government, regulator or NPCI); {len(secondary)} are "
            "secondary (company, news or other websites) and should be checked against "
            "primary sources before relying on them for decisions.",
            "",
        ]
    parts += ["---", "", f"*{VERIFICATION_DISCLAIMER}*", ""]
    return "\n".join(parts)


def build_failure_report(state: BookRunState) -> str:
    lines = [
        f"# Run {state.run_id}: book not published",
        "",
        "The book was not assembled because at least one chapter failed a mandatory check "
        "or the run stopped early. Details for each chapter follow.",
        "",
    ]
    lines += [f"**Run error:** {error}" for error in state.errors]
    for chapter in state.chapters:
        lines += ["", f"## Chapter {chapter.plan.chapter_number}: {chapter.plan.title}", ""]
        lines.append(f"Status: **{chapter.status.value}** after {chapter.revision_count} revision(s).")
        if chapter.failure_reason:
            lines += ["", f"Reason: {chapter.failure_reason}"]
        critical = [i for i in chapter.validation_issues if i.severity == Severity.CRITICAL]
        if critical:
            lines += ["", "Unresolved critical issues:", ""]
            lines += [f"- `{issue.code}`: {issue.message}" for issue in critical]
        if critical and chapter.draft is not None:
            # Diagnostic only: the rejected draft is never published or cited as research.
            ending = " | ".join(chapter.draft.body.strip().splitlines()[-2:])
            lines += ["", f"Rejected draft ending (diagnostic, not published): {ending[-400:]}"]
        if chapter.fact_check:
            for claim in chapter.fact_check.critical_claims:
                sources = ", ".join(claim.cited_source_ids) or "no citation"
                lines.append(f"- Claim ({claim.verdict.value}; {sources}): {claim.claim_text}")
    return "\n".join(lines) + "\n"


def _source_status_counts(registry: SourceRegistry) -> dict[str, int]:
    counts = {status.value: 0 for status in RetrievalStatus}
    for source in registry.sources.values():
        counts[source.retrieval_status.value] += 1
    return {status: count for status, count in counts.items() if count}


def build_run_report(
    state: BookRunState,
    registry: SourceRegistry,
    research: dict[int, ChapterEvidence],
    settings_summary: dict[str, Any],
    publication_status: str,
) -> dict[str, Any]:
    chapters = []
    for chapter in state.chapters:
        number = chapter.plan.chapter_number
        chapters.append(
            {
                "chapter_number": number,
                "title": chapter.draft.title if chapter.draft else chapter.plan.title,
                "status": chapter.status.value,
                "failure_reason": chapter.failure_reason,
                "word_count": chapter.word_count,
                "revision_count": chapter.revision_count,
                "cited_source_ids": chapter.source_ids,
                "research": research[number].model_dump() if number in research else None,
                "validation_issues": [i.model_dump(mode="json") for i in chapter.validation_issues],
                "editor_review": (
                    chapter.editor_review.model_dump(mode="json") if chapter.editor_review else None
                ),
                "fact_check_performed": chapter.fact_check is not None,
                "fact_check_summary": summarize_fact_check(chapter.fact_check),
                "fact_check_claims": (
                    [c.model_dump(mode="json") for c in chapter.fact_check.claims]
                    if chapter.fact_check
                    else []
                ),
            }
        )
    is_fixture_run = state.evidence_basis == EvidenceBasis.TEST_FIXTURES
    return {
        "run_id": state.run_id,
        "started_at": state.started_at.isoformat(),
        "finished_at": state.finished_at.isoformat() if state.finished_at else None,
        "elapsed_seconds": state.elapsed_seconds,
        "final_stage": state.stage.value,
        "publication_status": publication_status,
        "provider": state.provider_name,
        "model": state.model_name,
        "evidence_basis": state.evidence_basis.value,
        "verification_scope": (
            "fixture_consistency_only_not_real_world_verification"
            if is_fixture_run
            else "llm_fact_check_grounded_in_retrieved_excerpts"
        ),
        "verification_disclaimer": VERIFICATION_DISCLAIMER,
        "settings": settings_summary,
        "brief": state.brief.model_dump(mode="json"),
        "outline": state.outline.model_dump(mode="json") if state.outline else None,
        "chapters": chapters,
        "sources": {
            "total": len(registry.sources),
            "by_retrieval_status": _source_status_counts(registry),
            "by_tier": {
                tier: sum(1 for s in registry.sources.values() if s.is_real_world_evidence and s.tier == tier)
                for tier in ("official", "secondary")
            },
            "real_world_evidence_sources": sum(
                1 for s in registry.sources.values() if s.is_real_world_evidence
            ),
        },
        "errors": state.errors,
        "chapters_passed": sum(
            1 for c in state.chapters
            if c.status in {ChapterStatus.PASSED, ChapterStatus.PASSED_WITH_WARNINGS}
        ),
    }


def save_run_artifacts(
    run_dir: Path,
    registry: SourceRegistry,
    report: dict[str, Any],
    book_markdown: str | None,
    failure_report: str | None,
) -> dict[str, Path]:
    """Write the run's files. ``book.md`` exists only for a published book."""
    saved: dict[str, Path] = {}
    if book_markdown is not None:
        saved["book"] = run_dir / BOOK_FILENAME
        saved["book"].write_text(book_markdown, encoding="utf-8")
    if failure_report is not None:
        saved["failure_report"] = run_dir / FAILURE_REPORT_FILENAME
        saved["failure_report"].write_text(failure_report, encoding="utf-8")
    saved["sources"] = run_dir / SOURCES_FILENAME
    saved["sources"].write_text(registry.model_dump_json(indent=2), encoding="utf-8")
    saved["report"] = run_dir / REPORT_FILENAME
    saved["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return saved


def list_saved_runs(outputs_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Summaries of saved runs, newest first. Unreadable runs are listed, not hidden."""
    if not outputs_dir.is_dir():
        return []
    summaries = []
    for run_dir in sorted(outputs_dir.glob("run_*"), reverse=True)[:limit]:
        summary: dict[str, Any] = {"run_id": run_dir.name, "path": run_dir, "status": "unreadable"}
        try:
            report = json.loads((run_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
            summary.update(
                status=report.get("publication_status", "unknown"),
                provider=report.get("provider"),
                evidence_basis=report.get("evidence_basis"),
                chapters_passed=report.get("chapters_passed"),
            )
        except (OSError, ValueError):
            pass
        summary["has_book"] = (run_dir / BOOK_FILENAME).exists()
        summaries.append(summary)
    return summaries
