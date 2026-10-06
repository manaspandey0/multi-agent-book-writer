"""Decides what happens to a chapter after each check round.

Kept free of I/O and LLM calls so the termination rule can be tested directly:
a chapter gets one initial draft plus at most ``max_revision_rounds`` revisions.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from models.book_models import (
    EditorReview,
    FactCheckReport,
    Severity,
    ValidationIssue,
)

MAX_FEEDBACK_ITEMS = 15


class NextStep(StrEnum):
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    REVISE = "revise"
    FAIL = "fail"


class RevisionDecision(BaseModel):
    next_step: NextStep
    reason: str
    feedback: list[str] = Field(default_factory=list)


class RevisionManager:
    def __init__(self, max_revision_rounds: int) -> None:
        if not 0 <= max_revision_rounds <= 2:
            raise ValueError("max_revision_rounds must be between 0 and 2")
        self.max_revision_rounds = max_revision_rounds

    def can_revise(self, revisions_used: int) -> bool:
        return revisions_used < self.max_revision_rounds

    def decide_next_step(
        self,
        revisions_used: int,
        issues: list[ValidationIssue],
        editor_review: EditorReview | None,
    ) -> RevisionDecision:
        """Apply the publication policy to the latest round of checks.

        Critical issues force a revision or, once revisions are used up, a
        failure. An unhappy Editor triggers a revision while rounds remain but
        cannot fail a chapter on its own, because style judgements are advisory.
        """
        critical = [i for i in issues if i.severity == Severity.CRITICAL]
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        editor_rejected = editor_review is not None and not editor_review.approved
        feedback = build_revision_feedback(issues, editor_review)

        if critical:
            if self.can_revise(revisions_used):
                return RevisionDecision(
                    next_step=NextStep.REVISE,
                    reason=f"{len(critical)} critical issue(s) found",
                    feedback=feedback,
                )
            return RevisionDecision(
                next_step=NextStep.FAIL,
                reason=(
                    f"{len(critical)} critical issue(s) remain after "
                    f"{revisions_used} revision(s); limit is {self.max_revision_rounds}"
                ),
                feedback=feedback,
            )
        if editor_rejected and self.can_revise(revisions_used):
            return RevisionDecision(
                next_step=NextStep.REVISE, reason="Editor requested changes", feedback=feedback
            )
        if warnings or editor_rejected:
            reason = f"{len(warnings)} warning(s)"
            if editor_rejected:
                reason += "; Editor still requested changes after the revision limit"
            return RevisionDecision(next_step=NextStep.PASS_WITH_WARNINGS, reason=reason)
        return RevisionDecision(next_step=NextStep.PASS, reason="all checks passed")


def build_revision_feedback(
    issues: list[ValidationIssue], editor_review: EditorReview | None
) -> list[str]:
    """Merge deterministic issues and Editor instructions into one ordered list."""
    ordered = sorted(issues, key=lambda issue: issue.severity != Severity.CRITICAL)
    feedback = [issue.message for issue in ordered if issue.severity != Severity.INFO]
    if editor_review is not None and not editor_review.approved:
        feedback.extend(editor_review.revision_instructions)
    return list(dict.fromkeys(feedback))[:MAX_FEEDBACK_ITEMS]


def summarize_fact_check(report: FactCheckReport | None) -> dict[str, int]:
    if report is None:
        return {}
    summary: dict[str, int] = {}
    for claim in report.claims:
        summary[claim.verdict.value] = summary.get(claim.verdict.value, 0) + 1
    return summary
