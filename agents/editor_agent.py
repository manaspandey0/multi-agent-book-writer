"""Editor agent: reviews readability, tone, structure and brief compliance."""

from __future__ import annotations

from agents.prompts import EDITOR_SYSTEM_PROMPT, EDITOR_TASK
from models.book_models import BookBrief, ChapterPlan, EditorReview, ValidationIssue, WrittenChapter
from providers.llm_provider import LLMProvider, StructuredRequest
from utils.logging_config import get_logger, log_event

logger = get_logger("agents.editor")


class EditorAgent:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def review_chapter(
        self,
        brief: BookBrief,
        chapter_plan: ChapterPlan,
        draft: WrittenChapter,
        deterministic_issues: list[ValidationIssue],
    ) -> EditorReview:
        """Return a structured review.

        Deterministic findings are passed in so the Editor's instructions include
        them. The workflow does not rely on the Editor's ``approved`` flag for
        those checks; it enforces them separately.
        """
        request = StructuredRequest(
            agent_name="editor",
            system_prompt=EDITOR_SYSTEM_PROMPT,
            task_instructions=EDITOR_TASK,
            input_payload={
                "audience": brief.audience,
                "tone": brief.tone,
                "word_range": {
                    "min": brief.min_words_per_chapter,
                    "max": brief.max_words_per_chapter,
                },
                "chapter": chapter_plan.model_dump(mode="json"),
                "draft": draft.model_dump(),
                "deterministic_findings": [issue.message for issue in deterministic_issues],
            },
        )
        review = self._provider.generate_structured(request, EditorReview)
        log_event(
            logger,
            "chapter_reviewed",
            chapter=chapter_plan.chapter_number,
            approved=review.approved,
            issues=len(review.issues),
        )
        return review
