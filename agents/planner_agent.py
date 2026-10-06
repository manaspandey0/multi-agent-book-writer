"""Planner agent: turns a validated brief into a structured outline and research plan."""

from __future__ import annotations

from agents.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_TASK
from models.book_models import BookBrief, BookOutline
from providers.llm_provider import LLMProvider, StructuredRequest
from utils.logging_config import get_logger, log_event

logger = get_logger("agents.planner")


class PlanningError(RuntimeError):
    """Raised when the outline does not fit the brief."""


class PlannerAgent:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def create_book_outline(self, brief: BookBrief) -> BookOutline:
        """Return an outline with exactly ``brief.chapter_count`` chapters.

        The brief's title and audience are authoritative, so they overwrite
        whatever the model wrote for those fields.
        """
        request = StructuredRequest(
            agent_name="planner",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            task_instructions=PLANNER_TASK,
            input_payload={"brief": brief.model_dump(mode="json")},
        )
        outline = self._provider.generate_structured(request, BookOutline)
        if not outline.matches_brief(brief):
            raise PlanningError(
                f"The Planner produced {len(outline.chapters)} chapters but the brief asks for "
                f"{brief.chapter_count}."
            )
        outline = outline.model_copy(update={"book_title": brief.title, "audience": brief.audience})
        log_event(logger, "outline_created", chapters=len(outline.chapters))
        return outline
