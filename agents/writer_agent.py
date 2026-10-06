"""Writer agent: drafts or revises one chapter from its plan and supplied evidence."""

from __future__ import annotations

import re

from agents.prompts import WRITER_SYSTEM_PROMPT, WRITER_TASK
from models.book_models import BookBrief, BookOutline, ChapterPlan, WrittenChapter
from models.research_models import SourceRecord
from providers.llm_provider import LLMProvider, StructuredRequest
from utils.logging_config import get_logger, log_event

logger = get_logger("agents.writer")

# Evidence is shown to the model with excerpt IDs ("S02-E1"); models sometimes cite those.
EXCERPT_CITATION = re.compile(r"\[(S\d{2,})-E\d+\]")
REPEATED_CITATION = re.compile(r"(\[S\d{2,}\])(?:\s*\1)+")


def normalize_excerpt_citations(body: str) -> str:
    """Rewrite ``[S02-E1]`` as ``[S02]`` and collapse repeats.

    Only the token's form changes: the source ID is kept exactly, so the citation
    validator still rejects unknown or unusable sources and the Fact-checker still
    verifies the claim. No citation is added, invented or removed.
    """
    return REPEATED_CITATION.sub(r"\1", EXCERPT_CITATION.sub(r"[\1]", body))


class WriterAgent:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def write_chapter(
        self,
        brief: BookBrief,
        outline: BookOutline,
        chapter_plan: ChapterPlan,
        evidence_sources: list[SourceRecord],
        revision_feedback: list[str] | None = None,
        previous_draft: WrittenChapter | None = None,
    ) -> WrittenChapter:
        """Write a chapter that may cite only the IDs of ``evidence_sources``.

        Whether the draft obeys that rule is checked afterwards by the citation
        validator; the Writer itself is never trusted to have complied.
        """
        if not evidence_sources:
            raise ValueError("write_chapter requires at least one evidence source")
        payload = {
            "book_title": brief.title,
            "audience": brief.audience,
            "tone": brief.tone,
            "central_argument": outline.central_argument,
            "chapter": chapter_plan.model_dump(mode="json"),
            "word_range": {
                "min": brief.min_words_per_chapter,
                "max": brief.max_words_per_chapter,
                "target": (brief.min_words_per_chapter + brief.max_words_per_chapter) // 2,
            },
            "allowed_source_ids": [source.source_id for source in evidence_sources],
            "evidence": [source.as_prompt_evidence() for source in evidence_sources],
        }
        if revision_feedback:
            payload["revision_feedback"] = revision_feedback
            payload["revision_mode"] = "minimal_edit: change only the quoted problem sentences"
            payload["previous_draft"] = previous_draft.model_dump() if previous_draft else None

        request = StructuredRequest(
            agent_name="writer",
            system_prompt=WRITER_SYSTEM_PROMPT,
            task_instructions=WRITER_TASK,
            input_payload=payload,
        )
        draft = self._provider.generate_structured(request, WrittenChapter)
        normalized_body = normalize_excerpt_citations(draft.body)
        if normalized_body != draft.body:
            log_event(logger, "excerpt_citations_normalized", chapter=chapter_plan.chapter_number)
            draft = draft.model_copy(update={"body": normalized_body})
        log_event(
            logger,
            "chapter_drafted",
            chapter=chapter_plan.chapter_number,
            is_revision=bool(revision_feedback),
        )
        return draft
