"""Fact-checker agent: claim-level verdicts against retrieved excerpts.

The LLM proposes verdicts; ``ground_claim_verdicts`` then checks them in code.
An LLM verdict can still be wrong in ways code cannot detect (for example a
paraphrase that subtly changes meaning), so uncertain results are surfaced for
human review rather than hidden.
"""

from __future__ import annotations

import re

from agents.prompts import FACT_CHECKER_SYSTEM_PROMPT, FACT_CHECKER_TASK
from models.book_models import ClaimCheck, ClaimVerdict, FactCheckReport, WrittenChapter
from models.research_models import SourceRegistry
from providers.llm_provider import LLMProvider, StructuredRequest
from tools.citation_validator import extract_cited_source_ids
from utils.logging_config import get_logger, log_event

logger = get_logger("agents.fact_checker")

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers_in(text: str) -> set[str]:
    return {match.replace(",", "").rstrip(".") for match in NUMBER.findall(text)}


def _with_note(claim: ClaimCheck, verdict: ClaimVerdict, note: str) -> ClaimCheck:
    data = claim.model_dump()
    data.update(
        verdict=verdict,
        explanation=f"{claim.explanation} [Grounding check: {note}]",
        needs_human_review=True,
    )
    if verdict in {ClaimVerdict.UNSUPPORTED, ClaimVerdict.UNCERTAIN}:
        data["evidence_excerpt_ids"] = [
            eid for eid in claim.evidence_excerpt_ids if eid.split("-")[0] in claim.cited_source_ids
        ]
    return ClaimCheck.model_validate(data)


def ground_claim_verdicts(report: FactCheckReport, registry: SourceRegistry) -> FactCheckReport:
    """Check every verdict against the registry and downgrade ungrounded ones."""
    grounded: list[ClaimCheck] = []
    for claim in report.claims:
        unknown_sources = [sid for sid in claim.cited_source_ids if sid not in registry.sources]
        if unknown_sources:
            grounded.append(
                _with_note(
                    claim, ClaimVerdict.UNSUPPORTED,
                    f"cites unregistered source(s) {', '.join(unknown_sources)}",
                )
            )
            continue
        if claim.verdict not in {ClaimVerdict.SUPPORTED, ClaimVerdict.CONTRADICTED}:
            grounded.append(claim)
            continue

        excerpts = [registry.get_excerpt(eid) for eid in claim.evidence_excerpt_ids]
        if any(excerpt is None for excerpt in excerpts):
            grounded.append(
                _with_note(claim, ClaimVerdict.UNSUPPORTED, "named an excerpt ID that does not exist")
            )
            continue
        if any(excerpt.source_id not in claim.cited_source_ids for excerpt in excerpts):
            grounded.append(
                _with_note(
                    claim, ClaimVerdict.UNCERTAIN,
                    "relied on an excerpt from a source the claim does not cite",
                )
            )
            continue
        if claim.verdict == ClaimVerdict.SUPPORTED:
            evidence_numbers = set().union(*(_numbers_in(e.text) for e in excerpts))
            missing = sorted(_numbers_in(claim.claim_text) - evidence_numbers)
            if missing:
                grounded.append(
                    _with_note(
                        claim, ClaimVerdict.UNCERTAIN,
                        f"figure(s) {', '.join(missing)} not found in the cited excerpts",
                    )
                )
                continue
        grounded.append(claim)
    return FactCheckReport(claims=grounded)


class FactCheckerAgent:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def verify_chapter_citations(
        self, chapter_number: int, draft: WrittenChapter, registry: SourceRegistry
    ) -> FactCheckReport:
        """Extract claims, obtain verdicts, then ground each verdict in code."""
        cited_ids = extract_cited_source_ids(draft.body)
        cited_sources = [
            registry.sources[sid].as_prompt_evidence()
            for sid in cited_ids
            if sid in registry.sources
        ]
        request = StructuredRequest(
            agent_name="fact_checker",
            system_prompt=FACT_CHECKER_SYSTEM_PROMPT,
            task_instructions=FACT_CHECKER_TASK,
            input_payload={"chapter_body": draft.body, "cited_sources": cited_sources},
        )
        llm_report = self._provider.generate_structured(request, FactCheckReport)
        report = ground_claim_verdicts(llm_report, registry)
        log_event(
            logger,
            "chapter_fact_checked",
            chapter=chapter_number,
            claims=len(report.claims),
            critical=len(report.critical_claims),
            needs_review=len(report.claims_needing_review),
        )
        return report
