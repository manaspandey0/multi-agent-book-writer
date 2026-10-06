"""Deterministic offline provider for tests and demonstrations.

It reads the agent's structured input payload and returns schema-valid JSON
without calling any model. Its output is built only from the evidence it is
given, which in mock mode is fictional fixture text. It is a test double: its
"supported" verdicts mean "consistent with fixture text", nothing more.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from providers.llm_provider import LLMProvider, StructuredRequest
from tools.citation_validator import (
    CITATION_TOKEN,
    count_prose_words,
    split_into_sentences,
)

MockResponder = Callable[[dict[str, Any]], dict[str, Any] | str]

MOCK_CHAPTER_THEMES = (
    ("Saying yes to digital payments", "accepting the first payment"),
    ("Running the counter every day", "handling payments day to day"),
    ("Growing with digital records", "using payment records to grow"),
)
MOCK_FILLER_SENTENCES = (
    "This mock paragraph exists only to exercise the workflow and is not researched writing.",
    "A first-time owner can treat each new habit as a small experiment and adjust it calmly.",
    "Clear routines at the counter help customers feel confident about how to pay.",
    "Writing down what works and what does not makes the next decision easier.",
    "Talking with other shopkeepers nearby is a practical way to learn local habits.",
    "Patience matters, because customers and staff adapt to new tools at different speeds.",
    "Simple checks at the end of the day reduce stress and prevent small mistakes.",
    "Each section of this mock chapter follows the outline produced by the mock planner.",
)


class MockProvider(LLMProvider):
    """Offline provider. ``response_overrides`` lets tests script any agent's reply."""

    name = "mock"
    model_name = "mock-deterministic"
    is_test_provider = True

    def __init__(self, response_overrides: dict[str, MockResponder] | None = None) -> None:
        self._overrides = response_overrides or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._builders: dict[str, MockResponder] = {
            "planner": _build_outline,
            "writer": _build_chapter,
            "editor": _build_review,
            "fact_checker": _build_fact_check,
        }

    def _request_json(
        self,
        request: StructuredRequest,
        response_schema: dict[str, Any],
        repair_feedback: str | None,
    ) -> str:
        payload = dict(request.input_payload)
        if repair_feedback:
            payload["repair_feedback"] = repair_feedback
        self.calls.append((request.agent_name, payload))
        responder = self._overrides.get(request.agent_name) or self._builders.get(request.agent_name)
        if responder is None:
            raise KeyError(f"mock provider has no responder for agent {request.agent_name!r}")
        response = responder(payload)
        return response if isinstance(response, str) else json.dumps(response)

    def calls_for(self, agent_name: str) -> list[dict[str, Any]]:
        return [payload for name, payload in self.calls if name == agent_name]


def _build_outline(payload: dict[str, Any]) -> dict[str, Any]:
    brief = payload["brief"]
    chapters = []
    for number in range(1, brief["chapter_count"] + 1):
        title, focus = MOCK_CHAPTER_THEMES[(number - 1) % len(MOCK_CHAPTER_THEMES)]
        chapters.append(
            {
                "chapter_number": number,
                "title": f"{title} (mock chapter)",
                "purpose": f"Mock purpose: walk the reader through {focus}.",
                "key_points": [f"why {focus} matters", f"practical habits for {focus}"],
                "research_questions": [f"What do small merchants experience when {focus}?"],
                "search_queries": [f"upi small business merchant {focus}", "upi qr payments shop"],
            }
        )
    return {
        "book_title": brief["title"],
        "central_argument": "Mock argument: digital payments can be adopted step by step.",
        "audience": brief["audience"],
        "chapters": chapters,
    }


def _build_chapter(payload: dict[str, Any]) -> dict[str, Any]:
    plan = payload["chapter"]
    target_words = (payload["word_range"]["min"] + payload["word_range"]["max"]) // 2
    evidence_sentences = [
        f"{excerpt['text'].rstrip('.')} [{source['source_id']}]."
        for source in payload["evidence"]
        for excerpt in source["excerpts"]
    ]
    takeaway = "Takeaway: Start small, check every payment, and build one reliable habit at a time."

    paragraphs: list[str] = [
        f"This mock chapter follows the outline for \"{plan['title']}\" using fixture evidence."
    ]
    filler_index = 0
    while True:
        evidence = evidence_sentences[filler_index % len(evidence_sentences)] if evidence_sentences else ""
        filler = " ".join(
            MOCK_FILLER_SENTENCES[(filler_index + offset) % len(MOCK_FILLER_SENTENCES)]
            for offset in range(3)
        )
        candidate = paragraphs + [f"{evidence} {filler}".strip()]
        body = "\n\n".join(candidate + [takeaway])
        if count_prose_words(body) > target_words:
            break
        paragraphs = candidate
        filler_index += 1
    return {"title": plan["title"], "body": "\n\n".join(paragraphs + [takeaway])}


def _build_review(payload: dict[str, Any]) -> dict[str, Any]:
    findings = payload.get("deterministic_findings") or []
    if findings:
        return {
            "approved": False,
            "summary": "Mock editor: deterministic checks reported problems.",
            "issues": [],
            "revision_instructions": [f"Fix: {finding}" for finding in findings],
        }
    return {"approved": True, "summary": "Mock editor: no structural problems found."}


def _build_fact_check(payload: dict[str, Any]) -> dict[str, Any]:
    excerpts_by_source = {
        source["source_id"]: source["excerpts"] for source in payload["cited_sources"]
    }
    claims = []
    for sentence in split_into_sentences(payload["chapter_body"]):
        cited = CITATION_TOKEN.findall(sentence)
        if not cited:
            continue
        claim_text = re.sub(r"\s*\[S\d{2,}\]", "", sentence)
        evidence = [
            excerpt["excerpt_id"]
            for source_id in cited
            for excerpt in excerpts_by_source.get(source_id, [])
            if excerpt["text"].rstrip(".") in claim_text
        ]
        claims.append(
            {
                "claim_text": claim_text,
                "cited_source_ids": cited,
                "verdict": "supported" if evidence else "unsupported",
                "evidence_excerpt_ids": evidence,
                "explanation": "Mock fact-checker: compared against fixture text only.",
            }
        )
    return {"claims": claims}
