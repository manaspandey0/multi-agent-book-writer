"""Provider interface shared by every agent.

Agents call ``generate_structured`` with a ``StructuredRequest`` and a Pydantic
model. The base class owns JSON parsing, schema validation and the single
repair attempt, so each concrete provider only has to return raw JSON text.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError

from utils.logging_config import get_logger, log_event

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
logger = get_logger("providers")

MAX_ERROR_DETAIL_CHARS = 1500


class ProviderError(RuntimeError):
    """Base class for all LLM provider failures."""

    retryable: bool = False


class TransientProviderError(ProviderError):
    """A temporary failure (timeout, connection, 5xx, short rate limit) that outlived its retries."""

    retryable = True


class PermanentProviderError(ProviderError):
    """A failure that retrying will not fix."""


class AuthenticationFailedError(PermanentProviderError):
    """The API key is missing, invalid or lacks permission."""


class NetworkBlockedError(PermanentProviderError):
    """A proxy or firewall refused the connection before it reached the provider."""


class QuotaExhaustedError(PermanentProviderError):
    """The account's rate limit or quota requires a longer wait than the run allows."""


class ModelUnavailableError(PermanentProviderError):
    """The configured model does not exist or cannot serve this request."""


class InvalidResponseError(PermanentProviderError):
    """The model's output could not be parsed or validated, even after a repair attempt."""


class MalformedOutputError(Exception):
    """Internal signal from a provider that its raw output was not valid JSON."""


class StructuredRequest(BaseModel):
    """What an agent asks of the LLM.

    ``input_payload`` carries the agent's data as JSON-serializable values. Real
    providers render it into the prompt; the mock provider reads it directly.
    """

    agent_name: str
    system_prompt: str
    task_instructions: str
    input_payload: dict[str, Any] = Field(default_factory=dict)


class LLMProvider(ABC):
    """Common interface for real and mock language-model providers."""

    name: str
    model_name: str
    is_test_provider: bool = False

    def generate_structured(
        self, request: StructuredRequest, response_model: type[ResponseModel]
    ) -> ResponseModel:
        """Return a validated ``response_model`` instance for the request.

        Invalid JSON or a schema violation triggers exactly one repair request
        that includes the validation error. A second failure raises
        ``InvalidResponseError``.
        """
        schema = response_model.model_json_schema()
        repair_feedback: str | None = None
        for attempt in (1, 2):
            try:
                raw_output = self._request_json(request, schema, repair_feedback)
                return self._parse_response(raw_output, response_model)
            except (MalformedOutputError, json.JSONDecodeError, ValidationError) as error:
                repair_feedback = _summarize_output_error(error)
                log_event(
                    logger,
                    "llm_output_invalid",
                    agent=request.agent_name,
                    attempt=attempt,
                    error_type=type(error).__name__,
                )
        raise InvalidResponseError(
            f"{self.name} returned output for the {request.agent_name} agent that did not "
            f"match the {response_model.__name__} schema after one repair attempt. "
            f"Last error: {repair_feedback}"
        )

    @staticmethod
    def _parse_response(raw_output: str, response_model: type[ResponseModel]) -> ResponseModel:
        data = json.loads(_strip_code_fences(raw_output))
        if not isinstance(data, dict):
            raise MalformedOutputError("top-level JSON value must be an object")
        return response_model.model_validate(data)

    @abstractmethod
    def _request_json(
        self,
        request: StructuredRequest,
        response_schema: dict[str, Any],
        repair_feedback: str | None,
    ) -> str:
        """Return the model's raw output, which should be a JSON object."""


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def _summarize_output_error(error: Exception) -> str:
    if isinstance(error, ValidationError):
        details = "; ".join(
            f"{'.'.join(str(p) for p in item['loc']) or '(root)'}: {item['msg']}"
            for item in error.errors()[:10]
        )
        return f"schema validation failed: {details}"[:MAX_ERROR_DETAIL_CHARS]
    return f"invalid JSON: {error}"[:MAX_ERROR_DETAIL_CHARS]
