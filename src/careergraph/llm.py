"""Anthropic client wrapper.

Exists so that agent code never touches the SDK directly. That buys:

* **One place to enforce determinism.** ``temperature=0`` everywhere, set once.
* **A seam for tests.** Agents depend on the ``LLMClient`` protocol, so the
  unit suite injects a scripted fake and runs offline, for free, in
  milliseconds — no API key in CI, no flaky network, no bill per push.
* **Uniform error handling.** SDK exceptions become one ``LLMError`` that the
  API layer maps to a 503, instead of leaking ``anthropic.APIStatusError``
  into HTTP handlers.
* **JSON coercion in one place.** The JD matcher needs structured output, and
  models wrap JSON in prose or fences often enough that parsing needs to be
  defensive rather than optimistic.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from careergraph.config import Settings
from careergraph.logging_config import get_logger

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """Generation failed: no key, API error, or unparseable structured output."""


class LLMClient(Protocol):
    """What the agents need from a language model. Nothing more."""

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str: ...


class AnthropicClient:
    """``LLMClient`` backed by the Anthropic Messages API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self._settings.has_api_key:
                raise LLMError(
                    "ANTHROPIC_API_KEY is not set. Ingestion and retrieval work "
                    "without it, but answer generation does not."
                )
            from anthropic import Anthropic

            self._client = Anthropic(
                api_key=self._settings.anthropic_api_key.get_secret_value(),
                timeout=self._settings.llm_timeout_seconds,
                # SDK-level retry with backoff. Handles the transient 429/529s
                # that would otherwise surface as a failed user request.
                max_retries=self._settings.llm_max_retries,
            )
        return self._client

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        from anthropic import APIError

        try:
            response = self.client.messages.create(
                model=self._settings.anthropic_model,
                max_tokens=max_tokens or self._settings.llm_max_tokens,
                temperature=self._settings.llm_temperature,
                # The system prompt is a separate parameter, not a message.
                # This is not cosmetic: it is the channel the model is trained
                # to treat as instructions, which makes the grounding rules
                # meaningfully harder to override via injected text in a JD.
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except APIError as exc:
            logger.error("Anthropic API call failed", extra={"error": str(exc)})
            raise LLMError(f"Language model request failed: {exc}") from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        logger.info(
            "Generation complete",
            extra={
                "model": self._settings.anthropic_model,
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
            },
        )
        return text.strip()


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_response(raw: str) -> dict[str, Any]:
    """Extract a JSON object from a model response.

    Even with an explicit "respond with only JSON" instruction and a
    prefilled ``{``, models sometimes wrap output in ```` ```json ```` fences or
    add a sentence of preamble. Three escalating strategies, cheapest first:
    parse as-is, unwrap a code fence, then take the outermost brace-balanced
    span. Failing all three is a real error worth surfacing.
    """
    raw = raw.strip()

    try:
        return _require_object(json.loads(raw))
    except json.JSONDecodeError:
        pass

    if fenced := _FENCE_RE.search(raw):
        try:
            return _require_object(json.loads(fenced.group(1)))
        except json.JSONDecodeError:
            pass

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            return _require_object(json.loads(raw[start : end + 1]))
        except json.JSONDecodeError:
            pass

    logger.error("Model returned unparseable JSON", extra={"preview": raw[:300]})
    raise LLMError("Model did not return valid JSON.")


def _require_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise json.JSONDecodeError("expected a JSON object", str(value), 0)
    return value


def build_llm_client(settings: Settings) -> AnthropicClient:
    return AnthropicClient(settings)
