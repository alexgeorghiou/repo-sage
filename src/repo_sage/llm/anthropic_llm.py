"""Anthropic Messages API provider. Requires ``pip install repo-sage[anthropic]``."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from repo_sage.llm.base import LLMError, Message

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass(slots=True)
class AnthropicLLM:
    model: str = DEFAULT_MODEL
    max_tokens: int = 16_000
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def _params(self, system: str, messages: Sequence[Message]) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }

    def complete(self, system: str, messages: Sequence[Message]) -> str:
        resp = self._client.messages.create(**self._params(system, messages))
        if resp.stop_reason == "refusal":
            raise LLMError("the model declined to answer this request")
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def stream(self, system: str, messages: Sequence[Message]) -> Iterator[str]:
        with self._client.messages.stream(**self._params(system, messages)) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            raise LLMError("the model declined to answer this request")
