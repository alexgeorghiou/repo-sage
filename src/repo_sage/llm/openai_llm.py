"""OpenAI Chat Completions provider. Requires ``pip install repo-sage[openai]``."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from repo_sage.llm.base import Message

DEFAULT_MODEL = "gpt-4.1-mini"


@dataclass(slots=True)
class OpenAILLM:
    model: str = DEFAULT_MODEL
    max_tokens: int = 4096
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def _messages(self, system: str, messages: Sequence[Message]) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}] + [
            {"role": m.role, "content": m.content} for m in messages
        ]

    def complete(self, system: str, messages: Sequence[Message]) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self._messages(system, messages),
        )
        return resp.choices[0].message.content or ""

    def stream(self, system: str, messages: Sequence[Message]) -> Iterator[str]:
        chunks = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self._messages(system, messages),
            stream=True,
        )
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
