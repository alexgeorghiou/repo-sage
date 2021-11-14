from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


class LLM(Protocol):
    """Minimal chat-completion surface the answer engine needs."""

    @property
    def name(self) -> str: ...

    def complete(self, system: str, messages: Sequence[Message]) -> str: ...

    def stream(self, system: str, messages: Sequence[Message]) -> Iterator[str]: ...


class LLMError(RuntimeError):
    pass
