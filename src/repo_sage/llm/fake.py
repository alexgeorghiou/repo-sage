"""Deterministic LLM stand-in used by tests, CI and the zero-config demo.

It reads the numbered context blocks from the prompt and writes a short, extractive
answer that cites the most relevant ones in the exact format the real prompt asks
for, so the whole pipeline (including citation validation) runs end to end offline.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from repo_sage.llm.base import Message
from repo_sage.text import tokenize

_BLOCK = re.compile(
    r"^### \[(?P<n>\d+)\] (?P<path>\S+):(?P<start>\d+)-(?P<end>\d+)(?: \((?P<symbol>[^)]*)\))?\n"
    r"```[^\n]*\n(?P<body>.*?)\n```",
    re.MULTILINE | re.DOTALL,
)
_LINE_NO = re.compile(r"^\s*\d+ \| ?")
_QUESTION = re.compile(r"^Question:\s*(?P<q>.+)$", re.MULTILINE)


def _strip_number(line: str) -> str:
    return _LINE_NO.sub("", line).strip()


@dataclass(slots=True)
class FakeLLM:
    max_citations: int = 2
    scripted: list[str] = field(default_factory=list)
    """Optional canned replies, consumed in order before falling back to extraction."""
    calls: list[tuple[str, list[Message]]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fake"

    def complete(self, system: str, messages: Sequence[Message]) -> str:
        self.calls.append((system, list(messages)))
        if self.scripted:
            return self.scripted.pop(0)
        return self._extractive(messages[-1].content if messages else "")

    def stream(self, system: str, messages: Sequence[Message]) -> Iterator[str]:
        text = self.complete(system, messages)
        yield from re.findall(r"\S+\s*", text)

    def _extractive(self, prompt: str) -> str:
        blocks = list(_BLOCK.finditer(prompt))
        if not blocks:
            return "I could not find anything in the indexed code that answers this."
        q = _QUESTION.search(prompt)
        terms = set(tokenize(q.group("q"))) if q else set()

        def overlap(m: re.Match[str]) -> int:
            return len(terms & set(tokenize(m.group("body") + " " + (m.group("symbol") or ""))))

        ranked = sorted(blocks, key=lambda m: (-overlap(m), int(m.group("n"))))
        chosen = ranked[: self.max_citations]
        parts: list[str] = []
        for m in chosen:
            label = f"`{m.group('symbol')}`" if m.group("symbol") else "this code"
            cite = f"[{m.group('path')}:{m.group('start')}-{m.group('end')}]"
            first = next((ln for ln in map(_strip_number, m.group("body").splitlines()) if ln), "")
            parts.append(f"The relevant logic is in {label} {cite}, starting with `{first[:80]}`.")
        return " ".join(parts)
