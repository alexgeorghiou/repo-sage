"""Core domain types shared across ingestion, retrieval and answering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class ChunkKind(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    WINDOW = "window"


@dataclass(frozen=True, slots=True)
class SourceFile:
    """A file selected for indexing, relative to the repository root."""

    path: str
    language: str | None
    content: str

    @property
    def digest(self) -> str:
        return hashlib.sha1(self.content.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class Chunk:
    """A contiguous, citable span of a source file.

    Lines are 1-based and inclusive, matching what editors and GitHub show.
    """

    repo_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    language: str | None = None
    kind: ChunkKind = ChunkKind.WINDOW
    symbol: str | None = None
    embedding: list[float] | None = None

    @property
    def id(self) -> str:
        raw = f"{self.repo_id}:{self.path}:{self.start_line}-{self.end_line}:{self.symbol or ''}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def search_text(self) -> str:
        """Text used for lexical indexing: path and symbol boost recall on names."""
        header = " ".join(p for p in (self.path, self.symbol) if p)
        return f"{header}\n{self.text}"


@dataclass(slots=True)
class Repo:
    id: str
    name: str
    source: str
    branch: str | None = None
    last_commit: str | None = None
    indexed_at: datetime | None = None
    chunk_count: int = 0
    file_count: int = 0

    def mark_indexed(self, commit: str | None, files: int, chunks: int) -> None:
        self.last_commit = commit
        self.file_count = files
        self.chunk_count = chunks
        self.indexed_at = datetime.now(UTC)


@dataclass(slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    sources: dict[str, int] = field(default_factory=dict)
    """Rank of this chunk in each retriever that returned it (1-based)."""


@dataclass(frozen=True, slots=True)
class Citation:
    path: str
    start_line: int
    end_line: int

    def __str__(self) -> str:
        if self.start_line == self.end_line:
            return f"{self.path}:{self.start_line}"
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def overlaps(self, other: Citation) -> bool:
        return (
            self.path == other.path
            and self.start_line <= other.end_line
            and other.start_line <= self.end_line
        )


@dataclass(slots=True)
class Answer:
    question: str
    text: str
    citations: list[Citation]
    invalid_citations: list[Citation]
    context: list[ScoredChunk]

    @property
    def grounded(self) -> bool:
        return bool(self.citations) and not self.invalid_citations
