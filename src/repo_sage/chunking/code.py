"""Chunker facade: tree-sitter where a grammar exists, line windows everywhere else."""

from __future__ import annotations

from dataclasses import dataclass, field

from repo_sage.chunking.lines import LineWindowChunker
from repo_sage.chunking.treesitter import TreeSitterChunker
from repo_sage.models import Chunk, SourceFile


@dataclass(slots=True)
class CodeChunker:
    max_lines: int = 80
    window: int = 60
    overlap: int = 10
    _syntax: TreeSitterChunker = field(init=False)
    _lines: LineWindowChunker = field(init=False)

    def __post_init__(self) -> None:
        self._syntax = TreeSitterChunker(self.max_lines, self.window, self.overlap)
        self._lines = LineWindowChunker(self.window, self.overlap)

    def chunk(self, repo_id: str, source: SourceFile) -> list[Chunk]:
        if self._syntax.supports(source.language):
            try:
                chunks = self._syntax.chunk(repo_id, source)
            except (ValueError, RecursionError):
                chunks = []
            if chunks:
                return chunks
        return self._lines.chunk(repo_id, source)
