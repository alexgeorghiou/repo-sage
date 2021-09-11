"""Language-agnostic fallback chunker based on line windows."""

from __future__ import annotations

import re
from dataclasses import dataclass

from repo_sage.models import Chunk, ChunkKind, SourceFile

_MD_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")


def split_span(
    lines: list[str], start: int, end: int, window: int, overlap: int
) -> list[tuple[int, int]]:
    """Split the 1-based inclusive span ``start..end`` into overlapping windows.

    Window ends are nudged back to a blank line when one exists in the last
    ``overlap`` lines, so windows tend to break between logical blocks.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    overlap = max(0, min(overlap, window - 1))
    spans: list[tuple[int, int]] = []
    cur = start
    while cur <= end:
        stop = min(cur + window - 1, end)
        if stop < end:
            for candidate in range(stop, max(cur, stop - overlap), -1):
                if not lines[candidate - 1].strip():
                    stop = candidate
                    break
        spans.append((cur, stop))
        if stop >= end:
            break
        cur = max(stop - overlap + 1, cur + 1)
    return spans


def is_blank_span(lines: list[str], start: int, end: int) -> bool:
    return all(not lines[i - 1].strip() for i in range(start, end + 1))


def trim_span(lines: list[str], start: int, end: int) -> tuple[int, int] | None:
    while start <= end and not lines[start - 1].strip():
        start += 1
    while end >= start and not lines[end - 1].strip():
        end -= 1
    return (start, end) if start <= end else None


@dataclass(slots=True)
class LineWindowChunker:
    window: int = 60
    overlap: int = 10

    def chunk(self, repo_id: str, source: SourceFile) -> list[Chunk]:
        lines = source.content.split("\n")
        if source.language == "markdown":
            return self._markdown(repo_id, source, lines)
        span = trim_span(lines, 1, len(lines))
        if span is None:
            return []
        return [
            self._make(repo_id, source, lines, s, e, None)
            for s, e in split_span(lines, span[0], span[1], self.window, self.overlap)
            if not is_blank_span(lines, s, e)
        ]

    def _markdown(self, repo_id: str, source: SourceFile, lines: list[str]) -> list[Chunk]:
        """Split on headings so each section is cited under its own title."""
        sections: list[tuple[int, str | None]] = [(1, None)]
        in_fence = False
        for i, line in enumerate(lines, start=1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            m = None if in_fence else _MD_HEADING.match(line)
            if m:
                sections.append((i, m.group(2)))
        chunks: list[Chunk] = []
        for idx, (start, title) in enumerate(sections):
            end = sections[idx + 1][0] - 1 if idx + 1 < len(sections) else len(lines)
            span = trim_span(lines, start, end)
            if span is None:
                continue
            for s, e in split_span(lines, span[0], span[1], self.window, self.overlap):
                chunks.append(self._make(repo_id, source, lines, s, e, title))
        return chunks

    @staticmethod
    def _make(
        repo_id: str, source: SourceFile, lines: list[str], s: int, e: int, symbol: str | None
    ) -> Chunk:
        return Chunk(
            repo_id=repo_id,
            path=source.path,
            start_line=s,
            end_line=e,
            text="\n".join(lines[s - 1 : e]),
            language=source.language,
            kind=ChunkKind.WINDOW,
            symbol=symbol,
        )
