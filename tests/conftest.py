from __future__ import annotations

from collections.abc import Callable

import pytest

from repo_sage.models import Chunk, ScoredChunk


@pytest.fixture
def make_chunk() -> Callable[..., Chunk]:
    def _make(
        path: str = "a.py",
        start: int = 1,
        end: int = 10,
        text: str = "pass",
        symbol: str | None = None,
        repo_id: str = "r",
    ) -> Chunk:
        return Chunk(
            repo_id=repo_id, path=path, start_line=start, end_line=end, text=text, symbol=symbol
        )

    return _make


def scored(*chunks: Chunk) -> list[ScoredChunk]:
    """Turn chunks into a ranked list with descending dummy scores."""
    return [ScoredChunk(chunk=c, score=float(len(chunks) - i)) for i, c in enumerate(chunks)]
