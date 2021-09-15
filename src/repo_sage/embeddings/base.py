from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Maps text to fixed-size dense vectors. Implementations must L2-normalise."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product; equal to cosine similarity for normalised vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))
