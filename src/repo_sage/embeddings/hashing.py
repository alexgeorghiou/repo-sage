"""Deterministic, dependency-free embedder based on the hashing trick.

It is not semantic, but it is stable across runs and machines, which makes it the
right default for tests, CI and offline demos. Sub-word tokens and bigrams give it
enough signal for identifier-heavy code queries.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from repo_sage.embeddings.base import normalize
from repo_sage.text import tokenize


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    sign = 1.0 if value & 1 else -1.0
    return (value >> 1) % dim, sign


@dataclass(frozen=True, slots=True)
class HashingEmbedder:
    dimension: int = 384

    @property
    def dim(self) -> int:
        return self.dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        tokens = tokenize(text)
        features = Counter(tokens)
        features.update(f"{a}|{b}" for a, b in zip(tokens, tokens[1:], strict=False))
        vec = [0.0] * self.dimension
        for feature, count in features.items():
            idx, sign = _bucket(feature, self.dimension)
            vec[idx] += sign * (1.0 + math.log(count))
        return normalize(vec)
