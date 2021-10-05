"""Second-stage rerankers.

The default :class:`LexicalReranker` is a cheap, deterministic heuristic that rewards
query-term coverage and symbol/path matches; it needs no model download, so tests and
CI stay hermetic. :class:`CrossEncoderReranker` is the quality option.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from repo_sage.models import ScoredChunk
from repo_sage.text import split_identifier, tokenize


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class NoopReranker:
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        return list(candidates[:top_k])


@dataclass(frozen=True, slots=True)
class LexicalReranker:
    coverage_weight: float = 1.0
    symbol_weight: float = 0.6
    path_weight: float = 0.4
    prior_weight: float = 0.5

    def score(self, query: str, item: ScoredChunk, prior: float) -> float:
        terms = set(tokenize(query))
        if not terms:
            return prior
        c = item.chunk
        body = set(tokenize(c.text))
        coverage = len(terms & body) / len(terms)
        symbol_terms = set(split_identifier(c.symbol.replace(".", "_"))) if c.symbol else set()
        symbol_hit = len(terms & symbol_terms) / len(terms) if symbol_terms else 0.0
        path_terms = set(tokenize(c.path.replace("/", " ").replace(".", " ")))
        path_hit = len(terms & path_terms) / len(terms)
        return (
            self.coverage_weight * coverage
            + self.symbol_weight * symbol_hit
            + self.path_weight * path_hit
            + self.prior_weight * prior
        )

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        best = max(c.score for c in candidates) or 1.0
        rescored = [
            replace(item, score=self.score(query, item, item.score / best)) for item in candidates
        ]
        rescored.sort(key=lambda s: (-s.score, s.chunk.path, s.chunk.start_line))
        return rescored[:top_k]


@dataclass(slots=True)
class CrossEncoderReranker:
    """Wraps a sentence-transformers cross-encoder (``pip install repo-sage[rerank]``)."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_chars: int = 2000
    _model: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [
            (query, f"{c.chunk.path} {c.chunk.symbol or ''}\n{c.chunk.text[: self.max_chars]}")
            for c in candidates
        ]
        scores = self._model.predict(pairs)
        rescored = [replace(c, score=float(s)) for c, s in zip(candidates, scores, strict=True)]
        rescored.sort(key=lambda s: -s.score)
        return rescored[:top_k]


def build_reranker(kind: str, model_name: str | None = None) -> Reranker:
    if kind == "cross-encoder":
        return CrossEncoderReranker(model_name) if model_name else CrossEncoderReranker()
    if kind == "none":
        return NoopReranker()
    return LexicalReranker()
