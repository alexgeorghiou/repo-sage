from __future__ import annotations

from dataclasses import dataclass, field

from repo_sage.embeddings.base import Embedder
from repo_sage.models import ScoredChunk
from repo_sage.retrieval.fusion import collapse_overlaps, reciprocal_rank_fusion
from repo_sage.retrieval.rerank import LexicalReranker, Reranker
from repo_sage.store.base import Store


@dataclass(slots=True)
class RetrievalTrace:
    dense: list[ScoredChunk] = field(default_factory=list)
    lexical: list[ScoredChunk] = field(default_factory=list)
    fused: list[ScoredChunk] = field(default_factory=list)
    final: list[ScoredChunk] = field(default_factory=list)


@dataclass(slots=True)
class HybridRetriever:
    """Dense + lexical first stage, RRF fusion, then a reranker over the fused pool."""

    store: Store
    embedder: Embedder
    reranker: Reranker = field(default_factory=LexicalReranker)
    candidate_k: int = 40
    rrf_k: int = 60

    def retrieve(self, repo_id: str, query: str, top_k: int = 8) -> list[ScoredChunk]:
        return self.trace(repo_id, query, top_k).final

    def trace(self, repo_id: str, query: str, top_k: int = 8) -> RetrievalTrace:
        t = RetrievalTrace()
        [vector] = self.embedder.embed([query])
        t.dense = self.store.dense_search(repo_id, vector, self.candidate_k)
        t.lexical = self.store.lexical_search(repo_id, query, self.candidate_k)
        t.fused = reciprocal_rank_fusion(
            {"dense": t.dense, "lexical": t.lexical}, k=self.rrf_k, limit=self.candidate_k
        )
        reranked = self.reranker.rerank(query, t.fused, top_k=len(t.fused))
        t.final = collapse_overlaps(reranked)[:top_k]
        return t
