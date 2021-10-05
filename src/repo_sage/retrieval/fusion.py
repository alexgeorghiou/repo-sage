"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009).

RRF only looks at ranks, so it can combine retrievers whose raw scores live on
incomparable scales (cosine similarity vs. BM25 / ts_rank) without any tuning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from repo_sage.models import Chunk, ScoredChunk

DEFAULT_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[ScoredChunk]],
    *,
    k: int = DEFAULT_K,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Fuse named ranked lists into one list ordered by ``sum(w / (k + rank))``.

    Each result records the 1-based rank it had in every list it appeared in,
    which is handy for debugging and for the API's "why was this retrieved" view.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    sources: dict[str, dict[str, int]] = {}
    for name, results in ranked_lists.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        seen: set[str] = set()
        rank = 0
        for item in results:
            cid = item.chunk.id
            if cid in seen:  # a retriever returning duplicates must not double-count
                continue
            seen.add(cid)
            rank += 1
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            chunks.setdefault(cid, item.chunk)
            sources.setdefault(cid, {})[name] = rank
    ordered = sorted(scores, key=lambda cid: (-scores[cid], min(sources[cid].values()), cid))
    if limit is not None:
        ordered = ordered[:limit]
    return [ScoredChunk(chunk=chunks[c], score=scores[c], sources=sources[c]) for c in ordered]


def collapse_overlaps(results: Sequence[ScoredChunk], threshold: float = 0.5) -> list[ScoredChunk]:
    """Drop results mostly contained in a higher-ranked chunk of the same file.

    Overlapping line windows otherwise waste context budget on duplicated text.
    """
    kept: list[ScoredChunk] = []
    for item in results:
        c = item.chunk
        size = c.end_line - c.start_line + 1
        redundant = False
        for other in kept:
            o = other.chunk
            if o.path != c.path:
                continue
            inter = min(c.end_line, o.end_line) - max(c.start_line, o.start_line) + 1
            if inter > 0 and inter / size >= threshold:
                redundant = True
                break
        if not redundant:
            kept.append(item)
    return kept
