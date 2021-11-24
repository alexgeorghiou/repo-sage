from __future__ import annotations

from collections.abc import Iterable, Sequence


def unique_in_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant items that appear in the first ``k`` retrieved items."""
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rel & set(retrieved[:k])) / len(rel)


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    return 1.0 if set(relevant) & set(retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant item, or 0 when none was retrieved."""
    rel = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in rel:
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
