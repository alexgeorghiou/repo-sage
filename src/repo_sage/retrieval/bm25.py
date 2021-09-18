"""A small Okapi BM25 index used by the in-memory store."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from repo_sage.text import tokenize


@dataclass(slots=True)
class BM25Index:
    k1: float = 1.2
    b: float = 0.75
    _doc_len: dict[str, int] = field(default_factory=dict)
    _postings: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))
    _total_len: int = 0

    def __len__(self) -> int:
        return len(self._doc_len)

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._doc_len:
            self.remove(doc_id)
        tf = Counter(tokenize(text))
        length = sum(tf.values())
        self._doc_len[doc_id] = length
        self._total_len += length
        for term, count in tf.items():
            self._postings[term][doc_id] = count

    def remove(self, doc_id: str) -> None:
        length = self._doc_len.pop(doc_id, None)
        if length is None:
            return
        self._total_len -= length
        empty: list[str] = []
        for term, docs in self._postings.items():
            if docs.pop(doc_id, None) is not None and not docs:
                empty.append(term)
        for term in empty:
            del self._postings[term]

    def remove_many(self, doc_ids: Iterable[str]) -> None:
        for doc_id in doc_ids:
            self.remove(doc_id)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        n = len(self._doc_len)
        if n == 0:
            return []
        avg_len = self._total_len / n
        scores: dict[str, float] = defaultdict(float)
        for term in set(tokenize(query)):
            docs = self._postings.get(term)
            if not docs:
                continue
            df = len(docs)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, tf in docs.items():
                norm = self.k1 * (1 - self.b + self.b * self._doc_len[doc_id] / avg_len)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]
