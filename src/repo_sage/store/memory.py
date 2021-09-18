"""Process-local store. Used by tests and for quick local experiments without Postgres."""

from __future__ import annotations

import copy
import threading
from collections import defaultdict
from collections.abc import Iterable, Sequence

from repo_sage.embeddings.base import cosine
from repo_sage.models import Chunk, Repo, ScoredChunk
from repo_sage.retrieval.bm25 import BM25Index


class InMemoryStore:
    def __init__(self) -> None:
        self._repos: dict[str, Repo] = {}
        self._chunks: dict[str, dict[str, Chunk]] = defaultdict(dict)
        self._by_file: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._bm25: dict[str, BM25Index] = defaultdict(BM25Index)
        self._lock = threading.RLock()

    # repositories -----------------------------------------------------------------

    def upsert_repo(self, repo: Repo) -> None:
        with self._lock:
            self._repos[repo.id] = copy.copy(repo)

    def get_repo(self, repo_id: str) -> Repo | None:
        with self._lock:
            repo = self._repos.get(repo_id)
            return copy.copy(repo) if repo else None

    def list_repos(self) -> list[Repo]:
        with self._lock:
            return [copy.copy(r) for r in sorted(self._repos.values(), key=lambda r: r.name)]

    def delete_repo(self, repo_id: str) -> None:
        with self._lock:
            self.clear_chunks(repo_id)
            self._repos.pop(repo_id, None)

    # chunks -----------------------------------------------------------------------

    def replace_file(self, repo_id: str, path: str, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            self.delete_files(repo_id, [path])
            ids = self._by_file[(repo_id, path)]
            for chunk in chunks:
                self._chunks[repo_id][chunk.id] = chunk
                self._bm25[repo_id].add(chunk.id, chunk.search_text())
                ids.append(chunk.id)

    def delete_files(self, repo_id: str, paths: Iterable[str]) -> int:
        removed = 0
        with self._lock:
            for path in paths:
                ids = self._by_file.pop((repo_id, path), [])
                for chunk_id in ids:
                    self._chunks[repo_id].pop(chunk_id, None)
                self._bm25[repo_id].remove_many(ids)
                removed += len(ids)
        return removed

    def clear_chunks(self, repo_id: str) -> None:
        with self._lock:
            self._chunks.pop(repo_id, None)
            self._bm25.pop(repo_id, None)
            for key in [k for k in self._by_file if k[0] == repo_id]:
                del self._by_file[key]

    def chunks_for_file(self, repo_id: str, path: str) -> list[Chunk]:
        with self._lock:
            ids = self._by_file.get((repo_id, path), [])
            chunks = [self._chunks[repo_id][i] for i in ids]
        return sorted(chunks, key=lambda c: c.start_line)

    def indexed_paths(self, repo_id: str) -> set[str]:
        with self._lock:
            return {path for (rid, path), ids in self._by_file.items() if rid == repo_id and ids}

    def count_chunks(self, repo_id: str) -> int:
        with self._lock:
            return len(self._chunks.get(repo_id, {}))

    # search -----------------------------------------------------------------------

    def dense_search(self, repo_id: str, vector: Sequence[float], k: int) -> list[ScoredChunk]:
        with self._lock:
            candidates = [c for c in self._chunks.get(repo_id, {}).values() if c.embedding]
        scored = [(cosine(vector, c.embedding or []), c) for c in candidates]
        scored.sort(key=lambda sc: (-sc[0], sc[1].id))
        return [ScoredChunk(chunk=c, score=s) for s, c in scored[:k]]

    def lexical_search(self, repo_id: str, query: str, k: int) -> list[ScoredChunk]:
        with self._lock:
            hits = self._bm25[repo_id].search(query, k) if repo_id in self._bm25 else []
            chunks = self._chunks.get(repo_id, {})
            return [ScoredChunk(chunk=chunks[i], score=s) for i, s in hits if i in chunks]
