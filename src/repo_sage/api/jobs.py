"""Tracks indexing state per repository and serialises indexing runs."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from repo_sage.indexer import Indexer, IndexReport

log = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexStatus:
    state: str = "pending"
    error: str | None = None
    report: IndexReport | None = None


class IndexJobs:
    def __init__(self, indexer: Indexer) -> None:
        self._indexer = indexer
        self._status: dict[str, IndexStatus] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def status(self, repo_id: str, *, has_index: bool = False) -> IndexStatus:
        with self._guard:
            current = self._status.get(repo_id)
        if current is not None:
            return current
        # Repos indexed by a previous process (Postgres store) are ready on startup.
        return IndexStatus(state="ready" if has_index else "pending")

    def is_running(self, repo_id: str) -> bool:
        return self.status(repo_id).state == "indexing"

    def mark_queued(self, repo_id: str) -> None:
        with self._guard:
            prev = self._status.get(repo_id)
            self._status[repo_id] = IndexStatus(
                state="indexing", report=prev.report if prev else None
            )

    def run(self, repo_id: str, full: bool = False) -> None:
        with self._guard:
            lock = self._locks.setdefault(repo_id, threading.Lock())
        with lock:
            self.mark_queued(repo_id)
            try:
                report = self._indexer.index(repo_id, full=full, progress=log.info)
            except Exception as exc:  # surfaced to clients through the status endpoint
                log.exception("indexing %s failed", repo_id)
                with self._guard:
                    self._status[repo_id] = IndexStatus(state="failed", error=str(exc))
                return
            with self._guard:
                self._status[repo_id] = IndexStatus(state="ready", report=report)

    def forget(self, repo_id: str) -> None:
        with self._guard:
            self._status.pop(repo_id, None)
            self._locks.pop(repo_id, None)
