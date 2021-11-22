"""In-memory priority queue of jobs with visibility timeouts."""

from __future__ import annotations

import heapq
import itertools
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class Job:
    priority: int
    seq: int
    id: str = field(compare=False)
    name: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)
    attempts: int = field(compare=False, default=0)
    run_at: float = field(compare=False, default=0.0)


class QueueEmpty(Exception):
    """Raised by dequeue when no job is ready."""


class JobQueue:
    """A priority heap of jobs. Lower priority numbers are dequeued first.

    Dequeued jobs become invisible for ``visibility_timeout`` seconds; if they are
    not acknowledged in that window they are put back so another worker can run them.
    """

    def __init__(self, visibility_timeout: float = 30.0) -> None:
        self._heap: list[Job] = []
        self._inflight: dict[str, tuple[Job, float]] = {}
        self._counter = itertools.count()
        self.visibility_timeout = visibility_timeout

    def enqueue(self, name: str, payload: dict[str, Any] | None = None, priority: int = 10,
                delay: float = 0.0) -> Job:
        job = Job(priority=priority, seq=next(self._counter), id=uuid.uuid4().hex, name=name,
                  payload=payload or {}, run_at=time.monotonic() + delay)
        heapq.heappush(self._heap, job)
        return job

    def dequeue(self) -> Job:
        self._requeue_expired()
        now = time.monotonic()
        for _ in range(len(self._heap)):
            job = heapq.heappop(self._heap)
            if job.run_at <= now:
                job.attempts += 1
                self._inflight[job.id] = (job, now + self.visibility_timeout)
                return job
            heapq.heappush(self._heap, job)
            break
        raise QueueEmpty()

    def ack(self, job_id: str) -> None:
        """Mark a job as done so it is never redelivered."""
        self._inflight.pop(job_id, None)

    def nack(self, job_id: str, delay: float) -> None:
        """Return a failed job to the queue after ``delay`` seconds."""
        entry = self._inflight.pop(job_id, None)
        if entry is None:
            return
        job, _ = entry
        job.run_at = time.monotonic() + delay
        job.seq = next(self._counter)
        heapq.heappush(self._heap, job)

    def _requeue_expired(self) -> None:
        now = time.monotonic()
        expired = [jid for jid, (_, deadline) in self._inflight.items() if deadline <= now]
        for jid in expired:
            job, _ = self._inflight.pop(jid)
            heapq.heappush(self._heap, job)

    def __len__(self) -> int:
        return len(self._heap)
