"""Periodic job scheduling on fixed intervals."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from tinyqueue.queue import JobQueue


@dataclass
class PeriodicJob:
    name: str
    every_seconds: float
    next_run: float = 0.0


@dataclass
class Scheduler:
    queue: JobQueue
    jobs: list[PeriodicJob] = field(default_factory=list)

    def every(self, seconds: float, name: str) -> PeriodicJob:
        if seconds <= 0:
            raise ValueError("interval must be positive")
        job = PeriodicJob(name=name, every_seconds=seconds, next_run=time.monotonic())
        self.jobs.append(job)
        return job

    def tick(self, now: float | None = None) -> int:
        """Enqueue every periodic job that is due; returns how many were enqueued.

        Missed intervals are not replayed: if the process was paused for ten
        intervals the job runs once and the next run is scheduled from ``now``.
        """
        now = time.monotonic() if now is None else now
        fired = 0
        for job in self.jobs:
            if job.next_run <= now:
                self.queue.enqueue(job.name, {"scheduled_at": now}, priority=5)
                job.next_run = now + job.every_seconds
                fired += 1
        return fired
