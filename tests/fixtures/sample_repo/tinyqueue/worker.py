"""Worker loop: dequeue, run handler, ack or retry. Drains gracefully on SIGTERM."""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from collections.abc import Callable
from typing import Any

from tinyqueue.queue import Job, JobQueue, QueueEmpty
from tinyqueue.retry import RetryPolicy

log = logging.getLogger("tinyqueue.worker")
Handler = Callable[[dict[str, Any]], None]


class Worker:
    def __init__(self, queue: JobQueue, policy: RetryPolicy | None = None) -> None:
        self.queue = queue
        self.policy = policy or RetryPolicy()
        self.handlers: dict[str, Handler] = {}
        self._stopping = threading.Event()
        self.dead_letters: list[Job] = []

    def register(self, name: str) -> Callable[[Handler], Handler]:
        def decorator(fn: Handler) -> Handler:
            self.handlers[name] = fn
            return fn

        return decorator

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        signal.signal(signal.SIGINT, lambda *_: self.stop())

    def stop(self) -> None:
        log.info("stop requested; draining in-flight job")
        self._stopping.set()

    def run_once(self) -> bool:
        try:
            job = self.queue.dequeue()
        except QueueEmpty:
            return False
        handler = self.handlers.get(job.name)
        if handler is None:
            log.error("no handler for %s; dead-lettering", job.name)
            self.queue.ack(job.id)
            self.dead_letters.append(job)
            return True
        try:
            handler(job.payload)
        except Exception:
            if self.policy.should_retry(job.attempts):
                delay = self.policy.backoff(job.attempts)
                log.warning("job %s failed (attempt %d); retrying in %.1fs", job.id, job.attempts, delay)
                self.queue.nack(job.id, delay)
            else:
                log.error("job %s exhausted retries; dead-lettering", job.id)
                self.queue.ack(job.id)
                self.dead_letters.append(job)
        else:
            self.queue.ack(job.id)
        return True

    def run_forever(self, idle_sleep: float = 0.2) -> None:
        while not self._stopping.is_set():
            if not self.run_once():
                time.sleep(idle_sleep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tinyqueue worker.")
    parser.add_argument("--queue", default="default")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    worker = Worker(JobQueue())
    worker.install_signal_handlers()
    log.info("worker for queue %s with concurrency %d", args.queue, args.concurrency)
    worker.run_forever()


if __name__ == "__main__":
    main()
