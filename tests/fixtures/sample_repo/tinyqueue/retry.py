"""Retry policy with exponential backoff and full jitter."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 60.0

    def should_retry(self, attempts: int) -> bool:
        return attempts < self.max_attempts

    def backoff(self, attempts: int, rng: random.Random | None = None) -> float:
        """Delay before the next attempt: uniform in [0, min(max, base * 2**attempts)].

        Full jitter spreads retries of jobs that failed together, avoiding a
        thundering herd against the dependency that caused the failure.
        """
        rng = rng or random.Random()
        ceiling = min(self.max_delay, self.base_delay * (2 ** attempts))
        return rng.uniform(0, ceiling)
