"""Append-only JSON-lines persistence so a restarted process can recover its jobs."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class JsonlStorage:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, job_id: str, **fields: Any) -> None:
        record = {"event": event, "job_id": job_id, **fields}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def replay(self) -> Iterator[dict[str, Any]]:
        """Yield jobs that were enqueued but never acknowledged."""
        pending: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return iter(())
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if record["event"] == "enqueued":
                    pending[record["job_id"]] = record
                elif record["event"] == "acked":
                    pending.pop(record["job_id"], None)
        return iter(pending.values())

    def compact(self) -> int:
        """Rewrite the log keeping only pending jobs. Returns the number kept."""
        pending = list(self.replay())
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for record in pending:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        tmp.replace(self.path)
        return len(pending)
