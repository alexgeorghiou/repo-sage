from __future__ import annotations

from typing import Protocol

from repo_sage.models import Chunk, SourceFile


class Chunker(Protocol):
    def chunk(self, repo_id: str, source: SourceFile) -> list[Chunk]: ...
