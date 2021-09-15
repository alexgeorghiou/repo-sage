"""OpenAI embeddings (``text-embedding-3-*``). Requires the ``openai`` extra."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from repo_sage.embeddings.base import normalize


@dataclass(slots=True)
class OpenAIEmbedder:
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 96
    max_chars: int = 24_000
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()

    @property
    def dim(self) -> int:
        return self.dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[: self.max_chars] or " " for t in texts[i : i + self.batch_size]]
            resp = self._client.embeddings.create(
                model=self.model, input=batch, dimensions=self.dimension
            )
            out.extend(normalize(list(d.embedding)) for d in resp.data)
        return out
