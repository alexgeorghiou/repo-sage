from __future__ import annotations

from repo_sage.config import Settings
from repo_sage.embeddings.base import Embedder
from repo_sage.embeddings.hashing import HashingEmbedder


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "openai":
        from repo_sage.embeddings.openai import OpenAIEmbedder

        return OpenAIEmbedder(model=settings.embedding_model, dimension=settings.embedding_dim)
    return HashingEmbedder(dimension=settings.embedding_dim)
