from __future__ import annotations

from repo_sage.config import Settings
from repo_sage.store.base import Store
from repo_sage.store.memory import InMemoryStore


def build_store(settings: Settings) -> Store:
    if settings.store == "postgres":
        from repo_sage.store.postgres import PostgresStore

        return PostgresStore(settings.database_url, dim=settings.embedding_dim)
    return InMemoryStore()
