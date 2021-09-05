"""Runtime configuration, read from environment variables prefixed with ``REPO_SAGE_``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REPO_SAGE_", env_file=".env", extra="ignore")

    data_dir: Path = Field(default=Path(".repo-sage"), description="Clones and local state.")

    store: Literal["memory", "postgres"] = "memory"
    database_url: str = "postgresql://sage:sage@localhost:5432/sage"

    embedder: Literal["hashing", "openai"] = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 384

    llm: Literal["fake", "anthropic", "openai"] = "fake"
    llm_model: str | None = None
    llm_max_tokens: int = 1024

    reranker: Literal["lexical", "cross-encoder", "none"] = "lexical"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    max_file_bytes: int = 512_000
    chunk_max_lines: int = 80
    chunk_window_lines: int = 60
    chunk_window_overlap: int = 10

    top_k: int = 8
    candidate_k: int = 40


@lru_cache
def get_settings() -> Settings:
    return Settings()
