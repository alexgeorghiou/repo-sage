"""Postgres + pgvector store.

Dense search uses an HNSW index with cosine distance. Lexical search uses Postgres
full-text search over a pre-tokenised column: identifiers are split into sub-words
in Python first (``getUserName`` -> ``get user name``), so the ``simple`` text search
configuration is enough and behaves the same as the in-memory BM25 tokeniser.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from repo_sage.models import Chunk, ChunkKind, Repo, ScoredChunk
from repo_sage.text import tokenize

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS repos (
    id          text PRIMARY KEY,
    name        text NOT NULL,
    source      text NOT NULL,
    branch      text,
    last_commit text,
    indexed_at  timestamptz,
    chunk_count integer NOT NULL DEFAULT 0,
    file_count  integer NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id          text PRIMARY KEY,
    repo_id     text NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    path        text NOT NULL,
    start_line  integer NOT NULL,
    end_line    integer NOT NULL,
    language    text,
    kind        text NOT NULL,
    symbol      text,
    content     text NOT NULL,
    lexemes     text NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('simple', lexemes)) STORED,
    embedding   vector({dim})
);

CREATE INDEX IF NOT EXISTS chunks_repo_path_idx ON chunks (repo_id, path);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""

_CHUNK_COLS = "id, repo_id, path, start_line, end_line, language, kind, symbol, content"


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        repo_id=row["repo_id"],
        path=row["path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        text=row["content"],
        language=row["language"],
        kind=ChunkKind(row["kind"]),
        symbol=row["symbol"],
    )


def _row_to_repo(row: dict[str, Any]) -> Repo:
    return Repo(**row)


class PostgresStore:
    def __init__(self, dsn: str, dim: int, *, min_size: int = 1, max_size: int = 5) -> None:
        self.dim = dim
        self._pool = ConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            configure=self._configure,
            open=False,
        )
        self._bootstrap(dsn)
        self._pool.open(wait=True)

    def _bootstrap(self, dsn: str) -> None:
        # The extension must exist before register_vector can find the type OID.
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(SCHEMA.format(dim=int(self.dim)))

    @staticmethod
    def _configure(conn: psycopg.Connection[Any]) -> None:
        register_vector(conn)

    def close(self) -> None:
        self._pool.close()

    # repositories -----------------------------------------------------------------

    def upsert_repo(self, repo: Repo) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO repos (id, name, source, branch, last_commit, indexed_at,
                                   chunk_count, file_count)
                VALUES (%(id)s, %(name)s, %(source)s, %(branch)s, %(last_commit)s,
                        %(indexed_at)s, %(chunk_count)s, %(file_count)s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name, source = EXCLUDED.source, branch = EXCLUDED.branch,
                    last_commit = EXCLUDED.last_commit, indexed_at = EXCLUDED.indexed_at,
                    chunk_count = EXCLUDED.chunk_count, file_count = EXCLUDED.file_count
                """,
                {
                    "id": repo.id,
                    "name": repo.name,
                    "source": repo.source,
                    "branch": repo.branch,
                    "last_commit": repo.last_commit,
                    "indexed_at": repo.indexed_at,
                    "chunk_count": repo.chunk_count,
                    "file_count": repo.file_count,
                },
            )

    def get_repo(self, repo_id: str) -> Repo | None:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM repos WHERE id = %s", (repo_id,)).fetchone()
        return _row_to_repo(row) if row else None

    def list_repos(self) -> list[Repo]:
        with self._pool.connection() as conn:
            rows = conn.execute("SELECT * FROM repos ORDER BY name").fetchall()
        return [_row_to_repo(r) for r in rows]

    def delete_repo(self, repo_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM repos WHERE id = %s", (repo_id,))

    # chunks -----------------------------------------------------------------------

    def replace_file(self, repo_id: str, path: str, chunks: Sequence[Chunk]) -> None:
        rows = [
            (
                c.id,
                c.repo_id,
                c.path,
                c.start_line,
                c.end_line,
                c.language,
                c.kind.value,
                c.symbol,
                c.text,
                " ".join(tokenize(c.search_text())),
                c.embedding,
            )
            for c in chunks
        ]
        with self._pool.connection() as conn, conn.transaction():
            conn.execute("DELETE FROM chunks WHERE repo_id = %s AND path = %s", (repo_id, path))
            if rows:
                with conn.cursor() as cur:
                    cur.executemany(
                        f"""
                        INSERT INTO chunks ({_CHUNK_COLS}, lexemes, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content, lexemes = EXCLUDED.lexemes,
                            embedding = EXCLUDED.embedding
                        """,
                        rows,
                    )

    def delete_files(self, repo_id: str, paths: Iterable[str]) -> int:
        path_list = list(paths)
        if not path_list:
            return 0
        with self._pool.connection() as conn:
            cur = conn.execute(
                "DELETE FROM chunks WHERE repo_id = %s AND path = ANY(%s)", (repo_id, path_list)
            )
            return cur.rowcount

    def clear_chunks(self, repo_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE repo_id = %s", (repo_id,))

    def chunks_for_file(self, repo_id: str, path: str) -> list[Chunk]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT {_CHUNK_COLS} FROM chunks WHERE repo_id = %s AND path = %s "
                "ORDER BY start_line",
                (repo_id, path),
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def indexed_paths(self, repo_id: str) -> set[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT path FROM chunks WHERE repo_id = %s", (repo_id,)
            ).fetchall()
        return {r["path"] for r in rows}

    def count_chunks(self, repo_id: str) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT count(*) AS n FROM chunks WHERE repo_id = %s", (repo_id,)
            ).fetchone()
        return int(row["n"]) if row else 0

    # search -----------------------------------------------------------------------

    def dense_search(self, repo_id: str, vector: Sequence[float], k: int) -> list[ScoredChunk]:
        import numpy as np

        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT {_CHUNK_COLS}, 1 - (embedding <=> %(v)s) AS score
                FROM chunks
                WHERE repo_id = %(repo)s AND embedding IS NOT NULL
                ORDER BY embedding <=> %(v)s
                LIMIT %(k)s
                """,
                {"v": np.asarray(vector, dtype=np.float32), "repo": repo_id, "k": k},
            ).fetchall()
        return [ScoredChunk(chunk=_row_to_chunk(r), score=float(r["score"])) for r in rows]

    def lexical_search(self, repo_id: str, query: str, k: int) -> list[ScoredChunk]:
        terms = sorted(set(tokenize(query)))
        if not terms:
            return []
        # Tokens are [a-z0-9_] only, so joining with OR is a safe tsquery.
        tsquery = " | ".join(terms)
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT {_CHUNK_COLS}, ts_rank_cd(tsv, q, 1) AS score
                FROM chunks, to_tsquery('simple', %(q)s) AS q
                WHERE repo_id = %(repo)s AND tsv @@ q
                ORDER BY score DESC, id
                LIMIT %(k)s
                """,
                {"q": tsquery, "repo": repo_id, "k": k},
            ).fetchall()
        return [ScoredChunk(chunk=_row_to_chunk(r), score=float(r["score"])) for r in rows]
