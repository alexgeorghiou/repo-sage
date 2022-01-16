"""Integration tests for the pgvector store. Set REPO_SAGE_TEST_DSN to run them, e.g.

docker compose up -d db
REPO_SAGE_TEST_DSN=postgresql://sage:sage@localhost:5432/sage pytest -m postgres
"""

from __future__ import annotations

import os
import uuid

import pytest

from repo_sage.embeddings import HashingEmbedder
from repo_sage.models import Repo

DSN = os.environ.get("REPO_SAGE_TEST_DSN")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not DSN, reason="REPO_SAGE_TEST_DSN not set"),
]
DIM = 64


@pytest.fixture
def store():
    from repo_sage.store.postgres import PostgresStore

    s = PostgresStore(DSN or "", dim=DIM)
    yield s
    s.close()


@pytest.fixture
def repo_id(store):
    rid = f"test-{uuid.uuid4().hex[:8]}"
    store.upsert_repo(Repo(id=rid, name=rid, source="."))
    yield rid
    store.delete_repo(rid)


def test_roundtrip_search_and_incremental_replace(store, repo_id, make_chunk):
    emb = HashingEmbedder(DIM)
    a = make_chunk(repo_id=repo_id, path="a.py", text="def computeRetryDelay(): pass", symbol="x")
    b = make_chunk(repo_id=repo_id, path="b.py", text="def render_page(): pass")
    for c in (a, b):
        c.embedding = emb.embed([c.search_text()])[0]
        store.replace_file(repo_id, c.path, [c])

    assert store.indexed_paths(repo_id) == {"a.py", "b.py"}
    assert store.lexical_search(repo_id, "retry delay", 5)[0].chunk.path == "a.py"
    dense = store.dense_search(repo_id, emb.embed(["render page"])[0], 5)
    assert dense[0].chunk.path == "b.py"

    assert store.delete_files(repo_id, ["a.py"]) == 1
    assert store.count_chunks(repo_id) == 1
    repo = store.get_repo(repo_id)
    assert repo is not None and repo.name == repo_id
