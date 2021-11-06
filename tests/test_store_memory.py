from repo_sage.embeddings import HashingEmbedder
from repo_sage.models import Repo
from repo_sage.retrieval.bm25 import BM25Index
from repo_sage.store import InMemoryStore


def test_bm25_prefers_rare_terms_and_supports_removal():
    idx = BM25Index()
    idx.add("1", "parse config file loader")
    idx.add("2", "config value config default")
    idx.add("3", "retry backoff jitter")
    assert idx.search("jitter")[0][0] == "3"
    assert idx.search("config")[0][0] == "2"
    idx.remove("3")
    assert idx.search("jitter") == []
    assert len(idx) == 2


def test_bm25_splits_identifiers():
    idx = BM25Index()
    idx.add("a", "def computeRetryDelay(attempt): ...")
    idx.add("b", "def render_page(): ...")
    assert idx.search("retry delay")[0][0] == "a"


def test_memory_store_replace_and_delete(make_chunk):
    store = InMemoryStore()
    store.upsert_repo(Repo(id="r", name="r", source="."))
    emb = HashingEmbedder(dimension=64)
    c1 = make_chunk(path="a.py", start=1, end=5, text="def alpha(): pass")
    c2 = make_chunk(path="b.py", start=1, end=5, text="def beta(): pass")
    for c in (c1, c2):
        c.embedding = emb.embed([c.text])[0]
        store.replace_file("r", c.path, [c])
    assert store.indexed_paths("r") == {"a.py", "b.py"}

    c1b = make_chunk(path="a.py", start=1, end=9, text="def gamma(): pass")
    c1b.embedding = emb.embed([c1b.text])[0]
    store.replace_file("r", "a.py", [c1b])
    assert [c.text for c in store.chunks_for_file("r", "a.py")] == ["def gamma(): pass"]
    assert store.lexical_search("r", "alpha", 5) == []
    assert store.lexical_search("r", "gamma", 5)[0].chunk.path == "a.py"

    assert store.delete_files("r", ["b.py"]) == 1
    assert store.count_chunks("r") == 1
    top = store.dense_search("r", emb.embed(["gamma"])[0], 3)
    assert top[0].chunk.path == "a.py"


def test_memory_store_isolates_repos(make_chunk):
    store = InMemoryStore()
    store.replace_file("one", "x.py", [make_chunk(repo_id="one", path="x.py", text="shared")])
    store.replace_file("two", "x.py", [make_chunk(repo_id="two", path="x.py", text="shared")])
    store.clear_chunks("one")
    assert store.count_chunks("one") == 0
    assert store.count_chunks("two") == 1
