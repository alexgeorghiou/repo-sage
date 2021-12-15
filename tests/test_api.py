from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from repo_sage.api import create_app
from repo_sage.config import Settings
from repo_sage.llm import FakeLLM
from repo_sage.services import Services

SAMPLE = Path(__file__).parent / "fixtures" / "sample_repo"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(data_dir=tmp_path, store="memory", embedder="hashing", llm="fake")
    services = Services.build(settings, llm=FakeLLM())
    return TestClient(create_app(services, settings))


@pytest.fixture
def indexed(client: TestClient) -> str:
    resp = client.post("/api/repos", json={"source": str(SAMPLE), "name": "tinyqueue"})
    assert resp.status_code == 202
    return resp.json()["id"]


def test_health(client: TestClient):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm"] == "fake"


def test_register_indexes_in_background_and_lists(client: TestClient, indexed: str):
    repo = client.get(f"/api/repos/{indexed}").json()
    assert repo["state"] == "ready"
    assert repo["file_count"] >= 8
    assert repo["chunk_count"] > repo["file_count"]
    assert repo["last_report"]["mode"] == "full"
    assert [r["id"] for r in client.get("/api/repos").json()] == ["tinyqueue"]


def test_register_rejects_missing_path(client: TestClient, tmp_path: Path):
    resp = client.post("/api/repos", json={"source": str(tmp_path / "nope")})
    assert resp.status_code == 400


def test_ask_returns_validated_citations(client: TestClient, indexed: str):
    resp = client.post(
        f"/api/repos/{indexed}/ask",
        json={"question": "How does the retry backoff use jitter?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["invalid_citations"] == []
    assert body["citations"][0]["path"] == "tinyqueue/retry.py"
    cited = body["citations"][0]
    assert f"[{cited['path']}:{cited['start_line']}-{cited['end_line']}]" in body["answer"]
    assert body["sources"] and {"dense", "lexical"} & set(body["sources"][0]["retrievers"])


def test_ask_flags_hallucinated_citations(tmp_path: Path):
    settings = Settings(data_dir=tmp_path)
    llm = FakeLLM(scripted=["It lives in [tinyqueue/missing.py:1-5]."])
    client = TestClient(create_app(Services.build(settings, llm=llm), settings))
    repo_id = client.post("/api/repos", json={"source": str(SAMPLE)}).json()["id"]
    body = client.post(f"/api/repos/{repo_id}/ask", json={"question": "where?"}).json()
    assert body["grounded"] is False
    assert body["invalid_citations"][0]["path"] == "tinyqueue/missing.py"


def test_ask_stream_emits_sources_tokens_and_answer(client: TestClient, indexed: str):
    with client.stream(
        "POST", f"/api/repos/{indexed}/ask/stream", json={"question": "periodic scheduler tick"}
    ) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())
    events = [
        line.removeprefix("event: ").strip()
        for line in raw.splitlines()
        if line.startswith("event:")
    ]
    assert events[0] == "sources"
    assert "token" in events
    assert events[-1] == "answer"
    data_lines = [ln.removeprefix("data: ") for ln in raw.splitlines() if ln.startswith("data:")]
    final = json.loads(data_lines[-1])
    assert final["citations"]


def test_reindex_is_up_to_date_for_plain_directory_full_otherwise(client: TestClient, indexed: str):
    resp = client.post(f"/api/repos/{indexed}/reindex", json={"full": True})
    assert resp.status_code == 202
    repo = client.get(f"/api/repos/{indexed}").json()
    assert repo["last_report"]["mode"] == "full"


def test_search_and_snippet(client: TestClient, indexed: str):
    hits = client.get(f"/api/repos/{indexed}/search", params={"q": "visibility timeout"}).json()
    assert hits[0]["path"] == "tinyqueue/queue.py"
    snip = client.get(
        f"/api/repos/{indexed}/snippet",
        params={"path": "tinyqueue/retry.py", "start": 18, "end": 20, "context": 0},
    ).json()
    assert snip["start_line"] == 18 and snip["end_line"] == 20
    assert "def backoff" in snip["content"]


def test_snippet_blocks_path_traversal(client: TestClient, indexed: str):
    resp = client.get(f"/api/repos/{indexed}/snippet", params={"path": "../../pyproject.toml"})
    assert resp.status_code == 400


def test_unknown_repo_and_delete(client: TestClient, indexed: str):
    assert client.get("/api/repos/nope").status_code == 404
    assert client.post("/api/repos/nope/ask", json={"question": "x"}).status_code == 404
    assert client.delete(f"/api/repos/{indexed}").status_code == 204
    assert client.get("/api/repos").json() == []
