from __future__ import annotations

from pathlib import Path

import git
import pytest

from repo_sage.embeddings import HashingEmbedder
from repo_sage.indexer import Indexer, IndexMode, RepoNotFoundError
from repo_sage.store import InMemoryStore


class SpyStore(InMemoryStore):
    """Records which files were (re)written so tests can assert on incremental work."""

    def __init__(self) -> None:
        super().__init__()
        self.written: list[str] = []

    def replace_file(self, repo_id, path, chunks):  # type: ignore[override]
        self.written.append(path)
        super().replace_file(repo_id, path, chunks)


def _commit(repo: git.Repo, message: str) -> str:
    repo.git.add("-A")
    repo.index.commit(message)
    return repo.head.commit.hexsha


@pytest.fixture
def workspace(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    repo = git.Repo.init(root)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test")
        cw.set_value("user", "email", "test@example.com")
    (root / "app.py").write_text("def greet(name):\n    return f'hi {name}'\n")
    (root / "util.py").write_text("def slug(text):\n    return text.lower()\n")
    (root / "docs.md").write_text("# Docs\n\nUsage notes.\n")
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    _commit(repo, "initial")
    store = SpyStore()
    indexer = Indexer(store=store, embedder=HashingEmbedder(64), data_dir=tmp_path / "data")
    record = indexer.register(str(root), name="proj")
    return root, repo, store, indexer, record.id


def test_first_index_is_full_and_skips_binaries(workspace):
    _, repo, store, indexer, repo_id = workspace
    report = indexer.index(repo_id)
    assert report.mode is IndexMode.FULL
    assert report.commit == repo.head.commit.hexsha
    assert store.indexed_paths(repo_id) == {"app.py", "util.py", "docs.md"}
    assert report.skipped == {"logo.png": "binary"}
    saved = store.get_repo(repo_id)
    assert saved is not None and saved.last_commit == report.commit and saved.file_count == 3


def test_reindex_without_new_commits_is_a_noop(workspace):
    _, _, store, indexer, repo_id = workspace
    indexer.index(repo_id)
    store.written.clear()
    report = indexer.index(repo_id)
    assert report.mode is IndexMode.UP_TO_DATE
    assert store.written == []


def test_incremental_reindex_only_touches_changed_files(workspace):
    root, repo, store, indexer, repo_id = workspace
    first = indexer.index(repo_id)

    (root / "app.py").write_text("def greet(name):\n    return f'hello {name}!'\n")
    (root / "util.py").unlink()
    (root / "new_mod.py").write_text("def fresh():\n    return 42\n")
    second_sha = _commit(repo, "edit, delete, add")
    store.written.clear()

    report = indexer.index(repo_id)
    assert report.mode is IndexMode.INCREMENTAL
    assert report.previous_commit == first.commit
    assert report.commit == second_sha
    assert sorted(store.written) == ["app.py", "new_mod.py"]
    assert report.files_deleted == 1
    assert store.indexed_paths(repo_id) == {"app.py", "docs.md", "new_mod.py"}
    assert "hello" in store.chunks_for_file(repo_id, "app.py")[0].text


def test_renames_drop_old_path(workspace):
    _, repo, store, indexer, repo_id = workspace
    indexer.index(repo_id)
    repo.git.mv("util.py", "helpers.py")
    _commit(repo, "rename")
    report = indexer.index(repo_id)
    assert report.mode is IndexMode.INCREMENTAL
    assert "util.py" not in store.indexed_paths(repo_id)
    assert store.chunks_for_file(repo_id, "helpers.py")[0].symbol == "slug"


def test_gitignore_change_forces_full_reindex(workspace):
    root, repo, store, indexer, repo_id = workspace
    indexer.index(repo_id)
    (root / ".gitignore").write_text("docs.md\n")
    _commit(repo, "ignore docs")
    report = indexer.index(repo_id)
    assert report.mode is IndexMode.FULL
    assert "docs.md" not in store.indexed_paths(repo_id)


def test_rewritten_history_falls_back_to_full(workspace):
    _, _, store, indexer, repo_id = workspace
    indexer.index(repo_id)
    record = store.get_repo(repo_id)
    assert record is not None
    record.last_commit = "0" * 40  # no longer reachable, e.g. after a force-push
    store.upsert_repo(record)
    assert indexer.index(repo_id).mode is IndexMode.FULL


def test_force_full_and_unknown_repo(workspace):
    _, _, _, indexer, repo_id = workspace
    indexer.index(repo_id)
    assert indexer.index(repo_id, full=True).mode is IndexMode.FULL
    with pytest.raises(RepoNotFoundError):
        indexer.index("nope")


def test_plain_directory_without_git_is_indexed(tmp_path: Path):
    root = tmp_path / "plain"
    (root / "node_modules" / "dep").mkdir(parents=True)
    (root / "node_modules" / "dep" / "index.js").write_text("module.exports = 1\n")
    (root / "main.go").write_text("package main\n\nfunc main() {}\n")
    store = InMemoryStore()
    indexer = Indexer(store=store, embedder=HashingEmbedder(32), data_dir=tmp_path / "d")
    repo = indexer.register(str(root))
    report = indexer.index(repo.id)
    assert report.mode is IndexMode.FULL and report.commit is None
    assert store.indexed_paths(repo.id) == {"main.go"}
