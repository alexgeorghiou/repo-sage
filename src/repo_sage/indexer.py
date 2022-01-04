"""Full and incremental indexing of a repository into a :class:`Store`.

Incremental indexing keys off the last indexed commit SHA stored on the repo record:
``git diff --name-status <last>..HEAD`` tells us exactly which files to re-chunk and
which to drop, so a one-line commit to a large repo costs one file's worth of
embedding calls instead of the whole tree.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from repo_sage.chunking.base import Chunker
from repo_sage.chunking.code import CodeChunker
from repo_sage.embeddings.base import Embedder
from repo_sage.ingest.filters import FileFilter
from repo_sage.ingest.git_source import GitSource, is_remote, repo_name_from_source, slugify
from repo_sage.models import Chunk, Repo
from repo_sage.store.base import Store

log = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


class IndexMode(StrEnum):
    FULL = "full"
    INCREMENTAL = "incremental"
    UP_TO_DATE = "up-to-date"


@dataclass(slots=True)
class IndexReport:
    repo_id: str
    mode: IndexMode
    commit: str | None
    previous_commit: str | None = None
    files_indexed: int = 0
    files_deleted: int = 0
    chunks_written: int = 0
    skipped: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        at = self.commit[:10] if self.commit else "working tree"
        return (
            f"{self.repo_id}: {self.mode.value} @ {at} - "
            f"{self.files_indexed} files indexed, {self.files_deleted} removed, "
            f"{self.chunks_written} chunks written, {len(self.skipped)} skipped"
        )


class RepoNotFoundError(KeyError):
    pass


@dataclass(slots=True)
class Indexer:
    store: Store
    embedder: Embedder
    data_dir: Path
    chunker: Chunker = field(default_factory=CodeChunker)
    max_file_bytes: int = 512_000
    embed_batch: int = 128

    # registration -----------------------------------------------------------------

    def register(self, source: str, *, name: str | None = None, branch: str | None = None) -> Repo:
        """Create (or return the existing) repo record for ``source``. Does not index."""
        if not is_remote(source):
            if not Path(source).is_dir():
                raise FileNotFoundError(f"local repository not found: {source}")
            source = str(Path(source).resolve())
        display = name or repo_name_from_source(source)
        repo_id = slugify(display)
        existing = self.store.get_repo(repo_id)
        if existing is not None:
            return existing
        repo = Repo(id=repo_id, name=display, source=source, branch=branch)
        self.store.upsert_repo(repo)
        return repo

    def source_for(self, repo: Repo) -> GitSource:
        return GitSource(repo.source, self.data_dir / "repos" / repo.id, repo.branch)

    # indexing ---------------------------------------------------------------------

    def index(
        self, repo_id: str, *, full: bool = False, progress: ProgressFn | None = None
    ) -> IndexReport:
        repo = self.store.get_repo(repo_id)
        if repo is None:
            raise RepoNotFoundError(repo_id)
        notify = progress or (lambda _msg: None)

        src = self.source_for(repo)
        src.sync()
        head = src.head()
        filt = FileFilter(src.root, max_bytes=self.max_file_bytes)

        mode, report = self._plan(repo, src, head, full)
        if mode is IndexMode.UP_TO_DATE:
            notify(f"{repo.name} is up to date at {head and head[:10]}")
            return report

        if mode is IndexMode.FULL:
            notify(f"full index of {repo.name}")
            self.store.clear_chunks(repo.id)
            tracked = src.tracked_files()
            to_index = tracked if tracked is not None else filt.candidate_paths()
        else:
            assert repo.last_commit is not None and head is not None
            changes = src.diff(repo.last_commit, head)
            notify(
                f"incremental index {repo.last_commit[:10]}..{head[:10]}: "
                f"{len(changes.changed)} changed, {len(changes.deleted)} deleted"
            )
            report.files_deleted = len(changes.deleted & self.store.indexed_paths(repo.id))
            self.store.delete_files(repo.id, changes.deleted)
            to_index = sorted(changes.changed)

        self._index_paths(repo.id, filt, to_index, report, notify)
        report.skipped = dict(filt.skipped)

        repo.mark_indexed(
            head,
            files=len(self.store.indexed_paths(repo.id)),
            chunks=self.store.count_chunks(repo.id),
        )
        self.store.upsert_repo(repo)
        notify(report.summary())
        return report

    def _plan(
        self, repo: Repo, src: GitSource, head: str | None, full: bool
    ) -> tuple[IndexMode, IndexReport]:
        prev = repo.last_commit
        if full or head is None or prev is None or not src.has_commit(prev):
            mode = IndexMode.FULL
        elif prev == head:
            mode = IndexMode.UP_TO_DATE
        else:
            changes = src.diff(prev, head)
            # .gitignore edits can change which *unchanged* files are eligible.
            touched = changes.changed | changes.deleted
            mode = (
                IndexMode.FULL
                if any(p.rsplit("/", 1)[-1] == ".gitignore" for p in touched)
                else IndexMode.INCREMENTAL
            )
        return mode, IndexReport(repo_id=repo.id, mode=mode, commit=head, previous_commit=prev)

    def _index_paths(
        self,
        repo_id: str,
        filt: FileFilter,
        paths: list[str],
        report: IndexReport,
        notify: ProgressFn,
    ) -> None:
        pending: list[tuple[str, list[Chunk]]] = []
        pending_chunks = 0
        for n, path in enumerate(paths, start=1):
            source = filt.load(path)
            if source is None:
                # A file that became ignored, binary or too large must leave the index.
                self.store.delete_files(repo_id, [path])
                continue
            chunks = self.chunker.chunk(repo_id, source)
            pending.append((path, chunks))
            pending_chunks += len(chunks)
            if pending_chunks >= self.embed_batch:
                self._flush(repo_id, pending, report)
                pending, pending_chunks = [], 0
                notify(f"  {n}/{len(paths)} files")
        self._flush(repo_id, pending, report)

    def _flush(
        self, repo_id: str, pending: list[tuple[str, list[Chunk]]], report: IndexReport
    ) -> None:
        all_chunks = [c for _, chunks in pending for c in chunks]
        if all_chunks:
            vectors = self.embedder.embed([c.search_text() for c in all_chunks])
            for chunk, vec in zip(all_chunks, vectors, strict=True):
                chunk.embedding = vec
        for path, chunks in pending:
            self.store.replace_file(repo_id, path, chunks)
            report.files_indexed += 1
            report.chunks_written += len(chunks)
