"""Resolve a repository source (remote URL or local path) to a working tree and diff commits."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import git

log = logging.getLogger(__name__)

_REMOTE_RE = re.compile(r"^(https?://|git@|ssh://|git://)")


def is_remote(source: str) -> bool:
    return bool(_REMOTE_RE.match(source))


def repo_name_from_source(source: str) -> str:
    """``https://github.com/org/proj.git`` -> ``org/proj``; local paths -> directory name."""
    if is_remote(source):
        tail = re.sub(r"\.git$", "", source.rstrip("/"))
        parts = re.split(r"[/:]", tail)
        return "/".join(parts[-2:])
    return Path(source).resolve().name


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "repo"


@dataclass(slots=True)
class ChangeSet:
    """Files touched between two commits, relative to the repository root."""

    changed: set[str] = field(default_factory=set)
    deleted: set[str] = field(default_factory=set)

    @property
    def empty(self) -> bool:
        return not self.changed and not self.deleted


class GitSource:
    """A working tree we can read files from and diff between commits.

    Remote sources are cloned into ``workdir``; local sources are used in place.
    A local directory that is not a git repository is still indexable, but every
    re-index is a full one because there is no history to diff against.
    """

    def __init__(self, source: str, workdir: Path, branch: str | None = None) -> None:
        self.source = source
        self.branch = branch
        if is_remote(source):
            self.root = workdir
        else:
            self.root = Path(source).resolve()
            if not self.root.is_dir():
                raise FileNotFoundError(f"local repository not found: {source}")
        self._repo: git.Repo | None = None

    @property
    def repo(self) -> git.Repo | None:
        if self._repo is None:
            try:
                self._repo = git.Repo(self.root)
            except (git.InvalidGitRepositoryError, git.NoSuchPathError):
                return None
        return self._repo

    def sync(self) -> None:
        """Clone or fast-forward remote sources; local sources are left untouched."""
        if not is_remote(self.source):
            return
        if not (self.root / ".git").exists():
            log.info("cloning %s into %s", self.source, self.root)
            self.root.parent.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, object] = {"depth": 50}
            if self.branch:
                kwargs["branch"] = self.branch
            self._repo = git.Repo.clone_from(self.source, self.root, **kwargs)
            return
        repo = self.repo
        assert repo is not None
        log.info("fetching %s", self.source)
        repo.remotes.origin.fetch()
        ref = self.branch or repo.active_branch.name
        repo.git.reset("--hard", f"origin/{ref}")

    def head(self) -> str | None:
        repo = self.repo
        if repo is None:
            return None
        try:
            return repo.head.commit.hexsha
        except ValueError:  # repository without commits
            return None

    def tracked_files(self) -> list[str] | None:
        """Files tracked at HEAD, or ``None`` when the source has no git history."""
        repo = self.repo
        if repo is None or self.head() is None:
            return None
        out = repo.git.ls_files("-z")
        return sorted(p for p in out.split("\x00") if p)

    def diff(self, old: str, new: str) -> ChangeSet:
        """Compute changed and deleted paths from ``old`` to ``new``.

        Renames are reported as a delete of the old path plus a change of the new one,
        which is exactly what the index needs.
        """
        repo = self.repo
        if repo is None:
            raise RuntimeError("diff requested on a source without git history")
        cs = ChangeSet()
        out = repo.git.diff("--name-status", "-M", "-z", old, new)
        tokens = [t for t in out.split("\x00") if t]
        i = 0
        while i < len(tokens):
            status = tokens[i]
            code = status[0]
            if code in {"R", "C"}:
                src, dst = tokens[i + 1], tokens[i + 2]
                if code == "R":
                    cs.deleted.add(src)
                cs.changed.add(dst)
                i += 3
                continue
            path = tokens[i + 1]
            if code == "D":
                cs.deleted.add(path)
            else:  # A, M, T, U
                cs.changed.add(path)
            i += 2
        return cs

    def has_commit(self, sha: str) -> bool:
        """False when history was rewritten (force-push, shallow clone) and ``sha`` is gone."""
        repo = self.repo
        if repo is None:
            return False
        try:
            repo.commit(sha)
        except (ValueError, git.BadName, git.GitCommandError):
            return False
        return True
