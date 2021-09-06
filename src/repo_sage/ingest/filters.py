"""Decide which files in a working tree are worth indexing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

from repo_sage.ingest.languages import detect_language
from repo_sage.models import SourceFile

# Always skipped, regardless of the repository's own .gitignore.
DEFAULT_IGNORES: tuple[str, ...] = (
    ".git/",
    "node_modules/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "dist/",
    "build/",
    "target/",
    "vendor/",
    ".next/",
    "coverage/",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "uv.lock",
    "go.sum",
    "*.svg",
    "*.ipynb",
)

BINARY_EXTENSIONS = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".pdf", ".zip", ".gz",
        ".tar", ".tgz", ".7z", ".jar", ".class", ".so", ".dll", ".dylib", ".exe", ".bin",
        ".woff", ".woff2", ".ttf", ".otf", ".eot", ".mp3", ".mp4", ".mov", ".wav", ".pyc",
        ".o", ".a", ".wasm", ".parquet", ".sqlite", ".db",
    }
)  # fmt: skip


def looks_binary(data: bytes, sample: int = 8192) -> bool:
    """Heuristic used by git itself: a NUL byte in the first few KB means binary."""
    return b"\x00" in data[:sample]


@dataclass(slots=True)
class FileFilter:
    """Combines default ignores, the repo's ``.gitignore`` files, and size/binary checks."""

    root: Path
    max_bytes: int = 512_000
    extra_ignores: Iterable[str] = ()
    _spec: pathspec.PathSpec = field(init=False)
    skipped: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lines = [*DEFAULT_IGNORES, *self.extra_ignores, *self._read_gitignores()]
        self._spec = pathspec.GitIgnoreSpec.from_lines(lines)

    def _read_gitignores(self) -> list[str]:
        """Collect patterns from every .gitignore, re-rooted to the repository root."""
        patterns: list[str] = []
        for gi in sorted(self.root.rglob(".gitignore")):
            rel_dir = gi.parent.relative_to(self.root).as_posix()
            if any(part in {".git", "node_modules", ".venv"} for part in gi.parts):
                continue
            for raw in gi.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if rel_dir in ("", "."):
                    patterns.append(line)
                    continue
                negate = line.startswith("!")
                body = line[1:] if negate else line
                anchored = body.lstrip("/") if "/" in body.rstrip("/") else f"**/{body}"
                patterns.append(("!" if negate else "") + f"{rel_dir}/{anchored}")
        return patterns

    def is_ignored(self, rel_path: str) -> bool:
        return self._spec.match_file(rel_path)

    def load(self, rel_path: str) -> SourceFile | None:
        """Read a file if it passes all filters, recording the reason when it does not."""
        if self.is_ignored(rel_path):
            self.skipped[rel_path] = "ignored"
            return None
        full = self.root / rel_path
        if not full.is_file():
            self.skipped[rel_path] = "missing"
            return None
        if full.suffix.lower() in BINARY_EXTENSIONS:
            self.skipped[rel_path] = "binary"
            return None
        size = full.stat().st_size
        if size == 0:
            self.skipped[rel_path] = "empty"
            return None
        if size > self.max_bytes:
            self.skipped[rel_path] = "too-large"
            return None
        data = full.read_bytes()
        if looks_binary(data):
            self.skipped[rel_path] = "binary"
            return None
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
        return SourceFile(path=rel_path, language=detect_language(rel_path), content=text)

    def walk(self) -> Iterator[SourceFile]:
        for full in sorted(self.root.rglob("*")):
            if not full.is_file():
                continue
            rel = full.relative_to(self.root).as_posix()
            if rel.startswith(".git/"):
                continue
            sf = self.load(rel)
            if sf is not None:
                yield sf
