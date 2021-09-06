"""File extension to language mapping.

Only languages listed in :data:`TREE_SITTER_LANGUAGES` get syntax-aware chunking;
the rest are still indexed with the line-window chunker.
"""

from __future__ import annotations

from pathlib import PurePosixPath

EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".sql": "sql",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".vue": "vue",
    ".svelte": "svelte",
}

FILENAMES: dict[str, str] = {
    "Dockerfile": "dockerfile",
    "Makefile": "make",
    "Justfile": "make",
    "README": "text",
    "LICENSE": "text",
}

TREE_SITTER_LANGUAGES = frozenset({"python", "javascript", "typescript", "tsx", "go", "rust", "java"})


def detect_language(path: str) -> str | None:
    p = PurePosixPath(path)
    if p.name in FILENAMES:
        return FILENAMES[p.name]
    return EXTENSIONS.get(p.suffix.lower())
