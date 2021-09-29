"""Code-aware tokenisation shared by the lexical index, hashing embedder and reranker."""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "do",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "that",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "which",
        "can",
        "you",
        "me",
        "my",
        "we",
        "our",
        "there",
        "their",
        "its",
        "into",
    ]
)


def split_identifier(word: str) -> list[str]:
    """``parseHTTPResponse_v2`` -> ``["parse", "http", "response", "v", "2"]``."""
    parts: list[str] = []
    for piece in word.split("_"):
        parts.extend(m.group(0).lower() for m in _CAMEL.finditer(piece))
    return [p for p in parts if p]


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercased tokens: whole identifiers plus their camel/snake sub-words."""
    tokens: list[str] = []
    for match in _WORD.finditer(text):
        word = match.group(0)
        lower = word.lower()
        subs = split_identifier(word)
        if len(subs) > 1:
            tokens.append(lower)
            tokens.extend(subs)
        else:
            tokens.append(lower)
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
