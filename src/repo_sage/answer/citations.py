"""Extract ``[path:start-end]`` citations from model output and check them against the
context that was actually shown to the model."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from repo_sage.models import Chunk, Citation

_BRACKET = re.compile(r"\[([^\[\]\n]{3,400})\]")
_ONE = re.compile(
    r"^`?(?P<path>[\w.@+\-/]+):L?(?P<start>\d+)(?:\s*[-2013]\s*L?(?P<end>\d+))?`?$"
)


def parse_citations(text: str) -> list[Citation]:
    """Return citations in order of first appearance, de-duplicated.

    Accepts ``[a.py:10-20]``, ``[a.py:10]``, ``[a.py:L10-L20]`` and comma/semicolon
    separated lists inside one bracket. Markdown links such as ``[text](url)`` and
    plain bracketed prose are ignored because they do not match ``path:line``.
    """
    seen: set[Citation] = set()
    out: list[Citation] = []
    for bracket in _BRACKET.finditer(text):
        end = bracket.end()
        if end < len(text) and text[end] == "(":  # markdown link text
            continue
        for part in re.split(r"[;,]", bracket.group(1)):
            m = _ONE.match(part.strip())
            if not m:
                continue
            start = int(m.group("start"))
            stop = int(m.group("end") or start)
            if stop < start:
                start, stop = stop, start
            cite = Citation(m.group("path").removeprefix("./"), start, stop)
            if cite not in seen:
                seen.add(cite)
                out.append(cite)
    return out


def validate_citations(
    citations: Iterable[Citation], context: Sequence[Chunk], slack: int = 2
) -> tuple[list[Citation], list[Citation]]:
    """Split citations into (valid, invalid).

    A citation is valid when its range lies inside one of the context chunks for the
    same path, allowing ``slack`` lines either side for off-by-one models.
    """
    valid: list[Citation] = []
    invalid: list[Citation] = []
    for cite in citations:
        ok = any(
            c.path == cite.path
            and cite.start_line >= c.start_line - slack
            and cite.end_line <= c.end_line + slack
            for c in context
        )
        (valid if ok else invalid).append(cite)
    return valid, invalid
