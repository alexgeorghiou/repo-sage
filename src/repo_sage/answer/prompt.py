"""Prompt construction. Context blocks carry absolute line numbers so the model can
cite precise sub-ranges instead of whole chunks."""

from __future__ import annotations

from collections.abc import Sequence

from repo_sage.models import ScoredChunk

SYSTEM_PROMPT = """\
You are repo-sage, a senior engineer answering questions about one specific code repository.

Ground rules:
- Answer ONLY from the numbered context blocks supplied with the question. They are excerpts of \
the repository; each line is prefixed with its line number in the original file.
- Every factual statement about the code must carry at least one citation in the exact form \
[path/to/file.ext:START-END], using the file path and line numbers shown in the context. Cite the \
narrowest range that supports the statement. Several citations may share a bracket, separated by \
commas: [a.py:3-9, b.py:12-20].
- Never cite a file or line range that does not appear in the context.
- If the context does not contain the answer, say so plainly and suggest which file or symbol \
the user could look at next. Do not guess.
- Prefer short, direct answers. Use fenced code blocks only when quoting more than one line.
"""

NO_CONTEXT_ANSWER = (
    "I couldn't find anything in the indexed code related to that question. "
    "Try naming a file, function or concept that appears in the repository."
)


def _number(text: str, start_line: int) -> str:
    lines = text.split("\n")
    width = len(str(start_line + len(lines) - 1))
    return "\n".join(f"{start_line + i:>{width}} | {line}" for i, line in enumerate(lines))


def format_context(chunks: Sequence[ScoredChunk], max_chars: int = 24_000) -> str:
    blocks: list[str] = []
    used = 0
    for i, item in enumerate(chunks, start=1):
        c = item.chunk
        symbol = f" ({c.symbol})" if c.symbol else ""
        header = f"### [{i}] {c.path}:{c.start_line}-{c.end_line}{symbol}"
        block = f"{header}\n```{c.language or ''}\n{_number(c.text, c.start_line)}\n```"
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: Sequence[ScoredChunk], repo_name: str) -> str:
    return (
        f"Repository: {repo_name}\n\n"
        f"Context:\n\n{format_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer with citations in the form [path:START-END]."
    )
