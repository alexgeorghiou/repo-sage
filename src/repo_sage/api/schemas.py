from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from repo_sage.indexer import IndexReport
from repo_sage.models import Answer, Citation, Repo, ScoredChunk

IndexState = Literal["pending", "indexing", "ready", "failed"]


class RegisterRepoRequest(BaseModel):
    source: str = Field(description="Git URL (https/ssh) or a path on the API host.")
    name: str | None = None
    branch: str | None = None
    index: bool = True


class ReindexRequest(BaseModel):
    full: bool = False


class IndexReportOut(BaseModel):
    mode: str
    commit: str | None
    previous_commit: str | None
    files_indexed: int
    files_deleted: int
    chunks_written: int
    skipped: int

    @classmethod
    def of(cls, r: IndexReport) -> IndexReportOut:
        return cls(
            mode=r.mode.value,
            commit=r.commit,
            previous_commit=r.previous_commit,
            files_indexed=r.files_indexed,
            files_deleted=r.files_deleted,
            chunks_written=r.chunks_written,
            skipped=len(r.skipped),
        )


class RepoOut(BaseModel):
    id: str
    name: str
    source: str
    branch: str | None
    last_commit: str | None
    indexed_at: datetime | None
    file_count: int
    chunk_count: int
    state: IndexState
    error: str | None = None
    last_report: IndexReportOut | None = None

    @classmethod
    def of(
        cls,
        repo: Repo,
        state: IndexState,
        error: str | None = None,
        report: IndexReport | None = None,
    ) -> RepoOut:
        return cls(
            id=repo.id,
            name=repo.name,
            source=repo.source,
            branch=repo.branch,
            last_commit=repo.last_commit,
            indexed_at=repo.indexed_at,
            file_count=repo.file_count,
            chunk_count=repo.chunk_count,
            state=state,
            error=error,
            last_report=IndexReportOut.of(report) if report else None,
        )


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=30)


class CitationOut(BaseModel):
    path: str
    start_line: int
    end_line: int
    label: str

    @classmethod
    def of(cls, c: Citation) -> CitationOut:
        return cls(path=c.path, start_line=c.start_line, end_line=c.end_line, label=str(c))


class SourceOut(BaseModel):
    path: str
    start_line: int
    end_line: int
    symbol: str | None
    kind: str
    language: str | None
    score: float
    retrievers: dict[str, int]
    preview: str

    @classmethod
    def of(cls, s: ScoredChunk, preview_lines: int = 12) -> SourceOut:
        c = s.chunk
        return cls(
            path=c.path,
            start_line=c.start_line,
            end_line=c.end_line,
            symbol=c.symbol,
            kind=c.kind.value,
            language=c.language,
            score=round(s.score, 6),
            retrievers=s.sources,
            preview="\n".join(c.text.split("\n")[:preview_lines]),
        )


class AnswerOut(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
    invalid_citations: list[CitationOut]
    grounded: bool
    sources: list[SourceOut]

    @classmethod
    def of(cls, a: Answer) -> AnswerOut:
        return cls(
            question=a.question,
            answer=a.text,
            citations=[CitationOut.of(c) for c in a.citations],
            invalid_citations=[CitationOut.of(c) for c in a.invalid_citations],
            grounded=a.grounded,
            sources=[SourceOut.of(s) for s in a.context],
        )


class SnippetOut(BaseModel):
    path: str
    start_line: int
    end_line: int
    language: str | None
    content: str
