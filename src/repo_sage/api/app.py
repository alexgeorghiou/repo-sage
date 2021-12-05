"""HTTP API: register and index repositories, then ask questions about them."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from repo_sage import __version__
from repo_sage.api.deps import JobsDep, RepoDep, ServicesDep
from repo_sage.api.jobs import IndexJobs
from repo_sage.api.schemas import (
    AnswerOut,
    AskRequest,
    RegisterRepoRequest,
    ReindexRequest,
    RepoOut,
    SnippetOut,
    SourceOut,
)
from repo_sage.config import Settings, get_settings
from repo_sage.ingest.languages import detect_language
from repo_sage.llm.base import Message
from repo_sage.models import Repo
from repo_sage.services import Services

log = logging.getLogger(__name__)


def create_app(services: Services | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    svc = services or Services.build(settings)
    jobs = IndexJobs(svc.indexer)

    app = FastAPI(
        title="repo-sage",
        version=__version__,
        description="Chat with any git repository, with answers cited to file and line.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.services = svc
    app.state.jobs = jobs

    def repo_out(repo: Repo, jobs: IndexJobs) -> RepoOut:
        st = jobs.status(repo.id, has_index=repo.last_commit is not None or repo.chunk_count > 0)
        return RepoOut.of(repo, st.state, st.error, st.report)  # type: ignore[arg-type]

    def require_ready(repo: Repo, jobs: IndexJobs) -> None:
        if repo.chunk_count == 0 and jobs.is_running(repo.id):
            raise HTTPException(status.HTTP_409_CONFLICT, "repository is still being indexed")

    # -- meta ---------------------------------------------------------------------

    @app.get("/api/health")
    def health(services: ServicesDep) -> dict[str, str]:
        return {
            "status": "ok",
            "version": __version__,
            "store": services.settings.store,
            "embedder": services.settings.embedder,
            "llm": services.llm.name,
        }

    # -- repositories ---------------------------------------------------------------

    @app.get("/api/repos")
    def list_repos(services: ServicesDep, jobs: JobsDep) -> list[RepoOut]:
        return [repo_out(r, jobs) for r in services.store.list_repos()]

    @app.post("/api/repos", status_code=status.HTTP_202_ACCEPTED)
    def register_repo(
        body: RegisterRepoRequest,
        background: BackgroundTasks,
        services: ServicesDep,
        jobs: JobsDep,
    ) -> RepoOut:
        try:
            repo = services.indexer.register(body.source, name=body.name, branch=body.branch)
        except FileNotFoundError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        if body.index and not jobs.is_running(repo.id):
            jobs.mark_queued(repo.id)
            background.add_task(jobs.run, repo.id)
        return repo_out(repo, jobs)

    @app.get("/api/repos/{repo_id}")
    def get_repo(repo: RepoDep, jobs: JobsDep) -> RepoOut:
        return repo_out(repo, jobs)

    @app.post("/api/repos/{repo_id}/reindex", status_code=status.HTTP_202_ACCEPTED)
    def reindex(
        repo: RepoDep,
        background: BackgroundTasks,
        jobs: JobsDep,
        body: ReindexRequest | None = None,
    ) -> RepoOut:
        if jobs.is_running(repo.id):
            raise HTTPException(status.HTTP_409_CONFLICT, "an index run is already in progress")
        jobs.mark_queued(repo.id)
        background.add_task(jobs.run, repo.id, (body or ReindexRequest()).full)
        return repo_out(repo, jobs)

    @app.delete("/api/repos/{repo_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_repo(repo: RepoDep, services: ServicesDep, jobs: JobsDep) -> None:
        services.store.delete_repo(repo.id)
        jobs.forget(repo.id)

    # -- retrieval & answers ----------------------------------------------------------

    @app.get("/api/repos/{repo_id}/search")
    def search(
        repo: RepoDep,
        services: ServicesDep,
        q: Annotated[str, Query(min_length=1)],
        k: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> list[SourceOut]:
        return [SourceOut.of(s) for s in services.retriever.retrieve(repo.id, q, k)]

    @app.post("/api/repos/{repo_id}/ask")
    def ask(repo: RepoDep, body: AskRequest, services: ServicesDep, jobs: JobsDep) -> AnswerOut:
        require_ready(repo, jobs)
        history = [Message(t.role, t.content) for t in body.history]
        answer = services.engine.ask(repo.id, repo.name, body.question, history, body.top_k)
        return AnswerOut.of(answer)

    @app.post("/api/repos/{repo_id}/ask/stream")
    def ask_stream(
        repo: RepoDep, body: AskRequest, services: ServicesDep, jobs: JobsDep
    ) -> EventSourceResponse:
        require_ready(repo, jobs)
        history = [Message(t.role, t.content) for t in body.history]

        def events() -> Iterator[dict[str, str]]:
            stream = services.engine.ask_stream(
                repo.id, repo.name, body.question, history, body.top_k
            )
            try:
                for event in stream:
                    kind = event["type"]
                    if kind == "context":
                        data = [SourceOut.of(s).model_dump() for s in event["chunks"]]
                        yield {"event": "sources", "data": json.dumps(data)}
                    elif kind == "token":
                        yield {"event": "token", "data": json.dumps(event["text"])}
                    elif kind == "answer":
                        out = AnswerOut.of(event["answer"]).model_dump(exclude={"sources"})
                        yield {"event": "answer", "data": json.dumps(out)}
            except Exception as exc:
                log.exception("streaming answer failed")
                yield {"event": "error", "data": json.dumps(str(exc))}

        return EventSourceResponse(events())

    @app.get("/api/repos/{repo_id}/snippet")
    def snippet(
        repo: RepoDep,
        services: ServicesDep,
        path: Annotated[str, Query(min_length=1)],
        start: Annotated[int, Query(ge=1)] = 1,
        end: Annotated[int | None, Query(ge=1)] = None,
        context: Annotated[int, Query(ge=0, le=50)] = 3,
    ) -> SnippetOut:
        """Return file lines for a citation, read from the indexed working tree."""
        root = services.indexer.source_for(repo).root.resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or ".git" in Path(path).parts:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "path escapes the repository")
        if not target.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no such file '{path}'")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        end = min(end or start, len(lines))
        lo, hi = max(1, start - context), min(len(lines), end + context)
        return SnippetOut(
            path=path,
            start_line=lo,
            end_line=hi,
            language=detect_language(path),
            content="\n".join(lines[lo - 1 : hi]),
        )

    return app
