"""FastAPI dependencies resolving services and repositories from app state."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from repo_sage.api.jobs import IndexJobs
from repo_sage.models import Repo
from repo_sage.services import Services


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_jobs(request: Request) -> IndexJobs:
    return request.app.state.jobs


ServicesDep = Annotated[Services, Depends(get_services)]
JobsDep = Annotated[IndexJobs, Depends(get_jobs)]


def load_repo(repo_id: str, services: ServicesDep) -> Repo:
    repo = services.store.get_repo(repo_id)
    if repo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown repository '{repo_id}'")
    return repo


RepoDep = Annotated[Repo, Depends(load_repo)]
