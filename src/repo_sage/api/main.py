"""ASGI entry point: ``uvicorn repo_sage.api.main:app``."""

from repo_sage.api.app import create_app

app = create_app()
