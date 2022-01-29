# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git is needed at runtime to clone and diff repositories.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[postgres,openai,anthropic]"

COPY evals ./evals
COPY tests/fixtures ./tests/fixtures

RUN useradd --create-home --uid 10001 sage \
    && mkdir -p /data && chown sage:sage /data
USER sage

ENV REPO_SAGE_DATA_DIR=/data
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "repo_sage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
