# repo-sage

**Chat with any GitHub repository.** Point repo-sage at a git URL or a local checkout. It parses the code with tree-sitter, indexes it for hybrid dense + lexical search, and answers questions with citations down to `file:line` ranges. Every citation is checked against the code the model was actually shown.

[![CI](https://github.com/alexgeorghiou/repo-sage/actions/workflows/ci.yml/badge.svg)](https://github.com/alexgeorghiou/repo-sage/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Postgres + pgvector](https://img.shields.io/badge/Postgres-pgvector-4169e1?logo=postgresql&logoColor=white)
![React + TypeScript](https://img.shields.io/badge/React-TypeScript-61dafb?logo=react&logoColor=black)
![tree-sitter](https://img.shields.io/badge/parsing-tree--sitter-green)

```text
$ export REPO_SAGE_LLM=anthropic
$ repo-sage ask tests/fixtures/sample_repo "How are failed jobs retried?"

Failed jobs go back on the queue with a delay from RetryPolicy.backoff
[tinyqueue/worker.py:57-60]. The delay uses exponential backoff with full
jitter, capped at max_delay [tinyqueue/retry.py:18-26].

Citations
  tinyqueue/worker.py:57-60
  tinyqueue/retry.py:18-26
```

## Features

- **Code-aware chunking.** tree-sitter grammars for Python, JavaScript, TypeScript/TSX, Go, Rust and Java split files into functions, classes and methods, each with a qualified symbol name such as `Scheduler.tick` or `Stack.Push`. Comments directly above a definition stay attached to it. Large classes are split into their members. Leftover lines (imports, constants, fields) become their own chunks, so nothing in a file goes unindexed. Other languages fall back to a line-window chunker, and Markdown is split by heading.
- **Hybrid retrieval.** Dense vector search (pgvector HNSW, cosine) and lexical search (Postgres full-text, or BM25 in memory) are combined with **Reciprocal Rank Fusion**. A pluggable reranker then reorders the results, and windows that overlap a higher-ranked chunk are dropped.
- **Identifier-aware tokenisation.** `computeRetryDelay` and `compute_retry_delay` are both indexed as `compute`, `retry` and `delay`, so lexical search and the hashing embedder can match natural-language questions to code names.
- **Answers with verified citations.** The prompt shows code with absolute line numbers and requires `[path:START-END]` citations. The parser accepts several formats (lists in one bracket, `L10-L20`, en dashes). Each citation is checked against the retrieved context, and any that fall outside it are flagged in the API and UI.
- **Incremental re-indexing.** The last indexed commit SHA is stored per repository. A sync runs `git diff --name-status -M` from that commit to `HEAD` and re-chunks only the files that changed. Renames remove the old path. A change to `.gitignore` or rewritten history (the old SHA no longer exists) triggers a full rebuild.
- **Swappable providers.** Embeddings (deterministic hashing or OpenAI), LLM (fake, Anthropic or OpenAI), reranker (lexical heuristic, cross-encoder or none) and storage (in-memory or Postgres + pgvector) are each chosen by an environment variable.
- **Retrieval evals.** A YAML question set over a bundled sample repository reports recall@k, hit-rate@k and MRR. CI fails if MRR drops below a threshold.
- **API, CLI and UI.** FastAPI with background indexing and server-sent-event streaming. A Typer CLI (`index`, `ask`, `chat`, `repos`, `eval`, `serve`). A React + TypeScript UI with a repo list, streaming chat, citation chips and a code panel that highlights the cited lines.

## Architecture

```mermaid
flowchart LR
    subgraph Ingest
        SRC[git URL / local path] --> GS[GitSource<br/>clone · fetch · diff]
        GS --> FF[FileFilter<br/>.gitignore · binary · size]
        FF --> CH[CodeChunker<br/>tree-sitter → line windows]
        CH --> EMB[Embedder<br/>hashing | OpenAI]
    end

    EMB --> ST[(Store<br/>Postgres + pgvector<br/>or in-memory)]

    subgraph Query
        Q[question] --> DENSE[dense search<br/>HNSW cosine]
        Q --> LEX[lexical search<br/>tsvector / BM25]
        DENSE --> RRF[Reciprocal Rank Fusion]
        LEX --> RRF
        RRF --> RR[Reranker<br/>lexical | cross-encoder]
        RR --> P[citing prompt<br/>line-numbered context]
        P --> LLM[LLM<br/>fake | Anthropic | OpenAI]
        LLM --> CV[citation parser<br/>+ validator]
    end

    ST --> DENSE
    ST --> LEX
    CV --> API[FastAPI · SSE]
    API --> UI[React UI]
    CV --> CLI[Typer CLI]
```

Incremental sync, step by step:

```mermaid
sequenceDiagram
    participant C as Client
    participant I as Indexer
    participant G as git
    participant S as Store
    C->>I: reindex(repo)
    I->>G: fetch + reset (remote) / read HEAD
    I->>S: last_commit?
    alt unknown or rewritten history or .gitignore changed
        I->>S: clear chunks, index every tracked file
    else normal case
        I->>G: diff --name-status -M last..HEAD
        I->>S: delete chunks for deleted / renamed-from paths
        I->>S: replace chunks for changed paths only
    end
    I->>S: save repo(last_commit = HEAD)
```

## Quickstart

### Offline in 30 seconds (no keys, no database)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

repo-sage eval                                        # retrieval metrics on the bundled sample repo
repo-sage ask tests/fixtures/sample_repo "How does the worker shut down on SIGTERM?"
repo-sage chat .                                      # chat with this repository
```

The default stack (hashing embedder, BM25, lexical reranker, fake LLM) is deterministic and fully offline. The fake LLM writes extractive answers that cite the top chunks, so the whole pipeline, including citation validation, runs end to end.

### Real models

```bash
pip install -e ".[anthropic]"          # or ".[openai]"
export REPO_SAGE_LLM=anthropic         # or openai
export ANTHROPIC_API_KEY=...           # or OPENAI_API_KEY
export REPO_SAGE_LLM_MODEL=claude-opus-5-5   # optional; any model id works

# semantic embeddings
export REPO_SAGE_EMBEDDER=openai REPO_SAGE_EMBEDDING_DIM=1536
```

### Full stack with Docker

```bash
cp .env.example .env      # optionally set REPO_SAGE_LLM and API keys
docker compose up --build
# UI  → http://localhost:5173
# API → http://localhost:8000/docs
```

Compose runs `pgvector/pgvector:pg16`, the API (`REPO_SAGE_STORE=postgres`) and the UI behind nginx. nginx proxies `/api` with buffering turned off so streaming works. The schema is created on startup.

### API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/repos` | List repositories with index state, commit and counts |
| `POST` | `/api/repos` | Register `{source, name?, branch?}` and index it in the background |
| `POST` | `/api/repos/{id}/reindex` | Incremental sync, or `{"full": true}` |
| `DELETE` | `/api/repos/{id}` | Remove a repository and its chunks |
| `POST` | `/api/repos/{id}/ask` | `{question, history?, top_k?}` → answer, citations, sources |
| `POST` | `/api/repos/{id}/ask/stream` | Same as `ask`, streamed as SSE: `sources`, then `token` events, then `answer` |
| `GET` | `/api/repos/{id}/search?q=` | Inspect fused and reranked retrieval, with per-retriever ranks |
| `GET` | `/api/repos/{id}/snippet?path=&start=&end=` | Lines behind a citation chip (path-traversal safe) |

### Configuration

All settings are environment variables prefixed with `REPO_SAGE_`. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Options |
| --- | --- | --- |
| `REPO_SAGE_STORE` | `memory` | `memory`, `postgres` |
| `REPO_SAGE_DATABASE_URL` | `postgresql://sage:sage@localhost:5432/sage` | |
| `REPO_SAGE_EMBEDDER` | `hashing` | `hashing`, `openai` |
| `REPO_SAGE_EMBEDDING_DIM` | `384` | must match the pgvector column |
| `REPO_SAGE_LLM` | `fake` | `fake`, `anthropic`, `openai` |
| `REPO_SAGE_LLM_MODEL` | provider default | any model id |
| `REPO_SAGE_RERANKER` | `lexical` | `lexical`, `cross-encoder`, `none` |
| `REPO_SAGE_CHUNK_MAX_LINES` | `80` | max lines before a definition is split |

## Evaluation

[`evals/tinyqueue.yaml`](evals/tinyqueue.yaml) contains 14 hand-written questions about the bundled [`tinyqueue`](tests/fixtures/sample_repo) sample repository. Each question lists the files a developer would need to open to answer it. Metrics are computed per file, from the ranked chunks:

```bash
repo-sage eval evals/tinyqueue.yaml          # table view
repo-sage eval --json --min-mrr 0.7          # CI gate
```

With the offline stack (hashing embedder + BM25 + lexical reranker), the current results are MRR ≈ 0.83, recall@1 ≈ 0.68, recall@3 ≈ 0.93 and recall@5 = 1.0. The suite is small on purpose: it guards against regressions in chunking and fusion, and it is not a benchmark. To compare configurations, point it at your own repository and question set.

## Project structure

```text
repo-sage/
├── src/repo_sage/
│   ├── ingest/          # git clone/fetch/diff, .gitignore-aware filtering, language detection
│   ├── chunking/        # tree-sitter chunker, line-window + markdown fallback
│   ├── embeddings/      # Embedder protocol, hashing embedder, OpenAI
│   ├── store/           # Store protocol, in-memory (BM25) and Postgres/pgvector
│   ├── retrieval/       # BM25, Reciprocal Rank Fusion, rerankers, HybridRetriever
│   ├── llm/             # LLM protocol, FakeLLM, Anthropic, OpenAI
│   ├── answer/          # citing prompt, citation parser/validator, AnswerEngine
│   ├── evaluation/      # recall@k / MRR metrics and YAML eval runner
│   ├── api/             # FastAPI app, schemas, background index jobs, SSE
│   ├── indexer.py       # full + commit-diff incremental indexing
│   ├── services.py      # composition root (settings → object graph)
│   └── cli.py           # Typer CLI
├── tests/               # unit, API and opt-in pgvector integration tests
│   └── fixtures/sample_repo/   # tinyqueue: small Python/TS/Go repo used by tests and evals
├── evals/tinyqueue.yaml
├── web/                 # Vite + React + TypeScript UI
├── Dockerfile, web/Dockerfile, docker-compose.yml
└── .github/workflows/ci.yml
```

## Design decisions and trade-offs

- **Chunk by syntax, not size.** A function is the unit developers think and cite in, so it is the chunk. Fixed windows are kept only as a fallback. Trade-off: chunk sizes vary, so oversized definitions are re-split into windows that keep the symbol name.
- **RRF rather than score blending.** Cosine similarity and BM25 / `ts_rank_cd` are on scales that can't be compared. RRF uses only ranks, needs no tuning per corpus, and records which retriever found each chunk, which the API exposes for debugging.
- **Pre-tokenise for Postgres full-text.** Identifiers are split in Python and stored in a `lexemes` column with a generated `tsvector` (`simple` config). Postgres lexical search then behaves like the in-memory BM25 path. Trade-off: `ts_rank_cd` is not true BM25. It is close enough for candidate generation, and the reranker sees the fused pool anyway.
- **Validate citations rather than trust them.** A citation counts only if its range falls inside a chunk the model was shown, with two lines of slack. This catches invented paths and line numbers, but it can't prove the cited lines support the claim.
- **Diff against commits, not the working tree.** Incremental sync follows `HEAD`, so results can be reproduced and tied to a SHA. Uncommitted edits in a local checkout are picked up only by a full re-index.
- **Every external dependency sits behind a protocol.** Store, embedder, LLM and reranker each have a fake or local implementation, so the 60+ tests run in seconds without network access, keys or a database. The pgvector tests run in CI against a service container.
- **Hashing embedder as the default.** It isn't semantic, but it is deterministic and costs nothing, which makes evals reproducible. For quality, switch to a real embedding model. The eval command shows what changes.

## Roadmap

- [ ] Symbol graph (imports and call sites) to expand retrieval to definitions a chunk references
- [ ] Watch mode / webhook endpoint that syncs on push
- [ ] Per-language test-file down-weighting and `README` boosting for "how do I…" questions
- [ ] Answer-level evals (citation precision, faithfulness) next to the retrieval metrics
- [ ] Multi-repository questions across an organisation
- [ ] Token-aware context packing instead of a character budget

## License

[MIT](LICENSE) © 2026 Alex Georghiou
