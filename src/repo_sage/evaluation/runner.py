"""Run a YAML eval set against the retriever.

Eval files look like::

    name: tinyqueue
    repo: ../tests/fixtures/sample_repo   # relative to this file
    ks: [1, 3, 5]
    questions:
      - question: How long does a failed job wait before it is retried?
        expected_files: [tinyqueue/retry.py]

Metrics are computed at file granularity: a question is answered well when the
files a human would open appear near the top of the retrieved chunks.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from repo_sage.embeddings.base import Embedder
from repo_sage.embeddings.hashing import HashingEmbedder
from repo_sage.evaluation.metrics import (
    hit_at_k,
    mean,
    recall_at_k,
    reciprocal_rank,
    unique_in_order,
)
from repo_sage.indexer import Indexer
from repo_sage.retrieval.rerank import LexicalReranker, Reranker
from repo_sage.retrieval.retriever import HybridRetriever
from repo_sage.store.memory import InMemoryStore


@dataclass(frozen=True, slots=True)
class EvalCase:
    question: str
    expected_files: tuple[str, ...]


@dataclass(slots=True)
class EvalSet:
    name: str
    repo: Path
    cases: list[EvalCase]
    ks: tuple[int, ...] = (1, 3, 5)

    @classmethod
    def load(cls, path: Path) -> EvalSet:
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        repo = Path(data["repo"])
        if not repo.is_absolute():
            repo = (path.parent / repo).resolve()
        cases = [
            EvalCase(q["question"], tuple(q["expected_files"])) for q in data.get("questions", [])
        ]
        if not cases:
            raise ValueError(f"{path} contains no questions")
        return cls(
            name=data.get("name", path.stem),
            repo=repo,
            cases=cases,
            ks=tuple(sorted(data.get("ks", [1, 3, 5]))),
        )


@dataclass(slots=True)
class CaseResult:
    case: EvalCase
    retrieved_files: list[str]
    rr: float
    recall: dict[int, float]

    @property
    def first_hit_rank(self) -> int | None:
        return round(1 / self.rr) if self.rr else None


@dataclass(slots=True)
class EvalReport:
    name: str
    ks: tuple[int, ...]
    results: list[CaseResult] = field(default_factory=list)

    @property
    def mrr(self) -> float:
        return mean(r.rr for r in self.results)

    def recall_at(self, k: int) -> float:
        return mean(r.recall[k] for r in self.results)

    def hit_rate_at(self, k: int) -> float:
        return mean(hit_at_k(r.retrieved_files, r.case.expected_files, k) for r in self.results)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "questions": len(self.results),
            "mrr": round(self.mrr, 4),
            "recall": {f"@{k}": round(self.recall_at(k), 4) for k in self.ks},
            "hit_rate": {f"@{k}": round(self.hit_rate_at(k), 4) for k in self.ks},
            "cases": [
                {
                    "question": r.case.question,
                    "expected": list(r.case.expected_files),
                    "first_hit_rank": r.first_hit_rank,
                    "top_files": r.retrieved_files[: max(self.ks)],
                }
                for r in self.results
            ],
        }


def run_eval(
    eval_set: EvalSet,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    chunk_budget: int = 20,
) -> EvalReport:
    """Index the eval repo into a throwaway in-memory store and score retrieval."""
    store = InMemoryStore()
    embedder = embedder or HashingEmbedder()
    with tempfile.TemporaryDirectory(prefix="repo-sage-eval-") as tmp:
        indexer = Indexer(store=store, embedder=embedder, data_dir=Path(tmp))
        repo = indexer.register(str(eval_set.repo), name=eval_set.name)
        indexer.index(repo.id)
    retriever = HybridRetriever(
        store=store, embedder=embedder, reranker=reranker or LexicalReranker()
    )

    report = EvalReport(name=eval_set.name, ks=eval_set.ks)
    for case in eval_set.cases:
        hits = retriever.retrieve(repo.id, case.question, top_k=chunk_budget)
        files = unique_in_order(h.chunk.path for h in hits)
        report.results.append(
            CaseResult(
                case=case,
                retrieved_files=files,
                rr=reciprocal_rank(files, case.expected_files),
                recall={k: recall_at_k(files, case.expected_files, k) for k in eval_set.ks},
            )
        )
    return report
