from pathlib import Path

import pytest

from repo_sage.evaluation.metrics import (
    hit_at_k,
    mean,
    recall_at_k,
    reciprocal_rank,
    unique_in_order,
)
from repo_sage.evaluation.runner import EvalSet, run_eval

EVAL_FILE = Path(__file__).resolve().parents[1] / "evals" / "tinyqueue.yaml"


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["a", "c"], 1) == 0.5
    assert recall_at_k(retrieved, ["a", "c"], 3) == 1.0
    assert recall_at_k(retrieved, ["z"], 4) == 0.0
    assert recall_at_k(retrieved, [], 4) == 0.0


def test_reciprocal_rank_and_hits():
    assert reciprocal_rank(["x", "y", "a"], ["a", "y"]) == pytest.approx(1 / 2)
    assert reciprocal_rank(["x"], ["a"]) == 0.0
    assert hit_at_k(["x", "a"], ["a"], 1) == 0.0
    assert hit_at_k(["x", "a"], ["a"], 2) == 1.0


def test_helpers():
    assert unique_in_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert mean([]) == 0.0
    assert mean([1.0, 0.0, 0.5]) == pytest.approx(0.5)


def test_eval_set_loads_relative_repo_path():
    es = EvalSet.load(EVAL_FILE)
    assert es.repo.is_dir()
    assert (es.repo / "tinyqueue" / "queue.py").is_file()
    assert es.ks == (1, 3, 5)
    assert len(es.cases) >= 10


def test_eval_set_rejects_empty(tmp_path: Path):
    f = tmp_path / "empty.yaml"
    f.write_text("name: x\nrepo: .\nquestions: []\n")
    with pytest.raises(ValueError):
        EvalSet.load(f)


def test_sample_eval_meets_quality_floor():
    """Regression guard for the offline (hashing + lexical) retrieval stack."""
    report = run_eval(EvalSet.load(EVAL_FILE))
    summary = report.as_dict()
    assert summary["questions"] == len(report.results)
    assert report.recall_at(5) >= 0.9
    assert report.mrr >= 0.7
    assert all(c["first_hit_rank"] is not None for c in summary["cases"])
