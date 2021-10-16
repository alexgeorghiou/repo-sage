import pytest

from repo_sage.retrieval.fusion import collapse_overlaps, reciprocal_rank_fusion
from tests.conftest import scored


def test_rrf_rewards_agreement_between_retrievers(make_chunk):
    a, b, c, d = (make_chunk(path=f"{n}.py") for n in "abcd")
    fused = reciprocal_rank_fusion({"dense": scored(a, b, c), "lexical": scored(c, d, b)}, k=60)
    ids = [r.chunk.path for r in fused]
    # b and c appear in both lists, so they beat single-list hits
    assert set(ids[:2]) == {"b.py", "c.py"}
    assert ids.index("c.py") < ids.index("b.py")  # c ranks 3 + 1 vs b 2 + 3
    assert fused[0].sources == {"dense": 3, "lexical": 1}


def test_rrf_scores_match_formula(make_chunk):
    a, b = make_chunk(path="a.py"), make_chunk(path="b.py")
    fused = reciprocal_rank_fusion({"x": scored(a, b), "y": scored(b)}, k=10)
    by_path = {r.chunk.path: r.score for r in fused}
    assert by_path["a.py"] == pytest.approx(1 / 11)
    assert by_path["b.py"] == pytest.approx(1 / 12 + 1 / 11)


def test_rrf_ignores_duplicates_within_one_list(make_chunk):
    a, b = make_chunk(path="a.py"), make_chunk(path="b.py")
    fused = reciprocal_rank_fusion({"x": scored(a, a, b)}, k=1)
    assert [r.chunk.path for r in fused] == ["a.py", "b.py"]
    assert fused[1].sources == {"x": 2}


def test_rrf_weights_and_limit(make_chunk):
    a, b = make_chunk(path="a.py"), make_chunk(path="b.py")
    fused = reciprocal_rank_fusion(
        {"dense": scored(a), "lexical": scored(b)}, weights={"lexical": 2.0}, limit=1
    )
    assert [r.chunk.path for r in fused] == ["b.py"]


def test_rrf_rejects_non_positive_k(make_chunk):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"x": scored(make_chunk())}, k=0)


def test_collapse_overlaps_drops_contained_windows(make_chunk):
    first = make_chunk(path="a.py", start=1, end=60)
    overlapping = make_chunk(path="a.py", start=50, end=70)  # 11/21 lines shared
    contained = make_chunk(path="a.py", start=10, end=20)
    other_file = make_chunk(path="b.py", start=10, end=20)
    kept = collapse_overlaps(scored(first, contained, overlapping, other_file))
    assert [(k.chunk.path, k.chunk.start_line) for k in kept] == [("a.py", 1), ("b.py", 10)]
