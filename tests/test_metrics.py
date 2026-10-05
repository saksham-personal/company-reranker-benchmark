import math

import pytest

from reranker_bench.metrics import evaluate_run, oracle_run, paired_bootstrap


def test_hand_computed_ndcg_uses_full_known_qrels_and_stable_unique_run():
    qrels = {"q": {"a": 2, "b": 1, "outside": 3}}
    result = evaluate_run(qrels, {"q": ["a", "a", "unknown", "b"]}, ks=(2,), complete=True)
    # IDCG is over all qrels (3,2), while run DCG uses the deduplicated top 2 (2,0).
    expected = (3 / math.log2(2)) / (7 / math.log2(2) + 3 / math.log2(3))
    assert result["aggregate"]["ndcg@2"] == pytest.approx(expected)
    assert result["aggregate"]["known_positive_recall@2"] == pytest.approx(1 / 3)
    assert result["aggregate"]["precision@2"] == pytest.approx(1 / 2)
    assert result["aggregate"]["map@2"] == pytest.approx(1 / 3)


def test_incomplete_qrels_suppresses_precision_and_map_and_diagnoses_exclusions():
    result = evaluate_run({"positive": {"d": 1}, "empty": {"x": 0}}, {"positive": ["d"], "empty": ["x"], "unknown": ["z"]}, ks=(1,), complete=False)
    assert "precision@1" not in result["aggregate"]
    assert "map@1" not in result["aggregate"]
    assert result["aggregate"]["known_positive_recall@1"] == 1
    assert result["limitations"]["query_status_counts"] == {"eligible": 1, "zero_positive": 1, "unknown_qrels": 1}
    assert result["limitations"]["macro_query_count"] == 1
    assert result["per_query"]["unknown"]["status"] == "unknown_qrels"


def test_oracle_does_not_inject_positive_documents():
    assert oracle_run({"q": ["negative", "also-negative"]}, {"q": {"positive": 5}}) == {"q": ["negative", "also-negative"]}


def test_paired_bootstrap_matches_hand_delta_and_is_seeded():
    before = {"a": {"ndcg@10": 0.2}, "b": {"ndcg@10": 0.4}}
    after = {"a": {"ndcg@10": 0.3}, "b": {"ndcg@10": 0.6}, "c": {"ndcg@10": 1}}
    first = paired_bootstrap(before, after, samples=200)
    second = paired_bootstrap(before, after, samples=200)
    assert first == second
    assert first["n"] == 2
    assert first["mean_delta"] == pytest.approx(0.15)
    assert first["ci95"][0] <= first["mean_delta"] <= first["ci95"][1]


def test_non_finite_qrel_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_run({"q": {"d": float("nan")}}, {"q": ["d"]})
