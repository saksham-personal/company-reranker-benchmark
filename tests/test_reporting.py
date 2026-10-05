import csv
import json

from reranker_bench.reporting import generate_report


def _quality(status="ok", model="m1", dataset="d1"):
    per_query = {"q1": {"status": "eligible", "ndcg@10": 0.2, "ndcg@20": 0.3, "ndcg@100": 0.4, "known_positive_recall@10": 0.4, "known_positive_recall@20": 0.5, "known_positive_recall@100": 1.0}}
    score = {"aggregate": {"ndcg@10": 0.2}, "per_query": per_query, "limitations": {"macro_query_count": 1, "query_status_counts": {"eligible": 1}}}
    return {
        "schema_version": 1, "type": "quality", "status": status, "model_id": model,
        "artifact_id": f"{model}-artifact", "runtime": "torch-fp32", "precision": "fp32",
        "dataset_id": dataset, "dataset_digest": f"{dataset}-digest", "candidate_digest": "cand-digest",
        "representation": "passage", "query_count": 1, "rerank_depth": 500, "retrieve_depth": 1000,
        "max_length": 512, "threads": 1, "batch_size": 1, "machine": {"cpu": "test"},
        "metrics": {"baseline": score, "reranked": score, "oracle": score}, "elapsed_s": 1,
        "per_query_timings": {"q1": 1.0},
    }


def test_report_preserves_failures_mixed_comparison_groups_and_public_refs(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.json").write_text(json.dumps([_quality(), _quality("failed", "bad")] ), encoding="utf-8")
    (results / "b.json").write_text(json.dumps({"type": "speed", "status": "ok", "model_id": "m1", "speed": {"pairs_per_second": 3.0, "p50_ms": 4, "p95_ms": 8}}), encoding="utf-8")
    (results / "c.json").write_text(json.dumps(_quality(model="m2", dataset="d2")), encoding="utf-8")
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "models.json").write_text(json.dumps({"public_reference": [{"name": "paper-score", "value": 0.9}]}), encoding="utf-8")
    output = tmp_path / "reports"
    paths = generate_report(results, output, root)
    assert all(path.exists() for path in paths.values())
    report = paths["report"].read_text(encoding="utf-8")
    assert "Mixed comparison conditions: yes" in report
    assert "public_reference" in report and "paper-score" in report
    with paths["quality"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row["status"] for row in rows} == {"ok", "failed"}
    assert all(row["artifact_id"] for row in rows)
    with paths["speed"].open(encoding="utf-8", newline="") as handle:
        speed_rows = list(csv.DictReader(handle))
    assert speed_rows[0]["pairs_per_second"] == "3.0"
    with paths["paired"].open(encoding="utf-8", newline="") as handle:
        paired_rows = list(csv.DictReader(handle))
    assert len(paired_rows) == 12  # six metrics for each of two valid quality rows
    assert all(row["n"] == "1" for row in paired_rows)
