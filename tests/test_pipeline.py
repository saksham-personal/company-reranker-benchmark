from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from reranker_bench.adapters import rerank_prefix
from reranker_bench.datasets import load_dataset
from reranker_bench.retrieval import load_candidates


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("dataset", "candidates", "corpus_count", "query_count"),
    [
        ("public-company-pilot", "public-company-pilot-bm25.json", 130, 38),
        ("controlled-company-screening", "controlled-company-bm25.json", 2400, 120),
        ("long-context-adversarial", "long-context-bm25.json", 56, 7),
    ],
)
def test_frozen_candidates_match_dataset(dataset, candidates, corpus_count, query_count):
    data = load_dataset(ROOT / "data" / dataset)
    frozen = load_candidates(ROOT / "configs" / "candidates" / candidates, data)
    assert len(data.corpus) == corpus_count
    assert len(data.queries) == len(frozen) == query_count
    assert all(0 < len(ids) <= min(1000, corpus_count) and len(ids) == len(set(ids)) for ids in frozen.values())


def test_public_pilot_is_smaller_than_top_500_workload():
    data = load_dataset(Path(__file__).resolve().parents[1] / "data" / "public-company-pilot")
    assert not data.metadata["qrels_complete"]
    frozen = load_candidates(ROOT / "configs" / "candidates" / "public-company-pilot-bm25.json", data)
    # The 130-document pilot cannot substantiate a top-500 reranking claim.
    assert max(map(len, frozen.values())) == 130


def test_prefix_rerank_preserves_stage_one_ceiling_and_tail():
    original = [f"co-{i}" for i in range(1000)]
    reranked = rerank_prefix(original, [0.0] * 499 + [1.0], 500)
    assert reranked[0] == "co-499"
    assert set(reranked[:500]) == set(original[:500])
    assert reranked[500:] == original[500:]


def test_unknown_long_context_cases_are_not_negative_qrels():
    data = load_dataset(ROOT / "data" / "long-context-adversarial")
    rows = [json.loads(line) for line in
            (ROOT / "data" / "long-context-adversarial" / "decision_annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    unknown = [(row["query_id"], row["doc_id"]) for row in rows if row["label"] == "unknown"]
    assert unknown
    assert all(did not in data.qrels.get(qid, {}) for qid, did in unknown)


def test_annotation_change_invalidates_dataset_digest(tmp_path):
    source = ROOT / "data" / "long-context-adversarial"
    for file in source.iterdir():
        if file.is_file():
            shutil.copy2(file, tmp_path / file.name)
    before = load_dataset(tmp_path).digest
    annotation = tmp_path / "decision_annotations.jsonl"
    annotation.write_bytes(annotation.read_bytes().replace(b'"unknown"', b'"excluded"', 1))
    assert load_dataset(tmp_path).digest != before
