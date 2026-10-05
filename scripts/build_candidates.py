from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reranker_bench.datasets import load_dataset
from reranker_bench.retrieval import build_candidates, import_external_runs, save_candidates, load_candidates


def main() -> None:
    p = argparse.ArgumentParser(description="Freeze candidate pools for reranker evaluation.")
    p.add_argument("dataset", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--representation", default="description_keywords", choices=("description_only", "description_keywords", "rich"))
    p.add_argument("--depth", type=int, default=1000)
    p.add_argument("--external-jsonl", type=Path, help="Ordered JSONL rows: {query_id, doc_ids:[...]}; no qrel-positive injection")
    a = p.parse_args()
    dataset = load_dataset(a.dataset)
    if a.external_jsonl:
        payload = import_external_runs(a.external_jsonl, dataset, representation=a.representation, depth=a.depth)
        payload["retriever"].update({"source_file": a.external_jsonl.name, "ordered_input": True})
    else:
        payload = build_candidates(dataset, a.representation, a.depth)
    save_candidates(payload, a.output)
    loaded = load_candidates(a.output, dataset, a.representation)
    print(f"Saved {len(loaded)} query runs to {a.output}")


if __name__ == "__main__":
    main()
