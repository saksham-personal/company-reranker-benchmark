from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .datasets import Dataset, DatasetError, tokenize


def _bm25(query: list[str], docs: dict[str, list[str]], *, k1: float = 1.2, b: float = 0.75) -> dict[str, float]:
    n = len(docs)
    lengths = {did: len(tokens) for did, tokens in docs.items()}
    avgdl = sum(lengths.values()) / max(n, 1)
    postings: dict[str, list[tuple[str, int]]] = {}
    for did, tokens in docs.items():
        for term, tf in Counter(tokens).items():
            postings.setdefault(term, []).append((did, tf))
    scores = {did: 0.0 for did in docs}
    for term, qtf in Counter(query).items():
        entries = postings.get(term, [])
        df = len(entries)
        if not df:
            continue
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for did, tf in entries:
            denom = tf + k1 * (1 - b + b * lengths[did] / max(avgdl, 1e-12))
            scores[did] += qtf * idf * tf * (k1 + 1) / denom
    return scores


def build_candidates(dataset: Dataset, representation: str = "description_keywords", depth: int = 1000) -> dict[str, Any]:
    """Create deterministic BM25 candidate pools without consulting qrels."""
    if depth <= 0:
        raise ValueError("depth must be positive")
    docs = {did: tokenize(dataset.text(did, representation)) for did in dataset.corpus}
    # Build postings once: SciFact has 300 queries over 5,183 passages.
    lengths = {did: len(tokens) for did, tokens in docs.items()}
    avgdl = sum(lengths.values()) / max(len(lengths), 1)
    postings: dict[str, list[tuple[str, int]]] = {}
    for did, tokens in docs.items():
        for term, tf in Counter(tokens).items():
            postings.setdefault(term, []).append((did, tf))
    runs: dict[str, list[str]] = {}
    for qid, query in dataset.queries.items():
        scores = {did: 0.0 for did in docs}
        for term, qtf in Counter(tokenize(str(query["text"]))).items():
            entries = postings.get(term, [])
            df = len(entries)
            if not df:
                continue
            idf = math.log(1.0 + (len(docs) - df + 0.5) / (df + 0.5))
            for did, tf in entries:
                denom = tf + 1.2 * (1 - 0.75 + 0.75 * lengths[did] / max(avgdl, 1e-12))
                scores[did] += qtf * idf * tf * (1.2 + 1) / denom
        ranked = sorted(scores, key=lambda did: (-scores[did], did))
        runs[qid] = ranked[:min(depth, len(ranked))]
    return {
        "schema_version": 1,
        "dataset_digest": dataset.digest,
        "representation": representation,
        "retriever": {"name": "BM25", "implementation": "reranker_bench.retrieval", "k1": 1.2, "b": 0.75,
                      "tokenizer": "unicode_word_casefold_v1", "tie_break": "corpus_id_ascending"},
        "retrieve_k": depth,
        "runs": runs,
    }


def load_candidates(path: str | Path, dataset: Dataset, representation: str | None = None) -> dict[str, list[str]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot load candidate manifest: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise DatasetError("candidate manifest schema_version must be 1")
    if payload.get("dataset_digest") != dataset.digest:
        raise DatasetError("candidate manifest dataset digest does not match loaded dataset")
    if representation is not None and payload.get("representation") != representation:
        raise DatasetError("candidate representation mismatch")
    k = payload.get("retrieve_k")
    if not isinstance(k, int) or k <= 0:
        raise DatasetError("retrieve_k must be a positive integer")
    runs = payload.get("runs")
    if not isinstance(runs, dict) or set(runs) != set(dataset.queries):
        raise DatasetError("candidate runs must contain the complete dataset query set")
    result: dict[str, list[str]] = {}
    for qid, ids in runs.items():
        if not isinstance(ids, list) or len(ids) > k or len(ids) > len(dataset.corpus):
            raise DatasetError(f"invalid candidate list or depth for {qid}")
        if any(not isinstance(did, str) or did not in dataset.corpus for did in ids):
            raise DatasetError(f"unknown candidate id in {qid}")
        if len(ids) != len(set(ids)):
            raise DatasetError(f"duplicate candidate id in {qid}")
        result[qid] = ids
    return result


def import_external_runs(path: str | Path, dataset: Dataset, *, representation: str, depth: int = 1000,
                         retriever: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wrap ordered query_id/doc_ids JSONL runs as a frozen candidate manifest."""
    runs: dict[str, list[str]] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{number}: {exc}") from exc
            qid, ids = row.get("query_id"), row.get("doc_ids")
            if qid in runs or qid not in dataset.queries or not isinstance(ids, list):
                raise DatasetError(f"{path}:{number}: invalid or duplicate query_id")
            runs[qid] = [str(did) for did in ids[:depth]]
    return {"schema_version": 1, "dataset_digest": dataset.digest, "representation": representation,
            "retriever": retriever or {"name": "imported_external_run"}, "retrieve_k": depth, "runs": runs}


def save_candidates(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
