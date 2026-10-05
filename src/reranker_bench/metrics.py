"""Evaluation helpers for reranker benchmark runs.

Queries with unknown or zero-positive judgments are retained in diagnostics but
are excluded from macro quality denominators.  Relevance is always measured
against the full known qrels for a query, never just the retrieved pool.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


def _validated_qrels(qrels: Mapping[Any, Mapping[Any, Any]]) -> dict[str, dict[str, float]]:
    clean: dict[str, dict[str, float]] = {}
    for qid, judgments in qrels.items():
        if not isinstance(judgments, Mapping):
            raise TypeError(f"qrels for query {qid!r} must be a mapping")
        row: dict[str, float] = {}
        for did, raw in judgments.items():
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"non-finite relevance for query {qid!r}, document {did!r}")
            row[str(did)] = value
        clean[str(qid)] = row
    return clean


def _clean_run(run: Sequence[Any] | None) -> list[str]:
    """Convert to string IDs, retaining the first occurrence of each ID."""
    seen: set[str] = set()
    out: list[str] = []
    for item in run or ():
        doc_id = str(item)
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def _dcg(labels: Sequence[float]) -> float:
    return sum((2.0**rel - 1.0) / math.log2(rank + 2) for rank, rel in enumerate(labels))


def _average_precision(ranked: Sequence[str], positive: set[str]) -> float:
    if not positive:
        return 0.0
    hits = 0
    total = 0.0
    for rank, did in enumerate(ranked, 1):
        if did in positive:
            hits += 1
            total += hits / rank
    return total / len(positive)


def evaluate_run(
    qrels: dict[Any, dict[Any, float]],
    runs: dict[Any, list[str]],
    ks: Sequence[int] = (10, 20, 100, 500, 1000),
    complete: bool = False,
    categories: dict[Any, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a run using global known qrels and macro-average eligible queries.

    ``complete=False`` marks judgments as incomplete: recall remains useful but
    is explicitly labeled known-positive recall, while precision and MAP are
    omitted because unjudged documents must not count as negatives.
    """
    judgments = _validated_qrels(qrels)
    if isinstance(ks, (str, bytes)):
        raise TypeError("ks must be a sequence of positive integers")
    cutoffs = tuple(dict.fromkeys(int(k) for k in ks))
    if any(k <= 0 for k in cutoffs):
        raise ValueError("all cutoffs must be positive integers")
    normalized_runs = {str(qid): _clean_run(run) for qid, run in runs.items()}
    cat_by_qid = {str(qid): str(cat) for qid, cat in (categories or {}).items()}
    all_qids = list(dict.fromkeys([*judgments.keys(), *normalized_runs.keys()]))

    metric_names = [f"ndcg@{k}" for k in cutoffs] + [f"known_positive_recall@{k}" for k in cutoffs] + ["mrr"]
    if complete:
        metric_names += [f"precision@{k}" for k in cutoffs] + [f"map@{k}" for k in cutoffs]
    per_query: dict[str, dict[str, Any]] = {}
    candidate_coverage: dict[str, Any] = {}
    eligible: list[str] = []
    status_counts = {"eligible": 0, "zero_positive": 0, "unknown_qrels": 0}

    for qid in all_qids:
        has_judgments = qid in judgments
        qrel = judgments.get(qid, {})
        positive = {did for did, rel in qrel.items() if rel > 0}
        run = normalized_runs.get(qid, [])
        if not has_judgments:
            status = "unknown_qrels"
        elif not positive:
            status = "zero_positive"
        else:
            status = "eligible"
            eligible.append(qid)
        status_counts[status] += 1
        item: dict[str, Any] = {"status": status, "category": cat_by_qid.get(qid)}
        if status == "eligible":
            for k in cutoffs:
                prefix = run[:k]
                gains = [qrel.get(did, 0.0) for did in prefix]
                ideal = sorted((rel for rel in qrel.values() if rel > 0), reverse=True)[:k]
                idcg = _dcg(ideal)
                item[f"ndcg@{k}"] = _dcg(gains) / idcg if idcg else 0.0
                item[f"known_positive_recall@{k}"] = len(positive.intersection(prefix)) / len(positive)
                if complete:
                    item[f"precision@{k}"] = len(positive.intersection(prefix)) / k
                    item[f"map@{k}"] = _average_precision(prefix, positive)
            first = next((rank for rank, did in enumerate(run, 1) if did in positive), None)
            item["mrr"] = 1 / first if first else 0.0
        per_query[qid] = item
        if has_judgments:
            candidate_coverage[qid] = {
                "known_positive_count": len(positive),
                "candidate_count": len(run),
                "known_positive_candidates": len(positive.intersection(run)),
                "candidate_ceiling_recall": len(positive.intersection(run)) / len(positive) if positive else None,
            }

    aggregate: dict[str, float | None] = {}
    for metric in metric_names:
        values = [per_query[qid][metric] for qid in eligible]
        aggregate[metric] = sum(values) / len(values) if values else None
    category_metrics: dict[str, dict[str, Any]] = {}
    for category in sorted(set(cat_by_qid.values())):
        qids = [qid for qid in eligible if cat_by_qid.get(qid) == category]
        category_metrics[category] = {
            "query_count": len(qids),
            "aggregate": {
                metric: (sum(per_query[qid][metric] for qid in qids) / len(qids) if qids else None)
                for metric in metric_names
            },
        }

    return {
        "aggregate": aggregate,
        "per_query": per_query,
        "categories": category_metrics,
        "limitations": {
            "complete_qrels": bool(complete),
            "macro_query_count": len(eligible),
            "query_status_counts": status_counts,
            "excluded_queries": {qid: per_query[qid]["status"] for qid in all_qids if per_query[qid]["status"] != "eligible"},
            "unjudged_documents": "treated as zero gain for nDCG; not assumed non-relevant in recall",
            "recall_label": "known_positive_recall; completeness depends on qrels",
            "precision_map_available": bool(complete),
        },
        "coverage": {
            "queries_with_run": sum(qid in normalized_runs for qid in all_qids),
            "queries_with_qrels": sum(qid in judgments for qid in all_qids),
            "candidate_coverage": candidate_coverage,
        },
    }


def oracle_run(candidates: Mapping[Any, Sequence[Any]], qrels: Mapping[Any, Mapping[Any, Any]]) -> dict[str, list[str]]:
    """Sort each original candidate pool by known relevance; never inject qrels."""
    judgments = _validated_qrels(qrels)
    output: dict[str, list[str]] = {}
    for qid, pool in candidates.items():
        qkey = str(qid)
        original = _clean_run(pool)
        qrel = judgments.get(qkey, {})
        # Stable sort means equal-relevance documents preserve candidate order.
        output[qkey] = sorted(original, key=lambda did: qrel.get(did, 0.0), reverse=True)
    return output


def paired_bootstrap(
    before: Mapping[Any, Mapping[str, Any]],
    after: Mapping[Any, Mapping[str, Any]],
    metric: str = "ndcg@10",
    seed: int = 42,
    samples: int = 2000,
) -> dict[str, Any]:
    """Percentile 95% CI for the macro mean paired query delta (after-before)."""
    if samples < 1:
        raise ValueError("samples must be positive")
    matched = sorted(set(before).intersection(after), key=str)
    deltas: list[float] = []
    for qid in matched:
        left, right = before[qid].get(metric), after[qid].get(metric)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isfinite(float(left)) and math.isfinite(float(right)):
            deltas.append(float(right) - float(left))
    if not deltas:
        return {"metric": metric, "n": 0, "mean_delta": None, "ci95": [None, None], "samples": samples, "seed": seed}
    rng = random.Random(seed)
    means = [sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(samples)]
    means.sort()
    lower = means[max(0, math.ceil(0.025 * samples) - 1)]
    upper = means[min(samples - 1, math.ceil(0.975 * samples) - 1)]
    return {
        "metric": metric,
        "n": len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "ci95": [lower, upper],
        "samples": samples,
        "seed": seed,
    }
