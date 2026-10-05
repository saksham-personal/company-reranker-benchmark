"""Two-stage CPU evaluation with frozen stage-one candidates."""
from __future__ import annotations
import math
import os
import statistics
import threading
import time
from pathlib import Path

from .adapters import PairScorer, rerank_prefix, validate_model
from .common import hardware, read_json, sha256, timestamp, write_json
from .datasets import load_dataset
from .metrics import evaluate_run
from .retrieval import load_candidates

def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]

def _query_ids(dataset, query_limit):
    ids = sorted(dataset.queries)
    if query_limit is not None:
        if query_limit <= 0:
            raise ValueError("query_limit must be positive")
        ids = ids[:query_limit]
    return ids

def _load(dataset_path, candidates_path, representation):
    dataset = load_dataset(dataset_path)
    # The same frozen stage-one pool can be reranked using either sparse or rich text.
    candidates = load_candidates(candidates_path, dataset)
    dataset.text(next(iter(dataset.corpus)), representation)  # fail early on unknown representation
    return dataset, candidates

def _metadata(model_id, scorer, dataset, candidates_path, representation,
              query_count, rerank_depth, retrieve_depth, batch_size, machine_role):
    observed_hardware = hardware(machine_role)
    candidate_manifest = read_json(candidates_path)
    stable_hardware = {key: observed_hardware[key] for key in
                       ("role", "hostname", "os", "architecture", "processor", "logical_cores", "physical_cores", "ram_bytes", "python", "packages")}
    return {"schema_version": 1, "model_id": model_id, "artifact_id": model_id,
            "dataset_id": dataset.metadata["dataset_id"], "dataset_digest": dataset.digest,
            "candidate_digest": sha256(candidates_path), "representation": representation,
            "candidate_representation": candidate_manifest.get("representation"),
            "retriever": candidate_manifest.get("retriever"),
            "runtime": "pytorch-cpu", "precision": "fp32", "query_count": query_count,
            "rerank_depth": rerank_depth, "retrieve_depth": retrieve_depth,
            "max_length": scorer.max_length, "threads": scorer.threads, "batch_size": batch_size,
            "machine": stable_hardware, "available_ram_at_start_bytes": observed_hardware["available_ram_bytes"],
            "model_revision": scorer.info["revision"],
            "score_semantics": scorer.score_semantics, "timestamp_utc": timestamp(),
            "pair_truncation_policy": "tokenizer truncation=True, max_length exact; query-document pair",
            "offline_environment": {key: os.environ.get(key) for key in
                                    ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}}

def benchmark_quality(*, model_id, model_root, dataset_path, candidates_path, output,
                      representation="description_keywords", rerank_depth=500,
                      max_length=512, threads=4, batch_size=8, query_limit=None,
                      machine_role="development", require_validation=True):
    dataset, candidates = _load(dataset_path, candidates_path, representation)
    qids = _query_ids(dataset, query_limit)
    if not qids:
        raise ValueError("No queries")
    if rerank_depth < 1:
        raise ValueError("rerank_depth must be positive")
    scorer = PairScorer(model_id, model_root, max_length=max_length, threads=threads)
    gate = validate_model(scorer) if require_validation else {"status": "skipped", "reason": "explicitly bypassed"}
    if require_validation and gate["status"] != "passed":
        raise RuntimeError(f"Model semantic smoke gate failed: {gate}")
    baseline, reranked, oracle = {}, {}, {}
    token_count, truncated_pairs, score_times = [], 0, {}
    per_query_preview = {}
    annotated = {}
    annotation_file = Path(dataset_path) / "decision_annotations.jsonl"
    if annotation_file.is_file():
        import json
        for line in annotation_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                annotated[(row["query_id"], row["doc_id"])] = row["label"]
    hard_negative_file = Path(dataset_path) / "hard_negatives.jsonl"
    hard_negative_ids = {}
    if hard_negative_file.is_file():
        import json
        for line in hard_negative_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                hard_negative_ids.setdefault(row["query_id"], set()).add(row["company_id"])
    for qid in qids:
        original = candidates[qid]
        if not original:
            baseline[qid] = reranked[qid] = oracle[qid] = []
            continue
        depth = min(rerank_depth, len(original))
        query = dataset.queries[qid]["text"]
        pairs = [(query, dataset.text(did, representation)) for did in original[:depth]]
        pair_tokens = scorer.token_lengths(pairs)
        token_count.extend(pair_tokens)
        truncated_pairs += sum(n > max_length for n in pair_tokens)
        start = time.perf_counter()
        scores = scorer.score(pairs, batch_size=batch_size)
        ordered = rerank_prefix(original, scores, depth)
        score_times[qid] = (time.perf_counter() - start) * 1000
        judgments = dataset.qrels.get(qid, {})
        oracle_prefix = sorted(range(depth), key=lambda i: -judgments.get(original[i], 0.0))
        baseline[qid] = list(original)
        reranked[qid] = ordered
        oracle[qid] = [original[i] for i in oracle_prefix] + list(original[depth:])
        per_query_preview[qid] = {"query": query, "stage1_top10": list(original[:10]),
                                  "reranked_top10": ordered[:10], "oracle_top10": oracle[qid][:10],
                                  "original_prefix_length": depth,
                                  "scores_top10": [{"doc_id": did, "raw_score": scores[original.index(did)]}
                                                   for did in ordered[:min(10, depth)]]}
    eval_kwargs = {"complete": bool(dataset.metadata.get("qrels_complete", False)),
                   "categories": {qid: dataset.queries[qid].get("category", "unclassified") for qid in qids}}
    selected_qrels = {qid: dataset.qrels[qid] for qid in qids if qid in dataset.qrels}
    metrics = {"baseline": evaluate_run(selected_qrels, baseline, **eval_kwargs),
               "reranked": evaluate_run(selected_qrels, reranked, **eval_kwargs),
               "oracle": evaluate_run(selected_qrels, oracle, **eval_kwargs)}
    diagnostic_counts = {}
    for cutoff in (10, 20, 100):
        diagnostic_counts[f"excluded_in_top{cutoff}"] = {
            "baseline": sum(sum(did in hard_negative_ids.get(qid, set()) or
                                 annotated.get((qid, did)) == "excluded" for did in baseline[qid][:cutoff]) for qid in baseline),
            "reranked": sum(sum(did in hard_negative_ids.get(qid, set()) or
                                 annotated.get((qid, did)) == "excluded" for did in reranked[qid][:cutoff]) for qid in reranked)}
        diagnostic_counts[f"unknown_in_top{cutoff}"] = {
            "baseline": sum(sum(annotated.get((qid, did)) == "unknown" for did in baseline[qid][:cutoff]) for qid in baseline),
            "reranked": sum(sum(annotated.get((qid, did)) == "unknown" for did in reranked[qid][:cutoff]) for qid in reranked)}
    retrieve_depth = max((len(ids) for ids in candidates.values()), default=0)
    result = {**_metadata(model_id, scorer, dataset, candidates_path, representation, len(qids),
                          rerank_depth, retrieve_depth, batch_size, machine_role),
              "type": "quality", "status": "passed" if len(qids) == len(dataset.queries) else "exploratory_partial_queries",
              "validation": gate, "metrics": metrics, "per_query_timings": score_times,
              "ranked_ids": {"baseline": baseline, "reranked": reranked},
              "constraint_diagnostics": diagnostic_counts,
              "elapsed_s": sum(score_times.values()) / 1000,
              "per_query_preview": per_query_preview,
              "measured_pair_count": len(token_count), "truncated_pair_count": truncated_pairs,
              "truncated_pair_fraction": truncated_pairs / len(token_count) if token_count else 0,
              "pair_token_length": {"p50": _percentile(token_count, .5), "p95": _percentile(token_count, .95),
                                    "max": max(token_count) if token_count else None},
              "invariants": {"score_original_prefix_only": True,
                             "untouched_tail_after_prefix": True,
                             "recall_at_rerank_depth_cannot_improve_without_filtering": True},
              "limitations": ["Public company pilot has incomplete pooled judgments; no production ranking claim",
                              "Synthetic company labels are regression evidence only",
                              "SciFact measures scientific retrieval, not company screening"]}
    write_json(output, result)
    return result

def benchmark_speed(*, model_id, model_root, dataset_path, candidates_path, output,
                    representation="description_keywords", rerank_depth=500,
                    max_length=512, threads=4, batch_size=8, query_limit=20,
                    machine_role="development", rounds=3, warmups=1):
    if rounds < 1 or warmups < 0:
        raise ValueError("rounds must be >=1 and warmups >=0")
    dataset, candidates = _load(dataset_path, candidates_path, representation)
    qids = _query_ids(dataset, query_limit)
    load_start = time.perf_counter()
    scorer = PairScorer(model_id, model_root, max_length=max_length, threads=threads)
    cold_load_s = time.perf_counter() - load_start
    gate = validate_model(scorer)
    if gate["status"] != "passed":
        raise RuntimeError(f"Model semantic smoke gate failed: {gate}")
    workloads, token_count = [], []
    for qid in qids:
        ids = candidates[qid][:rerank_depth]
        query = dataset.queries[qid]["text"]
        pairs = [(query, dataset.text(did, representation)) for did in ids]
        workloads.append((qid, ids, pairs))
        token_count.extend(scorer.token_lengths(pairs))
    if not any(pairs for _, _, pairs in workloads):
        raise ValueError("No candidate pairs to score")
    peak_rss = [0]
    stop = threading.Event()
    def sample_memory():
        import psutil
        process = psutil.Process()
        while not stop.is_set():
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            stop.wait(.02)
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        for _ in range(warmups):
            for _, _, pairs in workloads:
                if pairs:
                    scorer.score(pairs, batch_size=batch_size)
        latencies, total_pairs, total_seconds = [], 0, 0.0
        per_query = {qid: [] for qid, _, _ in workloads}
        for _ in range(rounds):
            for qid, ids, pairs in workloads:
                if not pairs:
                    continue
                start = time.perf_counter()
                scores = scorer.score(pairs, batch_size=batch_size)
                rerank_prefix(ids, scores, len(ids))
                elapsed = time.perf_counter() - start
                latencies.append(elapsed * 1000)
                per_query[qid].append(elapsed * 1000)
                total_pairs += len(pairs)
                total_seconds += elapsed
    finally:
        stop.set()
        sampler.join(timeout=1)
    retrieve_depth = max((len(ids) for ids in candidates.values()), default=0)
    result = {**_metadata(model_id, scorer, dataset, candidates_path, representation, len(qids),
                          rerank_depth, retrieve_depth, batch_size, machine_role),
              "type": "speed", "status": "passed" if len(qids) == len(dataset.queries) and rounds >= 3 else "exploratory_short_run",
              "validation": gate, "speed": {"pairs_per_second": total_pairs / total_seconds,
                                             "query_latency_p50_ms": _percentile(latencies, .5),
                                             "query_latency_p95_ms": _percentile(latencies, .95),
                                             "peak_rss_mb": peak_rss[0] / (1024 ** 2),
                                             "cold_load_s": cold_load_s,
                                             "total_measured_pairs": total_pairs,
                                             "total_measured_seconds": total_seconds,
                                             "rounds": rounds, "warmups": warmups,
                                             "query_latency_samples_ms": per_query,
                                             "token_histogram": {"p50": _percentile(token_count, .5),
                                                                 "p95": _percentile(token_count, .95),
                                                                 "max": max(token_count) if token_count else None,
                                                                 "truncated_fraction": sum(n > max_length for n in token_count) / len(token_count) if token_count else 0}},
              "limitations": ["Latency includes tokenization, CPU forward pass and prefix sorting; excludes candidate retrieval",
                              "Cold load starts a fresh model object but does not flush OS filesystem cache",
                              "Peak RSS is 20-ms sampled current process memory; missed spikes are possible"]}
    write_json(output, result)
    return result
