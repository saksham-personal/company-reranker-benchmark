"""Build transparent CSV and Markdown summaries from raw benchmark result JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .metrics import paired_bootstrap


QUALITY_FIELDS = (
    "model_id", "artifact_id", "dataset_id", "dataset_digest", "candidate_digest",
    "representation", "runtime", "query_count", "rerank_depth", "retrieve_depth",
    "max_length", "threads", "batch_size", "machine",
)
SPEED_FIELDS = (
    "model_id", "artifact_id", "dataset_id", "dataset_digest", "candidate_digest",
    "representation", "runtime", "query_count", "rerank_depth", "retrieve_depth",
    "max_length", "threads", "batch_size", "machine", "pairs_per_second",
    "query_latency_p50_ms", "query_latency_p95_ms", "peak_rss_mb", "token_histogram", "elapsed_s",
)
COMPARE_FIELDS = (
    "dataset_id", "dataset_digest", "candidate_digest", "machine", "runtime",
    "threads", "batch_size", "max_length", "rerank_depth", "retrieve_depth",
)


def _jsonable(value: Any) -> Any:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) if isinstance(value, (dict, list)) else value


def _read_results(directory: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            entries.append({"type": "invalid", "status": "invalid_json", "_path": str(path), "_error": str(exc)})
            continue
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if isinstance(row, dict):
                entries.append({**row, "_path": str(path)})
            else:
                entries.append({"type": "invalid", "status": "invalid_row", "_path": str(path)})
    return entries


def _csv_write(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fields})


def _quality_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in entries:
        if row.get("type") != "quality":
            continue
        flat = {key: row.get(key) for key in QUALITY_FIELDS}
        flat.update({"type": "quality", "status": row.get("status", "unknown"), "source_file": row["_path"]})
        flat["_metrics"] = row.get("metrics")
        flat["per_query_timings"] = row.get("per_query_timings")
        for stage in ("baseline", "reranked", "oracle"):
            values = row.get("metrics", {}).get(stage, {}) if isinstance(row.get("metrics"), dict) else {}
            for metric, score in (values.get("aggregate", {}) if isinstance(values, dict) else {}).items():
                flat[f"{stage}_{metric}"] = score
            flat[f"{stage}_eligible_query_count"] = (values.get("limitations", {}).get("macro_query_count") if isinstance(values, dict) else None)
            flat[f"{stage}_query_status_counts"] = (values.get("limitations", {}).get("query_status_counts") if isinstance(values, dict) else None)
        result.append(flat)
    return result


def _speed_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in entries:
        if row.get("type") != "speed":
            continue
        speed = row.get("speed", row)
        flat = {key: row.get(key) for key in SPEED_FIELDS}
        if isinstance(speed, dict):
            aliases = {"pairs_per_second": ("pairs_per_second", "pairs_s"), "query_latency_p50_ms": ("query_latency_p50_ms", "latency_p50_ms", "p50_ms"), "query_latency_p95_ms": ("query_latency_p95_ms", "latency_p95_ms", "p95_ms"), "peak_rss_mb": ("peak_rss_mb", "peak_rss"), "token_histogram": ("token_histogram", "tokens")}
            for target, keys in aliases.items():
                if flat.get(target) is None:
                    flat[target] = next((speed[key] for key in keys if speed.get(key) is not None), None)
        flat.update({"type": "speed", "status": row.get("status", "unknown"), "source_file": row["_path"]})
        result.append(flat)
    return result


def _comparison_key(row: dict[str, Any]) -> tuple[str, ...]:
    # Canonicalized structured machine specifications avoid dict ordering issues.
    return tuple(str(_jsonable(row.get(field))) for field in COMPARE_FIELDS)


def _paired_rows(quality: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in quality:
        if row.get("status") not in ("ok", "success", "completed", "complete", "passed"):
            continue
        metrics = row.get("_metrics")
        if not isinstance(metrics, dict):
            continue
        baseline = metrics.get("baseline", {}).get("per_query", {})
        reranked = metrics.get("reranked", {}).get("per_query", {})
        if not isinstance(baseline, dict) or not isinstance(reranked, dict):
            continue
        for metric in ("ndcg@10", "ndcg@20", "ndcg@100", "known_positive_recall@10", "known_positive_recall@20", "known_positive_recall@100"):
            ci = paired_bootstrap(baseline, reranked, metric=metric)
            rows.append({
                **{key: row.get(key) for key in QUALITY_FIELDS},
                "model_id": row.get("model_id"), "artifact_id": row.get("artifact_id"),
                "metric": metric, **ci, "comparison_key": _comparison_key(row),
                "source_file": row.get("source_file"),
            })
    return rows


def _reference_text(config_path: Path) -> str:
    if not config_path.exists():
        return "`configs/models.json` was not present when this report was generated."
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"Could not parse `configs/models.json`: {exc}"
    # Keep public reference data in its own section. The input file remains the authority.
    return "```json\n" + json.dumps(raw, indent=2, ensure_ascii=False) + "\n```"


def generate_report(results_dir: str | Path, output_dir: str | Path, project_root: str | Path | None = None) -> dict[str, Path]:
    """Scan raw results and write REPORT.md plus quality, speed and paired CSVs."""
    results_path = Path(results_dir).resolve()
    output_path = Path(output_dir).resolve()
    root = Path(project_root).resolve() if project_root else results_path.parent
    output_path.mkdir(parents=True, exist_ok=True)
    entries = _read_results(results_path)
    quality = _quality_rows(entries)
    speed = _speed_rows(entries)
    paired = _paired_rows(quality)
    # Include raw status/identity even for malformed or skipped records.
    quality_columns = [*QUALITY_FIELDS, "type", "status", "source_file", "per_query_timings"]
    for row in quality:
        for key in row:
            if key not in quality_columns:
                quality_columns.append(key)
    speed_columns = [*SPEED_FIELDS, "type", "status", "source_file"]
    paired_columns = [*QUALITY_FIELDS, "metric", "n", "mean_delta", "ci95", "samples", "seed", "comparison_key", "source_file"]
    q_path, s_path, p_path = output_path / "quality.csv", output_path / "speed.csv", output_path / "paired.csv"
    _csv_write(q_path, quality, quality_columns)
    _csv_write(s_path, speed, speed_columns)
    _csv_write(p_path, paired, paired_columns)

    valid = [r for r in quality if r.get("status") in ("ok", "success", "completed", "complete", "passed")]
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in valid:
        grouped.setdefault(_comparison_key(row), []).append(row)
    mixed = len(grouped) > 1
    statuses: dict[str, int] = {}
    for row in entries:
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    lines = [
        "# Company reranker benchmark report", "",
        "This report summarizes raw benchmark records. It does not select a production winner. Quality comparisons are valid only within groups sharing dataset identity, candidate pool, machine, runtime, and runtime budget fields.", "",
        f"- Raw records: {len(entries)}", f"- Quality rows: {len(quality)}; speed rows: {len(speed)}", f"- Comparable quality groups: {len(grouped)}", f"- Mixed comparison conditions: {'yes' if mixed else 'no'}", "",
        "## Status", "", "| Status | Rows |", "|---|---:|",
    ]
    lines.extend(f"| {status} | {count} |" for status, count in sorted(statuses.items()))
    lines.extend(["", "## Local quality results", "", "Each row in `quality.csv` preserves model/artifact and dataset/candidate identity, status, stage metrics, and per-query timing data. Failed and skipped rows remain visible. `paired.csv` gives within-run baseline-to-reranked paired query deltas with a percentile bootstrap 95% interval.", ""])
    if mixed:
        lines.append("The valid rows contain multiple comparison keys. Do not rank or directly compare rows across these groups; inspect `quality.csv` for the exact dataset digests, candidate digests, machine, runtime, thread count, batch size, sequence limit, and depths.")
    elif grouped:
        lines.append("All successful quality rows share the comparison fields listed above.")
    else:
        lines.append("No successful quality rows were available for comparison.")
    lines.extend(["", "## Speed results", "", "See `speed.csv` for pairs per second, query latency percentiles, peak memory, token histograms, and hardware/runtime identity. Interpret timings only under their recorded thread, batch, length, and machine settings.", "", "## Public reference scores", "", "The data below is copied separately from `configs/models.json`; these published scores use different datasets/protocols and are contextual references, not local benchmark results.", "", _reference_text(root / "configs" / "models.json"), "", "## Files", "", "- `quality.csv`: local quality rows and status", "- `speed.csv`: local speed rows and status", "- `paired.csv`: paired reranking deltas and confidence intervals", "- Raw JSON files remain the provenance source", ""])
    report_path = output_path / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"report": report_path, "quality": q_path, "speed": s_path, "paired": p_path}
