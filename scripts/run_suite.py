"""Run a pinned suite serially, in a fresh CPU process per model and stage."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reranker_bench.adapters import verify_model_files
from reranker_bench.common import hardware, read_json, sha256, timestamp, write_json
from reranker_bench.datasets import load_dataset

CODE_FILES = ("adapters.py", "benchmark.py", "cli.py", "common.py", "datasets.py", "metrics.py", "retrieval.py", "reporting.py")


def _run_signature(command: list[str], dataset_digest: str, candidate_digest: str,
                   model_asset_sha: str, code_digest: str, model_config: dict,
                   machine: dict) -> str:
    payload = {"command": command, "dataset_digest": dataset_digest,
               "candidate_digest": candidate_digest, "model_asset_sha": model_asset_sha,
               "code_digest": code_digest, "model_config": model_config, "machine": machine,
               "python": sys.version,
               "packages": {name: importlib.metadata.version(name) for name in
                            ("torch", "transformers", "sentence-transformers", "numpy")}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _reusable(output: Path, signature: str, stage: str) -> bool:
    try:
        row = read_json(output)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    allowed = {"quality": {"passed", "exploratory_partial_queries"},
               "speed": {"passed", "exploratory_short_run"}}[stage]
    return isinstance(row, dict) and row.get("type") == stage and row.get("status") in allowed and row.get("suite_run_signature") == signature

def run_suite(plan_path: Path, work_dir: Path, results_dir: Path, *, force=False, dry_run=False):
    plan = read_json(plan_path)
    catalog = {m["model_id"]: m for m in read_json(ROOT / "configs" / "models.json")["models"]}
    assets = {a["id"]: a for a in read_json(ROOT / "configs" / "assets.json")["assets"] if a["kind"] == "model"}
    code_digest = hashlib.sha256(("".join(sha256(ROOT / "src" / "reranker_bench" / name)
                                             for name in CODE_FILES) + sha256(Path(__file__))).encode()).hexdigest()
    observed = hardware(plan.get("machine_role", "vdi"))
    stable_machine = {key: observed[key] for key in
                      ("role", "hostname", "os", "architecture", "processor", "logical_cores",
                       "physical_cores", "ram_bytes", "python", "packages")}
    verified_models: set[str] = set()
    if plan.get("schema_version") != 1 or not isinstance(plan.get("jobs"), list):
        raise ValueError("Invalid suite plan")
    results_dir.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for job in plan["jobs"]:
        job_name = job["name"]
        dataset = ((work_dir / job["dataset"][len("{work}/"):]) if job["dataset"].startswith("{work}/")
                   else (ROOT / job["dataset"])).resolve()
        candidates = (ROOT / job["candidates"]).resolve()
        if not dataset.is_dir() or not candidates.is_file():
            raise FileNotFoundError(f"Missing dataset/candidates for {job_name}")
        dataset_digest = load_dataset(dataset).digest
        candidate_digest = sha256(candidates)
        for model in job["models"]:
            if model not in catalog:
                raise ValueError(f"Unknown model in plan: {model}")
            for stage in ("quality", "speed"):
                options = job.get(stage)
                if not options:
                    continue
                output = results_dir / job_name / f"{model}-{stage}.json"
                cmd = [sys.executable, "-m", "reranker_bench.cli", stage,
                       "--model", model, "--model-root", str(work_dir / "models"),
                       "--dataset", str(dataset), "--candidates", str(candidates), "--output", str(output),
                       "--representation", job.get("representation", "description_keywords"),
                       "--rerank-depth", str(options.get("rerank_depth", 500)),
                       "--max-length", str(options.get("max_length", 512)),
                       "--threads", str(options.get("threads", 8)),
                       "--batch-size", str(options.get("batch_size", 8)),
                       "--machine-role", str(plan.get("machine_role", "vdi"))]
                if options.get("query_limit"):
                    cmd += ["--query-limit", str(options["query_limit"])]
                if stage == "speed":
                    cmd += ["--rounds", str(options.get("rounds", 3)),
                            "--warmups", str(options.get("warmups", 1))]
                signature = _run_signature(cmd, dataset_digest, candidate_digest,
                                           assets[model]["archive_sha256"], code_digest,
                                           catalog[model], stable_machine)
                if output.exists() and not force and _reusable(output, signature, stage):
                    if model not in verified_models:
                        verify_model_files(model, work_dir / "models")
                        verified_models.add(model)
                    print("REUSE", output, flush=True)
                    outcomes.append({"job": job_name, "model": model, "stage": stage, "status": "reused"})
                    continue
                print("RUN", job_name, model, stage, flush=True)
                if dry_run:
                    outcomes.append({"job": job_name, "model": model, "stage": stage, "status": "dry_run", "command": cmd})
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": str(options.get("threads", 8)),
                            "MKL_NUM_THREADS": str(options.get("threads", 8)), "PYTHONPATH": str(ROOT / "src")})
                proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
                log = output.with_suffix(".log")
                log.write_text("COMMAND " + " ".join(cmd) + "\n\nSTDOUT\n" + proc.stdout +
                               "\nSTDERR\n" + proc.stderr, encoding="utf-8")
                if proc.returncode:
                    write_json(output, {"schema_version": 1, "type": stage, "status": "failed",
                                        "suite_run_signature": signature,
                                        "job": job_name, "model_id": model, "dataset_path": str(dataset),
                                        "candidate_path": str(candidates), "timestamp_utc": timestamp(),
                                        "returncode": proc.returncode, "error_tail": proc.stderr[-4000:],
                                        "log": str(log)})
                else:
                    result = read_json(output)
                    result["suite_run_signature"] = signature
                    write_json(output, result)
                    print(proc.stdout[-350:].strip(), flush=True)
                outcomes.append({"job": job_name, "model": model, "stage": stage,
                                 "status": "passed" if proc.returncode == 0 else "failed", "output": str(output)})
    if not dry_run:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_report.py"),
                        "--results", str(results_dir), "--output", str(ROOT / "reports")],
                       cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=True)
    return outcomes

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=ROOT / "configs" / "suite-vdi.json")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--results", type=Path, default=ROOT / "results" / "raw")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    print(json.dumps(run_suite(args.plan, args.work_dir, args.results, force=args.force, dry_run=args.dry_run), indent=2))
