"""Run a pinned suite serially, in a fresh CPU process per model and stage."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reranker_bench.common import read_json, timestamp, write_json

def run_suite(plan_path: Path, work_dir: Path, results_dir: Path, *, force=False, dry_run=False):
    plan = read_json(plan_path)
    catalog = {m["model_id"] for m in read_json(ROOT / "configs" / "models.json")["models"]}
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
        for model in job["models"]:
            if model not in catalog:
                raise ValueError(f"Unknown model in plan: {model}")
            for stage in ("quality", "speed"):
                options = job.get(stage)
                if not options:
                    continue
                output = results_dir / job_name / f"{model}-{stage}.json"
                if output.exists() and not force:
                    print("REUSE", output, flush=True)
                    outcomes.append({"job": job_name, "model": model, "stage": stage, "status": "reused"})
                    continue
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
                                        "job": job_name, "model_id": model, "dataset_path": str(dataset),
                                        "candidate_path": str(candidates), "timestamp_utc": timestamp(),
                                        "returncode": proc.returncode, "error_tail": proc.stderr[-4000:],
                                        "log": str(log)})
                else:
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
