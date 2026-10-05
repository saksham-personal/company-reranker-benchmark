"""Offline benchmark command line."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .adapters import PairScorer, validate_model
from .benchmark import benchmark_quality, benchmark_speed
from .common import hardware, model_catalog, timestamp, write_json
from .datasets import load_dataset

def main(argv=None):
    parser = argparse.ArgumentParser(prog="reranker-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models")
    inspect = sub.add_parser("inspect-machine")
    inspect.add_argument("--role", default="vdi")
    inspect.add_argument("--output", type=Path)
    dataset = sub.add_parser("validate-dataset")
    dataset.add_argument("path", type=Path)
    validate = sub.add_parser("validate-model")
    validate.add_argument("--model", required=True)
    validate.add_argument("--model-root", type=Path, required=True)
    validate.add_argument("--max-length", type=int, default=512)
    validate.add_argument("--threads", type=int, default=4)
    validate.add_argument("--output", type=Path)
    for command in ("quality", "speed"):
        p = sub.add_parser(command)
        p.add_argument("--model", required=True)
        p.add_argument("--model-root", type=Path, required=True)
        p.add_argument("--dataset", type=Path, required=True)
        p.add_argument("--candidates", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--representation", choices=("description_only", "description_keywords", "rich"), default="description_keywords")
        p.add_argument("--rerank-depth", type=int, default=500)
        p.add_argument("--max-length", type=int, default=512)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--query-limit", type=int)
        p.add_argument("--machine-role", default="vdi")
        if command == "speed":
            p.add_argument("--rounds", type=int, default=3)
            p.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "models":
        print(json.dumps(model_catalog(), indent=2))
    elif args.command == "inspect-machine":
        payload = {"type": "hardware", "timestamp_utc": timestamp(), "machine": hardware(args.role)}
        if args.output:
            write_json(args.output, payload)
        print(json.dumps(payload, indent=2))
    elif args.command == "validate-dataset":
        data = load_dataset(args.path)
        print(json.dumps({"dataset_id": data.metadata["dataset_id"], "digest": data.digest,
                          "corpus": len(data.corpus), "queries": len(data.queries),
                          "judgments": sum(map(len, data.qrels.values())),
                          "qrels_complete": data.metadata.get("qrels_complete")}, indent=2))
    elif args.command == "validate-model":
        model = PairScorer(args.model, args.model_root, max_length=args.max_length, threads=args.threads)
        result = validate_model(model)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
        if result["status"] != "passed":
            return 2
    else:
        kwargs = {"model_id": args.model, "model_root": args.model_root,
                  "dataset_path": args.dataset, "candidates_path": args.candidates,
                  "output": args.output, "representation": args.representation,
                  "rerank_depth": args.rerank_depth, "max_length": args.max_length,
                  "threads": args.threads, "batch_size": args.batch_size,
                  "query_limit": args.query_limit, "machine_role": args.machine_role}
        if args.command == "quality":
            result = benchmark_quality(**kwargs)
            summary = result["metrics"]["reranked"]["aggregate"]
        else:
            result = benchmark_speed(**kwargs, rounds=args.rounds, warmups=args.warmups)
            summary = result["speed"]
        print(json.dumps({"status": result["status"], "output": str(args.output), "summary": summary}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
