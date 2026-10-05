from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{n}: expected object")
                rows.append(row)
    return rows


def import_beir(source: Path, output: Path, dataset_id: str, license_name: str, revision: str) -> None:
    corpus = read_jsonl(source / "corpus.jsonl")
    query_rows = read_jsonl(source / "queries.jsonl")
    qrel_path = source / "qrels" / "test.tsv"
    with qrel_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames or []):
            raise ValueError("BEIR qrels/test.tsv needs query-id, corpus-id, score columns")
        qrels = list(reader)
    qids = {str(row["query-id"]) for row in qrels}
    queries = [row for row in query_rows if str(row.get("_id")) in qids]
    if len(queries) != len(qids):
        raise ValueError("qrels refer to query ids missing from queries.jsonl")
    if any("_id" not in row for row in corpus):
        raise ValueError("corpus rows must use BEIR _id")
    output.mkdir(parents=True, exist_ok=True)
    names = ["corpus.jsonl", "queries.jsonl", "qrels.tsv"]
    (output / names[0]).write_bytes("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in corpus).encode("utf-8"))
    (output / names[1]).write_bytes("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in queries).encode("utf-8"))
    with (output / names[2]).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["query-id", "corpus-id", "score"])
        for row in qrels:
            w.writerow([row["query-id"], row["corpus-id"], row["score"]])
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names}
    manifest = {"schema_version": 1, "dataset_id": dataset_id, "license": license_name,
                "qrels_complete": False, "diagnostics": False, "source": "BEIR extracted release",
                "source_revision": revision, "files": hashes,
                "notes": "Qrels contain assessed pairs only; all absent qrels are unknown. Import performs no downloads."}
    (output / "dataset.json").write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="Import an already extracted BEIR-format dataset; no downloads.")
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--revision", required=True)
    a = p.parse_args()
    import_beir(a.source, a.output, a.dataset_id, a.license, a.revision)


if __name__ == "__main__":
    main()
