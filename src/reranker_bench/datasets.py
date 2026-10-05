from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetError(ValueError):
    """Invalid benchmark dataset or candidate manifest."""


@dataclass(frozen=True)
class Dataset:
    metadata: dict[str, Any]
    corpus: dict[str, dict[str, Any]]
    queries: dict[str, dict[str, Any]]
    qrels: dict[str, dict[str, float]]
    digest: str

    def text(self, did: str, representation: str = "description_keywords") -> str:
        try:
            row = self.corpus[str(did)]
        except KeyError as exc:
            raise DatasetError(f"unknown corpus id: {did}") from exc
        title = str(row.get("title", "")).strip()
        body = str(row.get("text", "")).strip()
        fields = row.get("fields") or {}
        if representation == "description_only":
            value = body
        elif representation == "title_text":
            value = " ".join(x for x in (title, body) if x)
        elif representation == "description_keywords":
            kws = fields.get("keywords", row.get("keywords", []))
            if isinstance(kws, str):
                kws = [kws]
            value = body + ((" Keywords: " + ", ".join(map(str, kws)) + ".") if kws else "")
        elif representation == "rich":
            extras = fields.get("rich") or fields
            if isinstance(extras, dict):
                details = [str(v).strip() for k, v in sorted(extras.items())
                           if k not in {"keywords", "rich"} and isinstance(v, (str, int, float)) and str(v).strip()]
            else:
                details = []
            chunks = [x for x in (title, body, *details) if x]
            value = " ".join(dict.fromkeys(chunks))
        else:
            raise DatasetError(f"unknown representation: {representation}")
        return value.strip()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{n}: {exc}") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{path}:{n}: expected object")
            rows.append(row)
    return rows


def _digest(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8") + b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def load_dataset(path: str | Path) -> Dataset:
    root = Path(path)
    try:
        metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot load dataset manifest at {root}: {exc}") from exc
    corpus_rows, query_rows = _jsonl(root / "corpus.jsonl"), _jsonl(root / "queries.jsonl")
    corpus: dict[str, dict[str, Any]] = {}
    queries: dict[str, dict[str, Any]] = {}
    for row in corpus_rows:
        key = str(row.get("_id", ""))
        if not key or key in corpus:
            raise DatasetError(f"corpus ids must be present and unique: {key!r}")
        corpus[key] = row
    for row in query_rows:
        key = str(row.get("_id", ""))
        if not key or key in queries:
            raise DatasetError(f"query ids must be present and unique: {key!r}")
        queries[key] = row
    qrels: dict[str, dict[str, float]] = {}
    with (root / "qrels.tsv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"query-id", "corpus-id", "score"}.issubset(reader.fieldnames or []):
            raise DatasetError("qrels.tsv must have query-id, corpus-id, score columns")
        for n, row in enumerate(reader, 2):
            qid, did = row["query-id"], row["corpus-id"]
            try:
                score = float(row["score"])
            except (TypeError, ValueError) as exc:
                raise DatasetError(f"qrels.tsv:{n}: invalid score") from exc
            if not math.isfinite(score):
                raise DatasetError(f"qrels.tsv:{n}: score must be finite")
            if did in qrels.setdefault(qid, {}):
                raise DatasetError(f"duplicate qrel: {qid}/{did}")
            qrels[qid][did] = score
    dataset = Dataset(metadata, corpus, queries, qrels,
                      _digest([root / "dataset.json", root / "corpus.jsonl", root / "queries.jsonl", root / "qrels.tsv"]))
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: Dataset) -> None:
    if not dataset.corpus or not dataset.queries:
        raise DatasetError("corpus and queries must be nonempty")
    for did, row in dataset.corpus.items():
        if not str(row.get("text", "")).strip():
            raise DatasetError(f"corpus {did} has empty text")
    for qid, row in dataset.queries.items():
        if not str(row.get("text", "")).strip():
            raise DatasetError(f"query {qid} has empty text")
    for qid, labels in dataset.qrels.items():
        if qid not in dataset.queries:
            raise DatasetError(f"qrels reference unknown query: {qid}")
        for did, score in labels.items():
            if did not in dataset.corpus:
                raise DatasetError(f"qrels reference unknown corpus id: {did}")
            if not math.isfinite(score):
                raise DatasetError(f"non-finite qrel score: {qid}/{did}")


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]
