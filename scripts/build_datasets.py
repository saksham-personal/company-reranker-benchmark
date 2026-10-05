from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))


def finish(path: Path, dataset_id: str, license_name: str, complete: bool, source: str, diagnostics: bool = False) -> None:
    names = ["corpus.jsonl", "queries.jsonl", "qrels.tsv"]
    hashes = {n: hashlib.sha256((path / n).read_bytes()).hexdigest() for n in names}
    manifest = {"schema_version": 1, "dataset_id": dataset_id, "license": license_name,
                "qrels_complete": complete, "diagnostics": diagnostics, "source": source, "files": hashes}
    (path / "dataset.json").write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def public_dataset(root: Path) -> None:
    source = ROOT / "data" / "public-source-original"
    out = ROOT / "data" / "public-company-pilot"
    companies = [json.loads(x) for x in (source / "companies.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    corpus = [{"_id": r["company_id"], "title": r["name"], "text": r["description"],
               "fields": {**r.get("fields", {}), "keywords": r.get("keywords", []),
                          "source_refs": r.get("source_refs", []), "field_source_refs": r.get("field_source_refs", {})}}
              for r in companies]
    qsource = [json.loads(x) for x in (source / "queries.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    queries = [{"_id": r["query_id"], "text": r["text"], "category": r.get("category"),
                "evaluation_role": r.get("evaluation_role"), "explicit_unknown_ids": r.get("explicit_unknown_ids", [])}
               for r in qsource]
    with (source / "qrels.tsv").open(encoding="utf-8", newline="") as f:
        qrows = list(csv.DictReader(f, delimiter="\t"))
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "corpus.jsonl", corpus)
    write_jsonl(out / "queries.jsonl", queries)
    with (out / "qrels.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["query-id", "corpus-id", "score"])
        for r in qrows:
            w.writerow([r["query_id"], r["company_id"], r["relevance"]])
    (out / "hard_negatives.jsonl").write_bytes((source / "hard_negatives.jsonl").read_bytes())
    # Keep the source's full citation/annotation bundle next to this conversion.
    finish(out, "public-company-screening-v1", "CC-BY-4.0", False,
           "Independent fact summaries and annotations adapted from the Company Embedding Benchmark public-source curation (2026-10-02); see ../public-source-original/README.md, dataset.json, sources.jsonl, rationales.jsonl, bibliography.json.")


PROFILES = [
    ("industrial coatings", "predictive maintenance", "regional manufacturers", "consumer households"),
    ("medical scheduling software", "workflow automation", "community hospitals", "individual patients"),
    ("cold-chain logistics", "temperature monitoring", "food producers", "local consumers"),
    ("industrial filtration", "custom engineering", "chemical plants", "homeowners"),
    ("payroll software", "tax reporting", "multi-site employers", "job seekers"),
    ("commercial roofing", "field installation", "property operators", "residential DIY buyers"),
    ("laboratory testing", "materials analysis", "manufacturers", "school students"),
    ("restaurant procurement", "supplier coordination", "restaurant groups", "home cooks"),
    ("fleet telematics", "route analytics", "delivery companies", "private drivers"),
    ("document printing", "variable-data production", "publishers", "walk-in consumers"),
]


def controlled(root: Path, *, company_count: int = 2400) -> None:
    """Convert the established, labeled 2,400-company / 120-query controlled set."""
    if company_count != 2400:
        raise ValueError("The pinned controlled source contains 2,400 companies; use the complete set")
    source = ROOT / "data" / "controlled-source-original"
    out = ROOT / "data" / "controlled-company-screening"
    companies = [json.loads(line) for line in (source / "companies.jsonl").read_text(encoding="utf-8").splitlines() if line]
    source_queries = [json.loads(line) for line in (source / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(companies) != 2400 or len(source_queries) != 120:
        raise ValueError("Controlled source count changed; inspect provenance before converting")
    corpus = [{"_id": row["company_id"], "title": row["name"], "text": row["description"],
               "fields": {**row.get("fields", {}), "keywords": row.get("keywords", [])},
               "attributes": row.get("attributes", {}), "generation": row.get("generation", {})}
              for row in companies]
    queries = [{"_id": row["query_id"], "text": row["text"], "category": row.get("category"),
                "constraints": row.get("constraints"), "judgement_scope": row.get("judgement_scope")}
               for row in source_queries]
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "corpus.jsonl", corpus)
    write_jsonl(out / "queries.jsonl", queries)
    with (source / "qrels.tsv").open(encoding="utf-8", newline="") as handle:
        source_qrels = list(csv.DictReader(handle, delimiter="\t"))
    with (out / "qrels.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        for row in source_qrels:
            writer.writerow([row["query_id"], row["company_id"], row["relevance"]])
    (out / "hard_negatives.jsonl").write_bytes((source / "hard_negatives.jsonl").read_bytes())
    finish(out, "controlled-company-screening-v1", "CC0-1.0", True,
           "Converted without changing the prior Company Embedding Benchmark controlled-v1 labels, seed 20261002; original source and hard-negative rationale records are in ../controlled-source-original/.")


def diagnostics(root: Path) -> None:
    out = ROOT / "data" / "long-context-adversarial"
    # Every question has positive, excluded, nonmatching and unknown documents.
    # Unknowns are annotations only, never recorded as relevance zero.
    cases = [
        ("explicit_exclusion", "Find industrial filtration for chemical plants; exclude direct sales to consumers.",
         ["Industrial filtration equipment for chemical plants, sold only to industrial operators.",
          "The firm supplies chemical plants with engineered filtration and has no consumer channel."],
         "Industrial filtration for chemical plants is also sold directly through a consumer web shop.",
         "The firm makes industrial filtration equipment, but its customer channels are not described."),
        ("mixed_b2b_b2c", "Find employer payroll software; exclude any company offering consumer tax filing.",
         ["Payroll software for employers only; no personal tax filing product.",
          "B2B employer payroll and tax reporting with explicitly no retail personal tax service."],
         "Payroll software for employers and a direct-to-consumer tax filing app.",
         "The firm offers payroll software; no statement about personal tax filing is available."),
        ("negation", "Find suppliers to manufacturers that do not serve individual consumers.",
         ["Only manufacturers purchase this firm's components. It does not sell to individual consumers.",
          "The supplier serves factories exclusively; household and retail sales are explicitly absent."],
         "The supplier serves manufacturers and individual consumers through an online retail store.",
         "The supplier serves manufacturers; other customer channels are not stated."),
        ("attribute_conjunction", "Find cold-chain food logistics with temperature monitoring for hospitals.",
         ["Food cold-chain logistics for hospitals with continuous temperature monitoring.",
          "The company transports hospital food under monitored temperature controls."],
         "The company ships hospital food but does not provide temperature monitoring.",
         "The company provides food logistics with temperature monitoring; customer types are unstated."),
        ("parent_subsidiary", "Find the subsidiary that directly performs laboratory materials testing for manufacturers.",
         ["This subsidiary directly operates materials testing labs for manufacturers.",
          "The subsidiary performs laboratory testing of manufacturer components using its own staff."],
         "The parent group owns a separate laboratory subsidiary; this entity only supplies accounting services.",
         "The business is a subsidiary of a laboratory group, but its own activities are not specified."),
        ("semantic_capability", "Find industrial coatings suppliers offering predictive maintenance for factories.",
         ["Industrial coatings for factories with predictive maintenance services.",
          "Coating systems for industrial plants include equipment condition forecasting support."],
         "The supplier sells industrial coatings but explicitly does not offer predictive maintenance.",
         "Industrial coating products are described; maintenance service information is missing."),
        ("customer_scope", "Find medical scheduling software sold to community hospitals, excluding direct patient subscriptions.",
         ["Scheduling software is licensed to community hospitals only, with no individual subscriptions.",
          "The hospital workflow platform serves community hospitals and excludes retail patient subscriptions."],
         "Medical scheduling software serves community hospitals and also sells direct patient subscriptions.",
         "Medical scheduling software is described without any sales channel information."),
    ]
    corpus, queries, qrows, decisions = [], [], [], []
    filler = " General company history describes office locations, hiring processes, supplier onboarding, invoicing practices, and routine administrative operations."
    for i, (category, query, positives, exclusion, unknown) in enumerate(cases):
        qid = f"diag-q-{i:02d}"
        queries.append({"_id": qid, "text": query, "category": category,
                        "judgement_scope": "constructed_evidence_only"})
        examples = [(positives[0], "match", 2), (positives[1], "match", 2),
                    (exclusion, "excluded", 0), (unknown, "unknown", None),
                    ("This organization offers general accounting administration for local retailers.", "negative_other", 0),
                    ("The parent organization may have a related product, but this operating company has no stated connection to the required activity.", "unknown", None),
                    ("The company explicitly does not provide the requested service to the requested customers.", "negative_other", 0),
                    ("A legacy website mentions the relevant sector, while current services and customer channels are unstated.", "unknown", None)]
        for j, (evidence, label, score) in enumerate(examples):
            did = f"diag-co-{i:02d}-{j:02d}"
            position = ("head", "middle", "tail")[(i + j) % 3]
            target_filler_chars = (1000, 4000, 12000, 24000)[(i + j) % 4]
            padding = (filler * (target_filler_chars // len(filler) + 1))[:target_filler_chars]
            if position == "head": long_text = evidence + padding
            elif position == "middle": long_text = padding[:len(padding)//2] + evidence + padding[len(padding)//2:]
            else: long_text = padding + evidence
            corpus.append({"_id": did, "title": f"Synthetic {category} case {j}",
                           "text": "Synthetic screening record. Decision evidence is in the full profile.",
                           "fields": {"rich": {"profile": long_text}, "evidence_position": position,
                                      "context_chars": len(long_text), "category": category}})
            decisions.append({"query_id": qid, "doc_id": did, "label": label, "evidence": evidence,
                              "evidence_position": position, "profile_chars": len(long_text),
                              "unknown_is_not_negative": label == "unknown"})
            if score is not None:
                qrows.append((qid, did, score))
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "corpus.jsonl", corpus)
    write_jsonl(out / "queries.jsonl", queries)
    write_jsonl(out / "decision_annotations.jsonl", decisions)
    with (out / "qrels.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["query-id", "corpus-id", "score"])
        w.writerows(qrows)
    finish(out, "long-context-adversarial-v1", "CC0-1.0", False,
           "Synthetic 7-query/56-document diagnostics labeled by construction. Explicit unknown cases are in decision_annotations.jsonl and omitted from qrels; this tests failure modes, not real-world accuracy.", True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--company-count", type=int, default=2400)
    args = parser.parse_args()
    if args.company_count < len(PROFILES):
        parser.error("company-count must be at least the profile count")
    public_dataset(ROOT)
    controlled(ROOT, company_count=args.company_count)
    diagnostics(ROOT)
    print("Built public-company-pilot, controlled-company-screening, and long-context-adversarial")


if __name__ == "__main__":
    main()
