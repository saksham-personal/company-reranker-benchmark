"""Maintainer: convert the pinned BEIR SciFact test ZIP to this kit's dataset format."""
from __future__ import annotations
import argparse
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reranker_bench.common import ROOT, read_json, sha256, write_json
from import_beir import import_beir
from prepare_assets import package

SOURCE_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"

def prepare(archive_path: Path, work_dir: Path):
    if sha256(archive_path) != SOURCE_SHA256:
        raise ValueError("BEIR SciFact source ZIP hash mismatch")
    extract_root = work_dir / "sources" / "scifact"
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if not parts or parts[0] != "scifact" or ".." in parts or "\\" in member.filename:
                raise ValueError("Unsafe SciFact ZIP path")
            target = (extract_root / member.filename).resolve()
            if not target.is_relative_to(extract_root.resolve()):
                raise ValueError("SciFact ZIP escapes extraction directory")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as sink:
                    import shutil
                    shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
    source = extract_root / "scifact"
    out = work_dir / "datasets" / "beir-scifact"
    import_beir(source, out, "beir-scifact-test", "corpus: ODC-By-1.0; queries and qrels: CC-BY-4.0", SOURCE_SHA256)
    (out / "ATTRIBUTION.md").write_text(
        "# SciFact test data\n\nSciFact: Wadden et al., Fact or Fiction: Verifying Scientific Claims (EMNLP 2020). "
        "BEIR: Thakur et al., BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models (NeurIPS 2021). "
        "Source: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip. "
        "Corpus ODC-By-1.0; queries/qrels CC-BY-4.0 per https://github.com/allenai/scifact/blob/master/LICENSE.md. "
        "The source archive SHA-256 is " + SOURCE_SHA256 + ".\n", encoding="utf-8")
    asset = package(out, work_dir / "packages" / "beir-scifact-test.zip", "beir-scifact", "dataset")
    path = ROOT / "configs" / "assets.json"
    current = read_json(path)
    current["assets"] = [entry for entry in current["assets"] if entry["id"] != asset["id"]] + [asset]
    write_json(path, current)
    print(json.dumps({"asset": asset["asset"], "size_bytes": asset["size_bytes"], "sha256": asset["archive_sha256"]}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_archive, args.work_dir)
