"""Maintainer: bundle portable Windows Python 3.12 and exact locked CPU wheels."""
from __future__ import annotations
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reranker_bench.common import ROOT, read_json, sha256, write_json

def package(runtime: Path, wheelhouse: Path, destination: Path):
    runtime, wheelhouse = runtime.resolve(), wheelhouse.resolve()
    if not (runtime / "python.exe").is_file() or not (runtime / "LICENSE.txt").is_file():
        raise ValueError("Portable Python runtime and license are required")
    if len(list(wheelhouse.glob("*.whl"))) < 40:
        raise ValueError("Offline wheelhouse seems incomplete")
    files = [(path, "python/" + path.relative_to(runtime).as_posix()) for path in sorted(runtime.rglob("*")) if path.is_file()]
    files += [(path, "wheels/" + path.name) for path in sorted(wheelhouse.glob("*.whl"))]
    requirements = ROOT / "requirements.lock"
    files.append((requirements, "requirements.lock"))
    manifest_files = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path, name in files:
            archive.write(path, name)
            manifest_files.append({"path": name, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    if destination.stat().st_size >= 2 * 1024**3:
        raise ValueError("Offline environment exceeds GitHub Release 2 GiB asset limit")
    asset = {"id": "offline-python-windows-x64", "kind": "runtime", "asset": destination.name,
             "github": {"repository": "saksham-personal/company-reranker-benchmark", "release": "assets-v1"},
             "archive_sha256": sha256(destination), "size_bytes": destination.stat().st_size,
             "files": manifest_files, "install_subdir": "runtime/offline-python-windows-x64"}
    path = ROOT / "configs/assets.json"
    existing = read_json(path) if path.exists() else {"schema_version": 1, "assets": []}
    existing["assets"] = [entry for entry in existing["assets"] if entry["id"] != asset["id"]] + [asset]
    write_json(path, existing)
    print(asset["asset"], asset["size_bytes"], asset["archive_sha256"])

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python-root", type=Path, required=True)
    p.add_argument("--wheelhouse", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    args = p.parse_args()
    package(args.python_root, args.wheelhouse, args.destination)
