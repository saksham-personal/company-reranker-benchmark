"""Download checksum-pinned GitHub Release assets; never contacts Hugging Face."""
from __future__ import annotations
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reranker_bench.common import ROOT, read_json, sha256

def safe_extract(archive_path, target_dir, files):
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    expected = {entry["path"]: entry for entry in files}
    with zipfile.ZipFile(archive_path) as archive:
        members = {info.filename: info for info in archive.infolist() if not info.is_dir()}
        if set(members) != set(expected):
            raise ValueError("Archive file list does not match signed manifest")
        for name, info in members.items():
            parts = PurePosixPath(name).parts
            if (name.startswith("/") or ".." in parts or ":" in name or "\\" in name or
                    (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f"Unsafe archive member: {name}")
            destination = (target_dir / name).resolve()
            if not destination.is_relative_to(target_dir):
                raise ValueError(f"Archive path escapes target: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp = destination.with_suffix(destination.suffix + ".partial")
            with archive.open(info) as source, temp.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
            if temp.stat().st_size != expected[name]["size_bytes"] or sha256(temp) != expected[name]["sha256"]:
                temp.unlink(missing_ok=True)
                raise ValueError(f"Extracted file hash mismatch: {name}")
            temp.replace(destination)

def install_asset(item, work_dir, offline=False):
    work_dir = Path(work_dir)
    archive = work_dir / "downloads" / item["asset"]
    if not archive.is_file() or sha256(archive) != item["archive_sha256"]:
        if offline:
            raise FileNotFoundError(f"Verified local archive missing: {archive}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        source = f"https://github.com/{item['github']['repository']}/releases/download/{item['github']['release']}/{item['asset']}"
        temporary = archive.with_suffix(archive.suffix + ".partial")
        with urllib.request.urlopen(source, timeout=300) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
        if sha256(temporary) != item["archive_sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"GitHub archive hash mismatch: {item['id']}")
        temporary.replace(archive)
    root = work_dir / item["install_subdir"]
    already = root.exists() and all((root / entry["path"]).is_file() and
                                    (root / entry["path"]).stat().st_size == entry["size_bytes"] and
                                    sha256(root / entry["path"]) == entry["sha256"] for entry in item["files"])
    if not already:
        safe_extract(archive, root, item["files"])
    return root

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--asset", action="append", help="Asset id; omit for all")
    parser.add_argument("--kind", choices=["model", "dataset", "runtime"])
    parser.add_argument("--offline", action="store_true", help="Only unpack pre-transferred verified archives")
    args = parser.parse_args()
    items = read_json(ROOT / "configs/assets.json")["assets"]
    selected = [item for item in items if (not args.asset or item["id"] in args.asset) and
                (not args.kind or item["kind"] == args.kind)]
    if args.asset and set(args.asset) != {item["id"] for item in selected}:
        parser.error("Unknown or filtered asset id")
    for item in selected:
        print(item["id"], install_asset(item, args.work_dir, args.offline), flush=True)

if __name__ == "__main__":
    main()
