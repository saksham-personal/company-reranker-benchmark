"""Maintainer: fetch pinned upstream files and create GitHub-only model packages.

Never run this on the blocked/offline VDI. See download_assets.py there instead.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reranker_bench.common import ROOT, read_json, sha256, write_json

MODELS = [(f"ettin-{size}m", f"cross-encoder/ettin-reranker-{size}m-v1", "sentence_transformers", 7999,
           params, score, "apache-2.0") for size, params, score in
          [(17, 17.6, .5576), (32, 32.8, .5779), (68, 68.6, .5915), (150, 150.9, .5994)]] + [
    ("jina-turbo-en", "jinaai/jina-reranker-v1-turbo-en", "transformers", 8192, 37.8, None, "apache-2.0"),
    ("gte-modernbert-base", "Alibaba-NLP/gte-reranker-modernbert-base", "transformers", 8192, 149, .5843, "apache-2.0")]

def json_url(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "company-reranker-benchmark/1.0"}), timeout=120) as response:
        return json.load(response)

def fetch(url, path, expected=None, expected_size=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and (expected is None or sha256(path) == expected) and (expected_size is None or path.stat().st_size == expected_size):
        return
    temp = path.with_suffix(path.suffix + ".part")
    for attempt in range(8):
        offset = temp.stat().st_size if temp.exists() else 0
        if expected_size and offset > expected_size:
            temp.unlink()
            offset = 0
        headers = {"User-Agent": "company-reranker-benchmark/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with temp.open(mode) as stream:
                    shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
        except (OSError, TimeoutError):
            if attempt == 7:
                raise
        if expected_size is None or temp.stat().st_size == expected_size:
            if expected is None or sha256(temp) == expected:
                temp.replace(path)
                return
        if expected_size and temp.stat().st_size == expected_size:
            temp.unlink()
    if expected and sha256(temp) != expected:
        raise ValueError(f"Upstream checksum failed: {path.name}")
    raise ValueError(f"Upstream file truncated: {path.name}")

def files_manifest(folder):
    return [{"path": p.relative_to(folder).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(folder.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
            and p.name != "MODEL_RECEIPT.json"]

def package(folder, destination, asset_id, kind, repository="saksham-personal/company-reranker-benchmark"):
    files = files_manifest(folder)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for entry in files:
            archive.write(folder / entry["path"], entry["path"])
    if destination.stat().st_size >= 2 * 1024 ** 3:
        raise ValueError(f"Asset {asset_id} exceeds GitHub 2 GiB limit")
    return {"id": asset_id, "kind": kind, "asset": destination.name,
            "github": {"repository": repository, "release": "assets-v1"},
            "archive_sha256": sha256(destination), "size_bytes": destination.stat().st_size,
            "files": files, "install_subdir": f"{kind}s/{asset_id}"}

def prepare(work):
    work = Path(work).resolve()
    previous = read_json(ROOT / "configs/upstream_lock.json") if (ROOT / "configs/upstream_lock.json").exists() else {"models": {}}
    lock = previous
    catalog, assets = [], []
    for mid, repo, loader, limit, params, score, license_ in MODELS:
        if mid not in lock["models"]:
            info = json_url(f"https://huggingface.co/api/models/{repo}?blobs=true")
            lock["models"][mid] = {"repo_id": repo, "revision": info["sha"], "siblings": info["siblings"], "license": info.get("cardData", {}).get("license", license_)}
            write_json(ROOT / "configs/upstream_lock.json", lock)
        pinned = lock["models"][mid]
        folder = work / "models" / mid
        folder.mkdir(parents=True, exist_ok=True)
        for sibling in pinned["siblings"]:
            name = sibling["rfilename"]
            if name.startswith(("onnx/", "openvino/", ".")) or name.endswith((".bin", ".onnx")):
                continue
            if not (Path(name).suffix in (".json", ".txt", ".py", ".md", ".safetensors", ".model") or Path(name).name.upper().startswith(("LICENSE", "NOTICE"))):
                continue
            if ".." in Path(name).parts:
                raise ValueError("Unsafe upstream path")
            print("DOWNLOAD", mid, name, flush=True)
            fetch(f"https://huggingface.co/{repo}/resolve/{pinned['revision']}/{name}", folder / name,
                  sibling.get("lfs", {}).get("sha256"), sibling.get("size"))
        if not list(folder.rglob("*.safetensors")):
            raise ValueError(f"No safetensors checkpoint for {mid}")
        modifications = []
        config_path = folder / "config.json"
        if config_path.is_file():
            config = read_json(config_path)
            code_repos = {value.split("--")[0] for value in config.get("auto_map", {}).values() if isinstance(value, str) and "--" in value}
            for code_repo in sorted(code_repos):
                if code_repo not in lock.setdefault("code", {}):
                    info = json_url(f"https://huggingface.co/api/models/{code_repo}?blobs=true")
                    lock["code"][code_repo] = {"repo_id": code_repo, "revision": info["sha"], "siblings": info["siblings"]}
                    write_json(ROOT / "configs/upstream_lock.json", lock)
                source = lock["code"][code_repo]
                for sibling in source["siblings"]:
                    name = sibling["rfilename"]
                    if name.endswith(".py"):
                        fetch(f"https://huggingface.co/{code_repo}/resolve/{source['revision']}/{name}", folder / name,
                              sibling.get("lfs", {}).get("sha256"), sibling.get("size"))
            if code_repos:
                if not (folder / "config.upstream.json").exists():
                    shutil.copyfile(config_path, folder / "config.upstream.json")
                config["auto_map"] = {key: value.split("--")[-1] if isinstance(value, str) else value for key, value in config["auto_map"].items()}
                write_json(config_path, config)
                modifications.append("auto_map resolves bundled pinned custom code locally; config.upstream.json retained")
        if mid == "jina-turbo-en":
            # The pinned 2024 Jina custom config defines an optional ONNX export
            # helper. Transformers 5 removed transformers.onnx; inference never
            # calls the helper. Retain its original source for audit.
            code_path = folder / "configuration_bert.py"
            upstream_path = folder / "configuration_bert.upstream.py"
            if not upstream_path.exists():
                shutil.copyfile(code_path, upstream_path)
            code = upstream_path.read_text(encoding="utf-8")
            needle = "from transformers.onnx import OnnxConfig"
            if needle not in code:
                raise ValueError("Jina compatibility patch target changed upstream")
            replacement = "try:\n    from transformers.onnx import OnnxConfig\nexcept ModuleNotFoundError:\n    class OnnxConfig:\n        pass  # Optional legacy ONNX export class; not used for CPU inference"
            code = code.replace(needle, replacement, 1)
            init_needle = "        super().__init__(pad_token_id=pad_token_id, **kwargs)"
            if init_needle not in code:
                raise ValueError("Jina config compatibility patch target changed upstream")
            init_replacement = init_needle + "\n        self.is_decoder = kwargs.get('is_decoder', False)\n        self.add_cross_attention = kwargs.get('add_cross_attention', False)\n        self.chunk_size_feed_forward = kwargs.get('chunk_size_feed_forward', 0)"
            code_path.write_text(code.replace(init_needle, init_replacement, 1), encoding="utf-8")
            modifications.append("Restored removed Transformers 5 legacy config defaults and guarded optional ONNX import; original source retained")
            model_code_path = folder / "modeling_bert.py"
            model_upstream_path = folder / "modeling_bert.upstream.py"
            if not model_upstream_path.exists():
                shutil.copyfile(model_code_path, model_upstream_path)
            model_code = model_upstream_path.read_text(encoding="utf-8")
            needle = "    find_pruneable_heads_and_indices,\n"
            if needle not in model_code:
                raise ValueError("Jina pruning compatibility patch target changed upstream")
            model_code = model_code.replace(needle, "", 1)
            slopes_needle = "torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device) * -1"
            if slopes_needle not in model_code:
                raise ValueError("Jina ALiBi compatibility patch target changed upstream")
            model_code = model_code.replace(slopes_needle,
                                            "torch.tensor(_get_alibi_head_slopes(n_heads), device=device, dtype=torch.float32) * -1", 1)
            head_needle = "        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)"
            if head_needle not in model_code:
                raise ValueError("Jina head-mask compatibility patch target changed upstream")
            model_code = model_code.replace(head_needle,
                                            "        if head_mask is not None:\n            raise NotImplementedError('Explicit attention head masks are unsupported by the offline Jina compatibility shim')\n        head_mask = [None] * self.config.num_hidden_layers", 1)
            # This deprecated helper is reached only through optional head
            # pruning, never through evaluation or scoring. Fail explicitly
            # if a caller attempts to use that unsupported pathway.
            model_code += "\n\ndef find_pruneable_heads_and_indices(*args, **kwargs):\n    raise NotImplementedError('Head pruning is unsupported by the offline Jina compatibility shim')\n"
            model_code_path.write_text(model_code, encoding="utf-8")
            modifications.append("Removed deprecated optional head-pruning helper import for Transformers 5.7; pruning now fails explicitly; original source retained")
            modifications.append("Construct ALiBi slopes on the same device as positions during Transformers lazy initialization; original source retained")
            modifications.append("Replace removed default head-mask helper for ordinary inference; explicit head masks fail; original source retained")
        if not any(p.name.upper().startswith("LICENSE") for p in folder.iterdir()):
            if pinned["license"] == "apache-2.0":
                fetch("https://www.apache.org/licenses/LICENSE-2.0.txt", folder / "LICENSE")
            else:
                raise ValueError(f"Missing license for {mid}: {pinned['license']}")
        write_json(folder / "PROVENANCE.json", {"repo_id": repo, "revision": pinned["revision"], "license": pinned["license"], "code": lock.get("code", {}) if mid == "jina-turbo-en" else {}, "modifications": modifications})
        resource = package(folder, work / "packages" / f"{mid}-pytorch.zip", mid, "model")
        assets.append(resource)
        catalog.append({"model_id": mid, "repo_id": repo, "revision": pinned["revision"], "parameters_millions": params,
                        "pair_limit": limit, "loader": loader, "trust_remote_code": mid == "jina-turbo-en",
                        "license": pinned["license"], "default_attention": "eager" if mid == "jina-turbo-en" else "sdpa",
                        "public_mteb_ndcg10": score, "public_mteb_ndcg20": None, "public_mteb_ndcg100": None,
                        "public_score_source": "https://huggingface.co/blog/ettin-reranker" if score is not None else None,
                        "public_score_protocol": "MTEB(eng,v2) Retrieval:10 tasks,rerank top100,mean of six embedding retriever pairings" if score is not None else None,
                        "reference_artifact": mid, "validation_status": "pending_local_validation"})
        write_json(ROOT / "configs/models.json", {"schema_version": 1, "models": catalog})
        write_json(ROOT / "configs/assets.json", {"schema_version": 1, "assets": assets})
        print("PACKAGED", mid, resource["size_bytes"], flush=True)
    return assets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.work_dir)
