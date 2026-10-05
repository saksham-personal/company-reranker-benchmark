"""Shared serialization, provenance and offline configuration."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

def offline():
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

def timestamp():
    return datetime.now(timezone.utc).isoformat()

def hardware(role="development"):
    import importlib.metadata
    import psutil
    versions = {}
    for package in ("torch", "transformers", "sentence-transformers", "numpy", "onnxruntime", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"role": role, "hostname": platform.node(), "os": platform.platform(),
            "architecture": platform.machine(), "processor": platform.processor(),
            "logical_cores": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False),
            "ram_bytes": psutil.virtual_memory().total, "available_ram_bytes": psutil.virtual_memory().available,
            "python": sys.version, "packages": versions}

def model_catalog(path=None):
    catalog = read_json(path or ROOT / "configs/models.json")
    return {m["model_id"]: m for m in catalog["models"]}
