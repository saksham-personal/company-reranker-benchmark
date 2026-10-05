# Offline assets v1

This release supplies six pinned PyTorch reranker archives, BEIR SciFact test data, and a locked Windows x64 Python 3.12 environment with CPU wheels. The VDI only needs GitHub access. Read [the setup guide](https://github.com/saksham-personal/company-reranker-benchmark/blob/main/docs/VDI_GUIDE.md) before running.

The repository's `configs/assets.json` records every ZIP's SHA-256, every contained file's SHA-256, exact upstream model revisions, and installation paths. `scripts/download_assets.py` and `scripts/setup_offline.ps1` verify these hashes before use. Model licenses and code provenance are included in their archives.

Archives:

- `ettin-17m-pytorch.zip`
- `ettin-32m-pytorch.zip`
- `ettin-68m-pytorch.zip`
- `ettin-150m-pytorch.zip`
- `jina-turbo-en-pytorch.zip`
- `gte-modernbert-base-pytorch.zip`
- `beir-scifact-test.zip`
- `offline-python-windows-x64.zip`

The local validation completed a 15-test suite, verified all eight archive hashes, installed the locked runtime from its ZIP without an index, and ran a short quality/speed/report smoke suite with Ettin 17M. Jina v1 Turbo passed local pair-order and batch-consistency checks with explicit float32 CPU weights. The full six-model quality and VDI timing suite is provided for execution on the VDI; no production winner is claimed from these development checks.
