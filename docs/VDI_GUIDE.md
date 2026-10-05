# Restricted VDI guide

1. Clone the repo from GitHub and choose a work directory with roughly 8 GB free. Windows x64 and Python 3.12 are pinned in the Release environment bundle. No Hugging Face request is made on the VDI.
2. Run `powershell -ExecutionPolicy Bypass -File scripts/setup_offline.ps1 -WorkDir D:\RerankerBench`. The script verifies archive and per-file SHA-256, creates a virtual environment, and installs exact hashed wheels using `--no-index`.
3. Run `D:\RerankerBench\.venv\Scripts\python.exe scripts/download_assets.py --work-dir D:\RerankerBench --kind model` and the same command with `--asset beir-scifact`. It downloads only from this repo's GitHub Release, verifies every file and extracts into `models/` and `datasets/` beneath the work directory.
4. Verify `D:\RerankerBench\.venv\Scripts\python.exe -m reranker_bench.cli validate-model --model ettin-17m --model-root D:\RerankerBench\models` and `... validate-dataset data\controlled-company-screening`.
5. Run `D:\RerankerBench\.venv\Scripts\python.exe scripts/run_suite.py --work-dir D:\RerankerBench --dry-run` to see the planned jobs. Remove `--dry-run` for the measured run. It executes serially and can take hours for the larger models and 4,096-token diagnostics.
6. Read `reports/REPORT.md` and raw JSON. A failed model has a `.log` and failure JSON; fix the cause and use `--force` to rerun. Files are never silently counted as success.

The default suite uses 512-token FP32 pairs and a 4,096-token diagnostic. If memory is tight, run one job at a time with the CLI and batch size 1 or 2. Keep thread, batch and token settings matched across models for a comparison. Record your VDI CPU quota and avoid overlapping runs.

To substitute production stage-one candidates, export one JSONL object per query with ordered `doc_ids`, then run:

```powershell
D:\RerankerBench\.venv\Scripts\python.exe scripts/build_candidates.py data\controlled-company-screening configs\candidates\my-stage1.json --external-jsonl D:\RerankerBench\my-stage1-runs.jsonl --depth 1000
```

Use an appropriately licensed private company corpus and corresponding judgments for a real transfer evaluation. The shipped controlled corpus is synthetic and should not be used to claim PitchBook performance.

If GitHub Release downloads are filtered too, transfer the ZIPs via an approved channel. Put model and SciFact archives under `D:\RerankerBench\downloads`, pass the runtime ZIP with `setup_offline.ps1 -ArchivePath`, and use `download_assets.py --offline`. Hash verification still applies.
