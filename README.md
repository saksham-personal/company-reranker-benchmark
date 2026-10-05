# Company reranker benchmark

Offline, CPU-only evaluation of six cross-encoder rerankers for company screening. The repository contains the benchmark code, frozen first-stage candidate lists, a 130-company public pilot, a 2,400-company controlled set, seven long-context diagnostics, and provenance for the separately packaged SciFact test set. Model weights, SciFact, and a locked Windows Python/wheel bundle are checksum-pinned [GitHub Release assets](https://github.com/saksham-personal/company-reranker-benchmark/releases/tag/assets-v1); the VDI does not need Hugging Face access.

| Model | Parameters | Published MTEB nDCG@10* | Pair limit |
| --- | ---: | ---: | ---: |
| Ettin 17M | 17.6M | 0.5576 | 7,999 |
| Ettin 32M | 32.8M | 0.5779 | 7,999 |
| Jina v1 Turbo EN | 37.8M | not in this comparison | 8,192 |
| Ettin 68M | 68.6M | 0.5915 | 7,999 |
| GTE ModernBERT base | 149M | 0.5843 | 8,192 |
| Ettin 150M | 150.9M | 0.5994 | 7,999 |

*The published scores come from the [Ettin reranker comparison](https://huggingface.co/blog/ettin-reranker): MTEB English v2 retrieval, 10 tasks, reranking top 100 from six first-stage systems. They are external reference scores, not results from this kit. Comparable public nDCG@20 and nDCG@100 were not published in that comparison; this kit calculates them from its own runs. Jina's separate published results use another protocol and are deliberately not inserted into this column.

## Run on the restricted Windows VDI

Clone this GitHub repo, choose a work directory with at least 8 GB free, and run in PowerShell:

```powershell
git clone https://github.com/saksham-personal/company-reranker-benchmark.git
cd company-reranker-benchmark
powershell -ExecutionPolicy Bypass -File scripts/setup_offline.ps1 -WorkDir D:\RerankerBench
D:\RerankerBench\.venv\Scripts\python.exe scripts/download_assets.py --work-dir D:\RerankerBench --kind model
D:\RerankerBench\.venv\Scripts\python.exe scripts/download_assets.py --work-dir D:\RerankerBench --asset beir-scifact
D:\RerankerBench\.venv\Scripts\python.exe scripts/run_suite.py --work-dir D:\RerankerBench
```

`setup_offline.ps1` downloads only from GitHub, verifies the archive and every bundled file, then installs the exact hashed Windows wheels without an index. All model loading sets offline flags and uses local paths. If GitHub Release assets must be transferred manually, copy the ZIPs into `D:\RerankerBench\downloads` and run `download_assets.py --offline`; pass the runtime ZIP to `setup_offline.ps1 -ArchivePath`.

The suite runs jobs serially in fresh CPU processes. It stores raw JSON and logs in `results/raw/` and produces `reports/REPORT.md` plus CSVs. Re-run with `--force` only when you intend to replace prior measurements. Start with [the VDI guide](docs/VDI_GUIDE.md) for setup checks and staged runs.

For a quick installation check before the full suite, run `D:\RerankerBench\.venv\Scripts\python.exe scripts/run_suite.py --plan configs\suite-smoke.json --work-dir D:\RerankerBench --results D:\RerankerBench\smoke-results`. This short synthetic result is exploratory.

## Evaluation design

For each query, stage one retrieves up to 1,000 companies. Each model scores `(screening criteria, one company record)` pairs for the **original top 500**, sorts that prefix, and leaves ranks 501 onward untouched. The reranker cannot recover a relevant company missing from stage one's top 500. All models receive the same frozen candidate IDs and chosen text representation. These BM25 pools are a reproducible fallback, not a claim that BM25 matches your current embedding stage. Replace them with exported ordered runs from the actual embedding system for production transfer.

The report compares baseline, reranked and candidate-oracle nDCG@10/20/100, known-positive recall, category slices, documented exclusions, unknown cases, token truncation, throughput, query latency and sampled memory. See [methodology](docs/METHODOLOGY.md) and [data cards](docs/DATASETS.md). A reranker score is a ranking signal. A company must still pass source-backed inclusion and exclusion checks; missing evidence remains unknown.

This kit does **not** select a production winner. The public pilot has incomplete judgments and only 130 companies; controlled and long-context data are synthetic; SciFact measures scientific claims. Add a representative, licensed company set with independent judgments, export your actual top-1,000 retrieval runs, and repeat matched VDI runs before deployment. Keep private company descriptions out of this public repository.

## Repo layout

- `configs/models.json`, `upstream_lock.json`, `assets.json`: pinned models, revisions, hashes, licenses, Release locations.
- `configs/candidates/`: frozen stage-one candidate lists with dataset digests and retrieval provenance.
- `data/`: public company pilot, controlled synthetic set, long-context diagnostics, and their original source bundles.
- `src/reranker_bench/`: local model adapters, validation, retrieval, metrics, benchmarking and reporting.
- `scripts/`: VDI setup, asset verification, data/candidate preparation, suite execution and reporting.
- `tests/`: metric, report and pipeline checks.

Models are distributed under their upstream Apache-2.0 licenses with attribution and unmodified original code files where the pinned Jina loader needed compatibility shims. See [third-party notices](THIRD_PARTY.md).
