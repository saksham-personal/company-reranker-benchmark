#!/usr/bin/env python3
"""Generate the benchmark report from raw result JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reranker_bench.reporting import generate_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="directory containing raw result JSON files")
    parser.add_argument("--output", default="reports", help="directory for REPORT.md and CSV files")
    args = parser.parse_args()
    result_path = Path(args.results)
    output_path = Path(args.output)
    if not result_path.is_absolute():
        result_path = PROJECT_ROOT / result_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    paths = generate_report(result_path, output_path, project_root=PROJECT_ROOT)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
