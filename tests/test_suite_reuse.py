from __future__ import annotations

import importlib.util
import json
from pathlib import Path


module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_suite.py"
spec = importlib.util.spec_from_file_location("run_suite", module_path)
assert spec and spec.loader
run_suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_suite)


def test_only_matching_successful_result_is_reused(tmp_path):
    target = tmp_path / "result.json"
    target.write_text(json.dumps({"type": "quality", "status": "failed", "suite_run_signature": "a"}), encoding="utf-8")
    assert not run_suite._reusable(target, "a", "quality")
    target.write_text(json.dumps({"type": "quality", "status": "passed", "suite_run_signature": "a"}), encoding="utf-8")
    assert run_suite._reusable(target, "a", "quality")
    assert not run_suite._reusable(target, "b", "quality")
    assert not run_suite._reusable(target, "a", "speed")
