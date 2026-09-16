from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_per_image_metrics_use_all_observations() -> None:
    module = load("sparc_per_image_suite_test", "run_sparc_per_image_development_suite.py")
    deltas = np.asarray([
        [1, 0],
        [-1, 1],
        [0, -1],
        [1, -1],
    ], dtype=np.int8)
    decisions = np.asarray([0, 0, -1, 1], dtype=np.int64)
    result = module.per_image_metrics(decisions, deltas)
    assert result["route_pct"] == 75.0
    assert result["fix_pct"] == 25.0
    assert result["break_pct"] == 50.0
    assert result["delta_accuracy_pp"] == -25.0
    assert result["unchanged_pct"] == 25.0


def test_table3_display_policy_is_predeclared() -> None:
    module = load("sparc_final_artifacts_test", "generate_sparc_final_artifacts.py")
    rows = [
        {"experts": 4}, {"experts": 7}, {"experts": 10}, {"experts": 13},
    ]
    assert [row["experts"] for row in module.display_table3_rows(rows)] == [4, 7, 13]
