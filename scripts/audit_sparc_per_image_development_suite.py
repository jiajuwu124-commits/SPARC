#!/usr/bin/env python3
"""Independently audit the final SPARC per-image development suite."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def audit_metric_rows(values: list[dict], label: str) -> None:
    if not values:
        raise RuntimeError(f"{label} is empty")
    for index, row in enumerate(values):
        delta = float(row["delta_accuracy_pp"])
        fix = float(row["fix_pct"])
        break_ = float(row["break_pct"])
        unchanged = float(row["unchanged_pct"])
        if abs(delta - (fix - break_)) > 1e-10:
            raise RuntimeError(f"{label}[{index}] violates DeltaAcc = Fix - Break")
        if abs(fix + break_ + unchanged - 100.0) > 1e-10:
            raise RuntimeError(f"{label}[{index}] transitions do not sum to 100")


def compare_csv(path: Path, expected: list[dict]) -> None:
    observed = rows(path)
    if len(observed) != len(expected):
        raise RuntimeError(f"row count mismatch: {path}")
    for row_index, (left, right) in enumerate(zip(observed, expected)):
        for key, value in right.items():
            if key not in left:
                raise RuntimeError(f"{path.name} missing {key}")
            if isinstance(value, (int, float)):
                if abs(float(left[key]) - float(value)) > 1e-12:
                    raise RuntimeError(f"{path.name}[{row_index}] differs at {key}")
            elif left[key] != str(value):
                raise RuntimeError(f"{path.name}[{row_index}] differs at {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    dataset = args.dataset.resolve()
    transitions = args.transitions.resolve()
    report = json.loads((suite / "report.json").read_text())
    manifest = json.loads((suite / "manifest.json").read_text())
    if report.get("status") != "completed_final_per_image_development_suite":
        raise RuntimeError("unexpected suite status")
    if report["dataset_sha256"] != sha256(dataset):
        raise RuntimeError("dataset hash mismatch")
    if report["transitions_sha256"] != sha256(transitions):
        raise RuntimeError("transition hash mismatch")
    with np.load(dataset, allow_pickle=False) as data, np.load(transitions, allow_pickle=False) as trans:
        if not np.array_equal(data["samples"], trans["samples"]):
            raise RuntimeError("sample alignment mismatch")
        if not np.array_equal(data["raw_sha256"], trans["raw_sha256"]):
            raise RuntimeError("raw hash alignment mismatch")
        deltas = trans["deltas"].astype(np.int8)
        utilities = data["utilities"].astype(np.float64)
        batch_ids = data["batch_ids"].astype(str)
        maximum_error = 0.0
        for batch_id in np.unique(batch_ids):
            selected = np.flatnonzero(batch_ids == batch_id)
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(deltas[selected].mean(axis=0) - utilities[selected[0]]))),
            )
    if maximum_error > 1e-12:
        raise RuntimeError("per-image evidence does not reconstruct batch targets")
    for fold in report["fold_isolation_audit"]:
        for key in ("base_image_overlap", "batch_overlap", "component_overlap"):
            if int(fold[key]) != 0:
                raise RuntimeError(f"fold leakage at {fold['fold']}: {key}")
    for audit in report["model_and_gate_audits"].values():
        if isinstance(audit, list):
            continue
        for outer in audit["nested_outer_folds"]:
            if outer["group_overlap"] != 0:
                raise RuntimeError("outer gate/model leakage")
            if any(inner["group_overlap"] != 0 for inner in outer["inner_folds"]):
                raise RuntimeError("inner gate-calibration leakage")

    table2 = report["table2_router_controls"]
    table3_complete = report["table3_nested_expert_bank_complete"]
    table3_display = report["table3_display_rows"]
    efficiency = report["figure3_data_efficiency"]
    for name, values in (
        ("table2", table2), ("table3_complete", table3_complete),
        ("table3_display", table3_display), ("efficiency", efficiency),
    ):
        audit_metric_rows(values, name)
    compare_csv(suite / "table2_router_controls.csv", table2)
    compare_csv(suite / "table3_nested_expert_bank_complete.csv", table3_complete)
    compare_csv(suite / "table3_ablations.csv", table3_display)
    compare_csv(suite / "data_efficiency.csv", efficiency)
    if [row["experts"] for row in table3_complete] != [4, 7, 10, 13]:
        raise RuntimeError("unexpected semantic bank sequence")
    if [row["experts"] for row in table3_display] != [4, 7, 13]:
        raise RuntimeError("unexpected paper display sequence")
    full_control = next(row for row in table2 if row["router"] == "SPARC + batch")
    full_bank = table3_complete[-1]
    full_efficiency = efficiency[-1]
    for key in ("delta_accuracy_pp", "fix_pct", "break_pct", "route_pct"):
        if not (
            abs(float(full_control[key]) - float(full_bank[key])) < 1e-12
            and abs(float(full_control[key]) - float(full_efficiency[key])) < 1e-12
        ):
            raise RuntimeError(f"full-data rows disagree at {key}")
    for name, metadata in manifest["outputs"].items():
        path = suite / name
        if not path.is_file() or sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"manifest mismatch: {name}")

    result = {
        "schema_version": 1,
        "status": "passed_independent_final_per_image_suite_audit",
        "rows": report["rows"],
        "outer_folds": len(report["fold_isolation_audit"]),
        "base_image_overlap": 0,
        "batch_overlap": 0,
        "component_overlap": 0,
        "metric_identity_passed": True,
        "batch_target_reconstruction_error": maximum_error,
        "full_row_consistency_passed": True,
    }
    target = suite / "independent_audit.json"
    temporary = target.with_suffix(target.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
