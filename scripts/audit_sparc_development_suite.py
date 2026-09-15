#!/usr/bin/env python3
"""Independent lineage and metric audit for the SPARC development suite."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from streammint.utility_router import connected_component_groups, seeded_group_folds  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "results/sparc_development_experiment_suite_v1"
DEFAULT_DATASET = (
    ROOT / "results/sparc_deployment_brightness_router_v1/results/"
    "utility_targets_dev15_with_deployment_brightness.npz"
)
SEED = 1109


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def realized_metrics(decisions: np.ndarray, utilities: np.ndarray) -> dict[str, float]:
    rows = np.arange(len(decisions))
    routed = decisions >= 0
    realized = np.zeros(len(decisions), dtype=np.float64)
    realized[routed] = utilities[rows[routed], decisions[routed]]
    return {
        "mean_utility_pp": 100.0 * float(realized.mean()),
        "coverage_pct": 100.0 * float(routed.mean()),
        "harm_pct": 100.0 * float((realized < 0).mean()),
        "benefit_pct": 100.0 * float((realized > 0).mean()),
    }


def assert_metrics(actual: dict, expected: dict, label: str) -> None:
    for key in ("mean_utility_pp", "coverage_pct", "harm_pct", "benefit_pct"):
        if not close(actual[key], expected[key]):
            raise RuntimeError(f"{label}: {key} mismatch {actual[key]} != {expected[key]}")


def audit_csv(path: Path, expected: list[dict], key: str) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if len(rows) != len(expected):
        raise RuntimeError(f"{path.name}: row count mismatch")
    for actual, reference in zip(rows, expected):
        if actual[key] != str(reference[key]):
            raise RuntimeError(f"{path.name}: row identity mismatch")
        for field, value in reference.items():
            if field not in actual or value is None:
                continue
            if isinstance(value, bool):
                if actual[field] != str(value):
                    raise RuntimeError(f"{path.name}: boolean mismatch for {field}")
            elif isinstance(value, (int, float)):
                if not close(float(actual[field]), float(value)):
                    raise RuntimeError(f"{path.name}: numeric mismatch for {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--router-report", type=Path,
        help="Portable override for the frozen router report referenced by report.json.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suite, dataset = args.suite.resolve(), args.dataset.resolve()
    output = args.output.resolve() if args.output else suite / "independent_audit.json"
    report_path, manifest_path = suite / "report.json", suite / "manifest.json"
    report = json.loads(report_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if report.get("status") != "completed_development_only_grouped_experiment_suite":
        raise RuntimeError("unexpected suite status")
    if report.get("seed") != SEED or report.get("dataset_sha256") != sha256(dataset):
        raise RuntimeError("seed or dataset lineage mismatch")
    if "untouched2k" in str(report.get("dataset", "")).lower():
        raise RuntimeError("development suite points to an untouched-test dataset")
    if manifest.get("files", {}).get("report.json", {}).get("sha256") != sha256(report_path):
        raise RuntimeError("suite manifest does not bind report.json")
    for name, metadata in manifest.get("files", {}).items():
        path = suite / name
        if not path.is_file() or sha256(path) != metadata.get("sha256"):
            raise RuntimeError(f"manifest file mismatch: {name}")

    with np.load(dataset, allow_pickle=False) as stored:
        utilities = stored["utilities"].astype(np.float64)
        observed = stored["observed"].astype(bool)
        groups = stored["groups"].astype(str)
        batch_ids = stored["batch_ids"].astype(str)
        scenarios = stored["scenarios"].astype(str)
        experts = tuple(str(value) for value in stored["expert_names"])
        seed = int(stored["seed"].item())
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    if seed != SEED or utilities.shape != (15000, 13) or not observed.all():
        raise RuntimeError("development matrix contract failed")
    if len(unique_groups) != 1000 or set(group_counts.tolist()) != {15}:
        raise RuntimeError("base-image grouping contract failed")
    if len(np.unique(scenarios)) != 15 or len(experts) != 13:
        raise RuntimeError("scenario or expert count failed")
    split_groups = connected_component_groups(groups, batch_ids)
    components, component_counts = np.unique(split_groups, return_counts=True)
    if len(components) != 50 or set(component_counts.tolist()) != {300}:
        raise RuntimeError("joint image/batch component contract failed")
    expected_fold_hashes = []
    for train, validation in seeded_group_folds(split_groups, n_splits=5, seed=SEED):
        if set(groups[train]) & set(groups[validation]):
            raise RuntimeError("base-image leakage across joint folds")
        if set(batch_ids[train]) & set(batch_ids[validation]):
            raise RuntimeError("batch leakage across joint folds")
        expected_fold_hashes.append(hashlib.sha256(
            "\n".join(sorted(str(v) for v in np.unique(split_groups[validation]).tolist()))
            .encode("utf-8")
        ).hexdigest())
    if report.get("fold_grouping") != "connected components over base-image IDs and batch IDs":
        raise RuntimeError("report does not declare joint image/batch grouping")

    controls = report["same_information_controls"]
    by_name = {row["method"]: row for row in controls}
    identity = realized_metrics(np.full(len(utilities), -1, dtype=np.int64), utilities)
    assert_metrics(identity, by_name["Identity / pass-through"], "identity")
    fixed_decisions = np.full(len(utilities), -1, dtype=np.int64)
    for train, validation in seeded_group_folds(split_groups, n_splits=5, seed=SEED):
        best = int(np.argmax(utilities[train, :4].mean(axis=0)))
        fixed_decisions[validation] = best
    single = realized_metrics(fixed_decisions, utilities)
    assert_metrics(single, by_name["Best single fixed expert"], "best single fixed")
    oracle_choice = np.argmax(utilities, axis=1)
    oracle_value = utilities[np.arange(len(utilities)), oracle_choice]
    oracle = realized_metrics(np.where(oracle_value > 0, oracle_choice, -1), utilities)
    assert_metrics(oracle, by_name["Per-sample utility oracle"], "oracle")

    router_report_path = (args.router_report.resolve() if args.router_report
                          else Path(report["router_report"]))
    if sha256(router_report_path) != report["router_report_sha256"]:
        raise RuntimeError("frozen router report hash mismatch")
    for section in ("best_single_fixed", "corruption_classifier", "ridge18_folds",
                    "ridge55_folds", "hgb18_folds", "full_nonlinear_folds"):
        folds = report["audits"][section]
        if len(folds) != 5:
            raise RuntimeError(f"grouped fold audit failed: {section}")
        actual_hashes = [row["validation_group_sha256"] for row in folds]
        if actual_hashes != expected_fold_hashes:
            raise RuntimeError(f"joint-fold hash mismatch: {section}")
        if section != "best_single_fixed" and any(
            int(row.get("group_overlap", 0)) != 0 for row in folds
        ):
            raise RuntimeError(f"group overlap reported: {section}")
    for section in ("ridge18_gate", "ridge55_gate", "hgb18_gate",
                    "threshold_only_gate", "complete_gate"):
        folds = report["audits"][section]
        if len(folds) != 5 or sum(int(row["evaluation_rows"]) for row in folds) != 15000:
            raise RuntimeError(f"gate cross-fit audit failed: {section}")
        if [row["evaluation_group_sha256"] for row in folds] != expected_fold_hashes:
            raise RuntimeError(f"gate joint-fold hash mismatch: {section}")

    rng = np.random.default_rng(SEED)
    ordered = components.copy()
    rng.shuffle(ordered)
    previous = set()
    for size in (100, 240, 500, 1000):
        chosen_components = set(ordered[:size // 20].tolist())
        selected = np.asarray([value in chosen_components for value in split_groups])
        chosen = set(groups[selected].tolist())
        if previous and not previous.issubset(chosen):
            raise RuntimeError("data-efficiency groups are not nested")
        digest = hashlib.sha256("\n".join(sorted(chosen)).encode("utf-8")).hexdigest()
        stored_digest = report["audits"]["data_efficiency"][str(size)]["selected_group_sha256"]
        if digest != stored_digest:
            raise RuntimeError(f"data-efficiency group hash mismatch at {size}")
        component_digest = hashlib.sha256(
            "\n".join(str(v) for v in sorted(chosen_components)).encode("utf-8")
        ).hexdigest()
        if component_digest != report["audits"]["data_efficiency"][str(size)][
            "selected_component_sha256"
        ]:
            raise RuntimeError(f"data-efficiency component hash mismatch at {size}")
        previous = chosen

    audit_csv(suite / "table2_router_controls.csv", controls, "method")
    audit_csv(suite / "table3_expert_bank_ablation.csv",
              report["expert_bank_ablation"], "ablation")
    audit_csv(suite / "table3_gate_ablation.csv", report["gate_ablation"], "ablation")
    audit_csv(suite / "data_efficiency.csv", report["data_efficiency"],
              "available_base_images")

    audit = {
        "schema_version": 1,
        "status": "passed_independent_lineage_metric_and_grouping_audit",
        "seed": SEED,
        "dataset_sha256": sha256(dataset),
        "suite_report_sha256": sha256(report_path),
        "rows": 15000,
        "base_image_groups": 1000,
        "scenarios": 15,
        "experts": 13,
        "group_overlap": 0,
        "batch_overlap": 0,
        "joint_components": 50,
        "nested_data_efficiency_groups": True,
        "untouched_test_results_read": False,
        "recomputed_controls": ["identity", "best single fixed", "per-sample oracle"],
        "verified_joint_fold_sections": 11,
        "verified_csv_views": 4,
        "limitation": (
            "This audit verifies lineage, grouping, deterministic source linkage, simple-control "
            "metrics and report/CSV consistency; it does not claim an independent reimplementation "
            "of every fitted sklearn model."
        ),
    }
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
