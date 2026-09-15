#!/usr/bin/env python3
"""Run the post-main SPARC mechanism experiments on development evidence only.

This suite never reads the untouched 2K test predictions.  It uses one audited
15K x 13 development utility matrix, seed 1109, image-and-batch-disjoint folds,
and the same expert bank for all directly compared routers.  Outputs are intended
for Table 2, ablations, and a development-data learning curve.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT / "results/sparc_deployment_brightness_router_v1/results/"
    "utility_targets_dev15_with_deployment_brightness.npz"
)
DEFAULT_ROUTER_REPORT = (
    ROOT / "results/sparc_deployment_brightness_router_v1/results/"
    "nonlinear_router_with_deployment_brightness/report.json"
)
DEFAULT_OUTPUT = ROOT / "results/sparc_development_experiment_suite_v1"
SEED = 1109

import sys
sys.path.insert(0, str(ROOT / "src"))
from streammint.nonlinear_utility_router import NonlinearUtilityRouter  # noqa: E402
from streammint.utility_router import (  # noqa: E402
    UtilityRouter,
    augment_batch_statistics,
    connected_component_groups,
    seeded_group_folds,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def group_digest(values: np.ndarray) -> str:
    unique = sorted(str(value) for value in np.unique(values).tolist())
    return hashlib.sha256("\n".join(unique).encode("utf-8")).hexdigest()


def metrics(decisions: np.ndarray, utilities: np.ndarray) -> dict[str, float]:
    rows = np.arange(len(decisions))
    routed = decisions >= 0
    realized = np.zeros(len(decisions), dtype=np.float64)
    realized[routed] = utilities[rows[routed], decisions[routed]]
    return {
        "mean_utility_pp": 100.0 * float(realized.mean()),
        "coverage_pct": 100.0 * float(routed.mean()),
        "harm_pct": 100.0 * float((realized < 0).mean()),
        "benefit_pct": 100.0 * float((realized > 0).mean()),
        "mean_utility_when_routed_pp": (
            100.0 * float(realized[routed].mean()) if routed.any() else 0.0
        ),
    }


def decisions(predicted: np.ndarray, threshold: float, margin: float) -> np.ndarray:
    best = np.argmax(predicted, axis=1)
    best_value = predicted[np.arange(len(predicted)), best]
    second = np.partition(predicted, -2, axis=1)[:, -2]
    accepted = (best_value > threshold) & (
        best_value - np.maximum(second, 0.0) >= margin
    )
    return np.where(accepted, best, -1)


def calibrate(predicted: np.ndarray, utilities: np.ndarray,
              margins: Iterable[float]) -> tuple[float, float]:
    finite = predicted[np.isfinite(predicted)]
    thresholds = np.unique(np.maximum(
        0.0, np.concatenate(([0.0], np.quantile(finite, np.linspace(0, 1, 41))))
    ))
    candidates = []
    for threshold in thresholds:
        for margin in margins:
            current = metrics(decisions(predicted, float(threshold), float(margin)), utilities)
            if current["harm_pct"] <= 5.0 and current["coverage_pct"] >= 5.0:
                key = (current["mean_utility_pp"], -current["harm_pct"],
                       current["coverage_pct"], -float(threshold), -float(margin))
                candidates.append((key, float(threshold), float(margin)))
    if not candidates:
        return float(np.max(finite)), 0.0
    return max(candidates, key=lambda item: item[0])[1:]


def nonlinear_oof(x: np.ndarray, utilities: np.ndarray, observed: np.ndarray,
                  groups: np.ndarray, expert_names: tuple[str, ...]) -> tuple[np.ndarray, list]:
    output = np.full_like(utilities, np.nan, dtype=np.float64)
    audits = []
    for fold, (train, validation) in enumerate(
        seeded_group_folds(groups, n_splits=5, seed=SEED)
    ):
        overlap = set(groups[train].tolist()) & set(groups[validation].tolist())
        if overlap:
            raise RuntimeError(f"group leakage in fold {fold}")
        router = NonlinearUtilityRouter.fit(
            x[train], utilities[train], observed[train], expert_names,
            feature_mode="mean_std_count", seed=SEED,
        )
        output[validation] = router.predict_utility(x[validation])
        audits.append({"fold": fold, "train_groups": int(len(np.unique(groups[train]))),
                       "validation_groups": int(len(np.unique(groups[validation]))),
                       "group_overlap": 0,
                       "validation_group_sha256": group_digest(groups[validation])})
    if not np.isfinite(output).all():
        raise RuntimeError("incomplete nonlinear OOF predictions")
    return output, audits


def linear_oof(x: np.ndarray, utilities: np.ndarray, observed: np.ndarray,
               groups: np.ndarray, expert_names: tuple[str, ...]) -> tuple[np.ndarray, list]:
    output = np.full_like(utilities, np.nan, dtype=np.float64)
    audits = []
    for fold, (train, validation) in enumerate(
        seeded_group_folds(groups, n_splits=5, seed=SEED)
    ):
        if set(groups[train].tolist()) & set(groups[validation].tolist()):
            raise RuntimeError(f"component leakage in linear fold {fold}")
        router = UtilityRouter.fit(
            x[train], utilities[train], observed[train], expert_names,
            feature_mode="none", ridge_alpha=1.0,
        )
        output[validation] = router.predict_utility(x[validation])
        audits.append({"fold": fold,
                       "train_groups": int(len(np.unique(groups[train]))),
                       "validation_groups": int(len(np.unique(groups[validation]))),
                       "group_overlap": 0,
                       "validation_group_sha256": group_digest(groups[validation])})
    if not np.isfinite(output).all():
        raise RuntimeError("incomplete linear OOF predictions")
    return output, audits


def cross_fitted_gate(oof: np.ndarray, utilities: np.ndarray, groups: np.ndarray,
                      margins: tuple[float, ...]) -> tuple[np.ndarray, list]:
    output = np.full(len(oof), -1, dtype=np.int64)
    audit = []
    all_rows = np.arange(len(oof))
    for fold, (_train, validation) in enumerate(
        seeded_group_folds(groups, n_splits=5, seed=SEED)
    ):
        calibration = np.setdiff1d(all_rows, validation, assume_unique=True)
        threshold, margin = calibrate(oof[calibration], utilities[calibration], margins)
        output[validation] = decisions(oof[validation], threshold, margin)
        audit.append({"fold": fold, "threshold": threshold, "margin": margin,
                      "calibration_rows": int(len(calibration)),
                      "evaluation_rows": int(len(validation)),
                      "evaluation_group_sha256": group_digest(groups[validation])})
    return output, audit


def classifier_then_map(x: np.ndarray, utilities: np.ndarray, groups: np.ndarray,
                        scenarios: np.ndarray) -> tuple[dict, list]:
    labels = tuple(dict.fromkeys(str(value) for value in scenarios))
    label_to_index = {name: index for index, name in enumerate(labels)}
    y = np.asarray([label_to_index[str(value)] for value in scenarios], dtype=np.int64)
    routed = np.full(len(x), -1, dtype=np.int64)
    predicted_scenario = np.full(len(x), -1, dtype=np.int64)
    audits = []
    for fold, (train, validation) in enumerate(
        seeded_group_folds(groups, n_splits=5, seed=SEED)
    ):
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=False, random_state=SEED,
        )
        model.fit(x[train], y[train])
        predicted_scenario[validation] = model.predict(x[validation])
        mapping = np.full(len(labels), -1, dtype=np.int64)
        for label in range(len(labels)):
            rows = train[y[train] == label]
            means = utilities[rows].mean(axis=0)
            best = int(np.argmax(means))
            if means[best] > 0:
                mapping[label] = best
        routed[validation] = mapping[predicted_scenario[validation]]
        audits.append({"fold": fold, "train_groups": int(len(np.unique(groups[train]))),
                       "validation_groups": int(len(np.unique(groups[validation]))),
                       "group_overlap": 0,
                       "validation_group_sha256": group_digest(groups[validation]),
                       "active_mapped_experts": int(len(set(mapping[mapping >= 0].tolist())))})
    report = metrics(routed, utilities)
    report["scenario_accuracy_pct"] = 100.0 * float((predicted_scenario == y).mean())
    report["active_experts"] = int(len(set(routed[routed >= 0].tolist())))
    return report, audits


def best_single_fixed(utilities: np.ndarray, groups: np.ndarray,
                      fixed_indices: np.ndarray) -> tuple[dict, list]:
    routed = np.full(len(utilities), -1, dtype=np.int64)
    audit = []
    for fold, (train, validation) in enumerate(
        seeded_group_folds(groups, n_splits=5, seed=SEED)
    ):
        local = int(np.argmax(utilities[train][:, fixed_indices].mean(axis=0)))
        chosen = int(fixed_indices[local])
        routed[validation] = chosen
        audit.append({"fold": fold, "expert_index": chosen,
                      "validation_group_sha256": group_digest(groups[validation])})
    report = metrics(routed, utilities)
    report["active_experts"] = int(len(set(routed.tolist())))
    return report, audit


def report_from_existing(payload: dict, mode: str) -> dict:
    value = payload["modes"][mode]["grouped_cross_fitted_gate"]
    distribution = payload["modes"][mode]["selection_distribution"]["routed"]
    return {
        "mean_utility_pp": 100.0 * value["mean_utility"],
        "coverage_pct": 100.0 * value["coverage"],
        "harm_pct": 100.0 * value["harm_rate"],
        "benefit_pct": 100.0 * value["benefit_rate"],
        "mean_utility_when_routed_pp": 100.0 * value["mean_utility_when_routed"],
        "active_experts": sum(int(count) > 0 for count in distribution.values()),
    }


def csv_text(rows: list[dict]) -> str:
    from io import StringIO
    buffer = StringIO(newline="")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--router-report", type=Path, default=DEFAULT_ROUTER_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset, router_report = args.dataset.resolve(), args.router_report.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment suite: {output}")

    with np.load(dataset, allow_pickle=False) as stored:
        raw = stored["features"].astype(np.float64)
        utilities = stored["utilities"].astype(np.float64)
        observed = stored["observed"].astype(bool)
        groups = stored["groups"].astype(str)
        batch_ids = stored["batch_ids"].astype(str)
        scenarios = stored["scenarios"].astype(str)
        expert_names = tuple(str(value) for value in stored["expert_names"])
        seed = int(stored["seed"].item())
    if seed != SEED or raw.shape != (15000, 18) or utilities.shape != (15000, 13):
        raise RuntimeError("unexpected development dataset contract")
    if not observed.all() or not np.isfinite(raw).all() or not np.isfinite(utilities).all():
        raise RuntimeError("incomplete or non-finite development evidence")
    unique_groups, counts = np.unique(groups, return_counts=True)
    if len(unique_groups) != 1000 or set(counts.tolist()) != {15}:
        raise RuntimeError("each of 1,000 base-image groups must occur in all 15 scenarios")
    expected = (
        "gaussian15", "median3", "unsharp2", "autocontrast", "inverse_zoom10",
        "learned_defocus_r1", "learned_glass_r075", "learned_motion_r1",
        "learned_snow_r1", "learned_frost_r075", "learned_fog_r1",
        "inverse_brightness_c055", "deployment_brightness",
    )
    if expert_names != expected:
        raise RuntimeError(f"unexpected expert order: {expert_names}")
    split_groups = connected_component_groups(groups, batch_ids)
    component_ids, component_row_counts = np.unique(split_groups, return_counts=True)
    if len(component_ids) != 50 or set(component_row_counts.tolist()) != {300}:
        raise RuntimeError("expected 50 image/batch components with 300 rows each")
    x55 = augment_batch_statistics(raw, batch_ids, mode="mean_std_count")
    existing = json.loads(router_report.read_text())
    if existing.get("dataset_sha256") != sha256(dataset) or existing.get("seed") != SEED:
        raise RuntimeError("existing frozen router report is not bound to this dataset")

    fixed = np.asarray([0, 1, 2, 3], dtype=np.int64)
    fixed_analytic = np.asarray([0, 1, 2, 3, 4, 11], dtype=np.int64)
    all_experts = np.arange(13, dtype=np.int64)
    no_brightness = np.arange(12, dtype=np.int64)

    identity = metrics(np.full(len(raw), -1, dtype=np.int64), utilities)
    identity["active_experts"] = 0
    fixed_report, fixed_audit = best_single_fixed(utilities, split_groups, fixed)
    class_report, class_audit = classifier_then_map(
        x55, utilities, split_groups, scenarios
    )
    ridge18_oof, ridge18_audit = linear_oof(
        raw, utilities, observed, split_groups, expert_names
    )
    ridge18_decisions, ridge18_gate_audit = cross_fitted_gate(
        ridge18_oof, utilities, split_groups, (0.0, 0.01, 0.02, 0.05, 0.10)
    )
    ridge18_report = {**metrics(ridge18_decisions, utilities),
                      "active_experts": int(len(np.unique(
                          ridge18_decisions[ridge18_decisions >= 0])))}
    ridge55_oof, ridge55_audit = linear_oof(
        x55, utilities, observed, split_groups, expert_names
    )
    ridge55_decisions, ridge55_gate_audit = cross_fitted_gate(
        ridge55_oof, utilities, split_groups, (0.0, 0.01, 0.02, 0.05, 0.10)
    )
    ridge55_report = {**metrics(ridge55_decisions, utilities),
                      "active_experts": int(len(np.unique(
                          ridge55_decisions[ridge55_decisions >= 0])))}
    hgb18_oof, hgb18_audit = nonlinear_oof(
        raw, utilities, observed, split_groups, expert_names
    )
    hgb18_decisions, hgb18_gate_audit = cross_fitted_gate(
        hgb18_oof, utilities, split_groups, (0.0, 0.01, 0.02, 0.05, 0.10)
    )
    hgb18_report = {**metrics(hgb18_decisions, utilities),
                    "active_experts": int(len(np.unique(
                        hgb18_decisions[hgb18_decisions >= 0])))}
    full_oof, full_fold_audit = nonlinear_oof(
        x55, utilities, observed, split_groups, expert_names
    )
    complete_decisions, complete_gate_audit = cross_fitted_gate(
        full_oof, utilities, split_groups, (0.0, 0.01, 0.02, 0.05, 0.10)
    )
    hgb55_report = {**metrics(complete_decisions, utilities),
                    "active_experts": int(len(np.unique(
                        complete_decisions[complete_decisions >= 0])))}
    oracle_choice = np.argmax(utilities, axis=1)
    oracle_value = utilities[np.arange(len(utilities)), oracle_choice]
    oracle_decision = np.where(oracle_value > 0, oracle_choice, -1)
    oracle_report = metrics(oracle_decision, utilities)
    oracle_report["active_experts"] = int(len(np.unique(oracle_decision[oracle_decision >= 0])))

    controls = []
    for name, model, dimensions, deployable, report in (
        ("Identity / pass-through", "None", 0, True, identity),
        ("Best single fixed expert", "Grouped selection", 0, True, fixed_report),
        ("Corruption classifier then map", "HGB classifier", 55, True, class_report),
        ("Ridge direct utility", "Linear", 18, True, ridge18_report),
        ("Ridge direct utility + batch", "Linear", 55, True, ridge55_report),
        ("HGB direct utility", "Nonlinear", 18, True, hgb18_report),
        ("SPARC direct utility + batch", "Nonlinear", 55, True, hgb55_report),
        ("Per-sample utility oracle", "Diagnostic ceiling", 0, False, oracle_report),
    ):
        controls.append({"method": name, "model": model, "dimensions": dimensions,
                         "deployable": deployable, **report})

    bank_rows = []
    for name, indices in (
        ("Fixed only", fixed),
        ("Fixed + analytic", fixed_analytic),
        ("Full bank without learned brightness", no_brightness),
    ):
        oof, audit = nonlinear_oof(
            x55, utilities[:, indices], observed[:, indices], split_groups,
            tuple(expert_names[index] for index in indices),
        )
        routed, gate_audit = cross_fitted_gate(
            oof, utilities[:, indices], split_groups, (0.0, 0.01, 0.02, 0.05, 0.10)
        )
        row = {"ablation": name, "experts": int(len(indices)),
               **metrics(routed, utilities[:, indices])}
        bank_rows.append(row)
    bank_rows.append({"ablation": "Full 13-expert bank", "experts": 13, **hgb55_report})

    best = np.argmax(full_oof, axis=1)
    gate_rows = [
        {"ablation": "No abstention", **metrics(best, utilities)},
        {"ablation": "Predicted utility > 0", **metrics(
            decisions(full_oof, 0.0, 0.0), utilities
        )},
    ]
    threshold_decisions, threshold_audit = cross_fitted_gate(
        full_oof, utilities, split_groups, (0.0,)
    )
    gate_rows.extend([
        {"ablation": "Calibrated utility threshold", **metrics(threshold_decisions, utilities)},
        {"ablation": "Threshold + runner-up margin", **metrics(complete_decisions, utilities)},
    ])

    # Nested available-development-size curve.  Each point is grouped OOF on
    # exactly the declared nested group subset; it is mechanism evidence, not
    # an untouched-test accuracy claim.  Batch context is computed once from
    # the full deployment-like batches before selecting training subsets.
    rng = np.random.default_rng(SEED)
    ordered_components = component_ids.copy()
    rng.shuffle(ordered_components)
    efficiency_rows = []
    efficiency_audits = {}
    for size in (100, 240, 500, 1000):
        chosen_components = set(ordered_components[:size // 20].tolist())
        selected = np.asarray(
            [value in chosen_components for value in split_groups], dtype=bool
        )
        chosen_images = set(groups[selected].tolist())
        if len(chosen_images) != size:
            raise RuntimeError("data-efficiency component size mismatch")
        sub_oof, sub_audit = nonlinear_oof(
            x55[selected], utilities[selected], observed[selected],
            split_groups[selected], expert_names
        )
        sub_decisions, sub_gate_audit = cross_fitted_gate(
            sub_oof, utilities[selected], split_groups[selected],
            (0.0, 0.01, 0.02, 0.05, 0.10),
        )
        efficiency_rows.append({"available_base_images": size,
                                **metrics(sub_decisions, utilities[selected])})
        efficiency_audits[str(size)] = {
            "selected_group_sha256": hashlib.sha256(
                "\n".join(sorted(chosen_images)).encode("utf-8")
            ).hexdigest(),
            "selected_component_sha256": hashlib.sha256(
                "\n".join(str(v) for v in sorted(chosen_components)).encode("utf-8")
            ).hexdigest(),
            "model_folds": sub_audit,
            "gate_folds": sub_gate_audit,
        }

    payload = {
        "schema_version": 1,
        "status": "completed_development_only_grouped_experiment_suite",
        "seed": SEED,
        "scope": "development mechanism evidence; untouched test outputs never read",
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "router_report": str(router_report),
        "router_report_sha256": sha256(router_report),
        "rows": 15000,
        "base_image_groups": 1000,
        "batch_groups": int(len(np.unique(batch_ids))),
        "joint_components": int(len(component_ids)),
        "fold_grouping": "connected components over base-image IDs and batch IDs",
        "group_leakage_detected": False,
        "same_information_controls": controls,
        "expert_bank_ablation": bank_rows,
        "gate_ablation": gate_rows,
        "data_efficiency": efficiency_rows,
        "audits": {
            "best_single_fixed": fixed_audit,
            "corruption_classifier": class_audit,
            "ridge18_folds": ridge18_audit,
            "ridge18_gate": ridge18_gate_audit,
            "ridge55_folds": ridge55_audit,
            "ridge55_gate": ridge55_gate_audit,
            "hgb18_folds": hgb18_audit,
            "hgb18_gate": hgb18_gate_audit,
            "full_nonlinear_folds": full_fold_audit,
            "threshold_only_gate": threshold_audit,
            "complete_gate": complete_gate_audit,
            "data_efficiency": efficiency_audits,
        },
    }
    output.mkdir(parents=True)
    atomic_text(output / "report.json", json.dumps(payload, indent=2) + "\n")
    atomic_text(output / "table2_router_controls.csv", csv_text(controls))
    atomic_text(output / "table3_expert_bank_ablation.csv", csv_text(bank_rows))
    atomic_text(output / "table3_gate_ablation.csv", csv_text(gate_rows))
    atomic_text(output / "data_efficiency.csv", csv_text(efficiency_rows))
    manifest = {
        "status": payload["status"],
        "source_sha256": sha256(Path(__file__).resolve()),
        "files": {},
    }
    for path in sorted(output.glob("*")):
        if path.name != "manifest.json":
            manifest["files"][path.name] = {"bytes": path.stat().st_size,
                                             "sha256": sha256(path)}
    atomic_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(output),
                      "controls": controls, "expert_bank": bank_rows,
                      "gate": gate_rows, "data_efficiency": efficiency_rows}, indent=2))


if __name__ == "__main__":
    main()
