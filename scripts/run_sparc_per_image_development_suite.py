#!/usr/bin/env python3
"""Run the final SPARC development controls with realized per-image metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from io import StringIO
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streammint.nonlinear_utility_router import NonlinearUtilityRouter  # noqa: E402
from streammint.utility_router import (  # noqa: E402
    UtilityRouter,
    augment_batch_statistics,
    connected_component_groups,
    seeded_group_folds,
)


SEED = 1109


def group_digest(values: np.ndarray) -> str:
    unique = sorted(str(value) for value in np.unique(values).tolist())
    return hashlib.sha256("\n".join(unique).encode("utf-8")).hexdigest()


def proxy_metrics(decisions_: np.ndarray, utilities: np.ndarray) -> dict[str, float]:
    rows = np.arange(len(decisions_))
    routed = decisions_ >= 0
    realized = np.zeros(len(decisions_), dtype=np.float64)
    realized[routed] = utilities[rows[routed], decisions_[routed]]
    return {
        "mean_utility_pp": 100.0 * float(realized.mean()),
        "coverage_pct": 100.0 * float(routed.mean()),
        "harm_pct": 100.0 * float((realized < 0).mean()),
    }


def route_decisions(predicted: np.ndarray, threshold: float, margin: float) -> np.ndarray:
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
            current = proxy_metrics(
                route_decisions(predicted, float(threshold), float(margin)), utilities,
            )
            if current["harm_pct"] <= 5.0 and current["coverage_pct"] >= 5.0:
                key = (current["mean_utility_pp"], -current["harm_pct"],
                       current["coverage_pct"], -float(threshold), -float(margin))
                candidates.append((key, float(threshold), float(margin)))
    if not candidates:
        return float(np.max(finite)), 0.0
    return max(candidates, key=lambda item: item[0])[1:]


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


def csv_text(rows: list[dict]) -> str:
    target = StringIO(newline="")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue()


def load_inputs(dataset: Path, transitions: Path) -> dict:
    with np.load(dataset, allow_pickle=False) as stored:
        data = {
            "features": stored["features"].astype(np.float64),
            "utilities": stored["utilities"].astype(np.float64),
            "observed": stored["observed"].astype(bool),
            "groups": stored["groups"].astype(str),
            "batch_ids": stored["batch_ids"].astype(str),
            "samples": stored["samples"].astype(np.int64),
            "scenarios": stored["scenarios"].astype(str),
            "raw_sha256": stored["raw_sha256"].astype(str),
            "expert_names": tuple(str(value) for value in stored["expert_names"]),
            "seed": int(stored["seed"].item()),
        }
    with np.load(transitions, allow_pickle=False) as stored:
        expected_dataset_hash = str(stored["dataset_sha256"].item())
        if expected_dataset_hash != sha256(dataset):
            raise RuntimeError("transition evidence/dataset hash mismatch")
        for key in ("samples", "scenarios", "groups", "batch_ids", "raw_sha256"):
            if not np.array_equal(data[key], stored[key]):
                raise RuntimeError(f"transition evidence differs at {key}")
        names = tuple(str(value) for value in stored["expert_names"])
        if names != data["expert_names"]:
            raise RuntimeError("expert order mismatch")
        if int(stored["seed"].item()) != data["seed"]:
            raise RuntimeError("seed mismatch")
        data["deltas"] = stored["deltas"].astype(np.int8)
        data["identity_correct"] = stored["identity_correct"].astype(bool)
        data["expert_correct"] = stored["expert_correct"].astype(bool)
    if data["seed"] != SEED or not data["observed"].all():
        raise RuntimeError("unexpected frozen development contract")
    if data["deltas"].shape != data["utilities"].shape:
        raise RuntimeError("transition matrix shape mismatch")
    if not np.array_equal(
        data["deltas"],
        data["expert_correct"].astype(np.int8) - data["identity_correct"][:, None].astype(np.int8),
    ):
        raise RuntimeError("stored correctness arrays do not reconstruct deltas")
    return data


def verify_batch_targets(deltas: np.ndarray, utilities: np.ndarray, batch_ids: np.ndarray) -> float:
    maximum_error = 0.0
    for batch_id in np.unique(batch_ids):
        rows = np.flatnonzero(batch_ids == batch_id)
        if len(rows) != 20:
            raise RuntimeError(f"batch {batch_id} has {len(rows)} rows")
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(deltas[rows].mean(axis=0) - utilities[rows[0]]))),
        )
    if maximum_error > 1e-12:
        raise RuntimeError(f"batch-target reconstruction error: {maximum_error}")
    return maximum_error


def per_image_metrics(decisions: np.ndarray, deltas: np.ndarray) -> dict[str, float | int]:
    rows = np.arange(len(decisions))
    routed = decisions >= 0
    realized = np.zeros(len(decisions), dtype=np.int8)
    realized[routed] = deltas[rows[routed], decisions[routed]]
    fix = 100.0 * float(np.mean(realized == 1))
    break_ = 100.0 * float(np.mean(realized == -1))
    delta = 100.0 * float(realized.mean())
    if abs(delta - (fix - break_)) > 1e-12:
        raise RuntimeError("DeltaAcc != Fix - Break")
    return {
        "active_experts": int(len(np.unique(decisions[routed]))) if routed.any() else 0,
        "route_pct": 100.0 * float(routed.mean()),
        "delta_accuracy_pp": delta,
        "fix_pct": fix,
        "break_pct": break_,
        "unchanged_pct": 100.0 * float(np.mean(realized == 0)),
    }


def best_single_decisions(
    utilities: np.ndarray, groups: np.ndarray, fixed_indices: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    routed = np.full(len(utilities), -1, dtype=np.int64)
    audits = []
    for fold, (train, validation) in enumerate(
        seeded_group_folds(groups, n_splits=5, seed=SEED)
    ):
        local = int(np.argmax(utilities[train][:, fixed_indices].mean(axis=0)))
        chosen = int(fixed_indices[local])
        routed[validation] = chosen
        audits.append({
            "fold": fold,
            "expert_index": chosen,
            "validation_group_sha256": group_digest(groups[validation]),
        })
    return routed, audits


def fit_decisions(
    kind: str,
    x: np.ndarray,
    utilities: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    expert_names: tuple[str, ...],
) -> tuple[np.ndarray, dict]:
    def train_predict(train: np.ndarray, predict: np.ndarray) -> np.ndarray:
        if kind == "linear":
            router = UtilityRouter.fit(
                x[train], utilities[train], observed[train], expert_names,
                feature_mode="none", ridge_alpha=1.0,
            )
        elif kind == "nonlinear":
            router = NonlinearUtilityRouter.fit(
                x[train], utilities[train], observed[train], expert_names,
                feature_mode="mean_std_count", seed=SEED,
            )
        else:
            raise ValueError(kind)
        return router.predict_utility(x[predict])

    routed = np.full(len(x), -1, dtype=np.int64)
    filled = np.zeros(len(x), dtype=bool)
    audits = []
    margins = (0.0, 0.01, 0.02, 0.05, 0.10)
    for outer_fold, (outer_train, outer_validation) in enumerate(
        seeded_group_folds(groups, n_splits=5, seed=SEED)
    ):
        train_groups = set(groups[outer_train].tolist())
        validation_groups = set(groups[outer_validation].tolist())
        if train_groups & validation_groups:
            raise RuntimeError(f"outer group leakage in fold {outer_fold}")
        local_groups = groups[outer_train]
        inner_splits = min(5, len(np.unique(local_groups)))
        if inner_splits < 2:
            raise RuntimeError(f"too few inner groups in outer fold {outer_fold}")
        inner_oof = np.full((len(outer_train), utilities.shape[1]), np.nan)
        inner_audits = []
        for inner_fold, (inner_train_local, inner_validation_local) in enumerate(
            seeded_group_folds(local_groups, n_splits=inner_splits, seed=SEED)
        ):
            inner_train = outer_train[inner_train_local]
            inner_validation = outer_train[inner_validation_local]
            overlap = set(groups[inner_train].tolist()) & set(groups[inner_validation].tolist())
            if overlap:
                raise RuntimeError(f"inner group leakage in {outer_fold}/{inner_fold}")
            inner_oof[inner_validation_local] = train_predict(inner_train, inner_validation)
            inner_audits.append({"fold": inner_fold, "group_overlap": 0})
        if not np.isfinite(inner_oof).all():
            raise RuntimeError(f"incomplete inner OOF predictions in outer fold {outer_fold}")
        threshold, margin = calibrate(inner_oof, utilities[outer_train], margins)
        prediction = train_predict(outer_train, outer_validation)
        routed[outer_validation] = route_decisions(prediction, threshold, margin)
        filled[outer_validation] = True
        audits.append({
            "fold": outer_fold,
            "train_rows": int(len(outer_train)),
            "validation_rows": int(len(outer_validation)),
            "group_overlap": 0,
            "threshold": threshold,
            "margin": margin,
            "inner_folds": inner_audits,
        })
    if not filled.all():
        raise RuntimeError("incomplete outer OOF decisions")
    return routed, {"nested_outer_folds": audits}


def fold_audit(groups: np.ndarray, batch_ids: np.ndarray, components: np.ndarray) -> list[dict]:
    rows = []
    for fold, (train, validation) in enumerate(
        seeded_group_folds(components, n_splits=5, seed=SEED)
    ):
        image_overlap = set(groups[train]) & set(groups[validation])
        batch_overlap = set(batch_ids[train]) & set(batch_ids[validation])
        component_overlap = set(components[train]) & set(components[validation])
        if image_overlap or batch_overlap or component_overlap:
            raise RuntimeError(f"fold {fold} is not isolated")
        rows.append({
            "fold": fold,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "base_image_overlap": 0,
            "batch_overlap": 0,
            "component_overlap": 0,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    transitions = args.transitions.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    data = load_inputs(dataset, transitions)
    raw = data["features"]
    utilities = data["utilities"]
    observed = data["observed"]
    groups = data["groups"]
    batch_ids = data["batch_ids"]
    deltas = data["deltas"]
    expert_names = data["expert_names"]
    maximum_error = verify_batch_targets(deltas, utilities, batch_ids)
    components = connected_component_groups(groups, batch_ids)
    isolation = fold_audit(groups, batch_ids, components)
    x55 = augment_batch_statistics(raw, batch_ids, mode="mean_std_count")
    fixed = np.asarray([0, 1, 2, 3], dtype=np.int64)
    banks = (
        ("Fixed operators", np.asarray([0, 1, 2, 3], dtype=np.int64)),
        ("+ blur specialists", np.asarray([0, 1, 2, 3, 5, 6, 7], dtype=np.int64)),
        ("+ weather specialists", np.asarray([0, 1, 2, 3, 5, 6, 7, 8, 9, 10], dtype=np.int64)),
        ("Full expert bank", np.arange(13, dtype=np.int64)),
    )
    audits: dict[str, object] = {}
    controls: list[dict] = []

    decisions, audit = best_single_decisions(utilities, components, fixed)
    controls.append({"router": "Best single fixed expert", "model": "--",
                     **per_image_metrics(decisions, deltas)})
    audits["best_single_fixed"] = audit
    for name, model, kind, features in (
        ("Ridge direct utility", "Linear", "linear", raw),
        ("Ridge direct utility + batch", "Linear", "linear", x55),
        ("Nonlinear direct utility", "Nonlinear", "nonlinear", raw),
    ):
        decisions, audit = fit_decisions(
            kind, features, utilities, observed, components, expert_names,
        )
        controls.append({"router": name, "model": model, **per_image_metrics(decisions, deltas)})
        audits[name] = audit
    full_decisions, full_audit = fit_decisions(
        "nonlinear", x55, utilities, observed, components, expert_names,
    )
    controls.append({"router": "SPARC + batch", "model": "Nonlinear",
                     **per_image_metrics(full_decisions, deltas)})
    audits["SPARC + batch"] = full_audit

    bank_rows: list[dict] = []
    for name, indices in banks[:-1]:
        decisions, audit = fit_decisions(
            "nonlinear", x55, utilities[:, indices], observed[:, indices], components,
            tuple(expert_names[index] for index in indices),
        )
        bank_rows.append({"expert_bank": name, "experts": int(len(indices)),
                          **per_image_metrics(decisions, deltas[:, indices])})
        audits[f"bank:{name}"] = audit
    bank_rows.append({"expert_bank": "Full expert bank", "experts": 13,
                      **per_image_metrics(full_decisions, deltas)})

    component_ids = np.unique(components)
    rng = np.random.default_rng(SEED)
    rng.shuffle(component_ids)
    efficiency: list[dict] = []
    for size in (100, 240, 500, 1000):
        chosen = set(component_ids[: size // 20].tolist())
        selected = np.asarray([value in chosen for value in components], dtype=bool)
        if len(np.unique(groups[selected])) != size:
            raise RuntimeError(f"nested subset does not contain {size} images")
        decisions, audit = fit_decisions(
            "nonlinear", x55[selected], utilities[selected], observed[selected],
            components[selected], expert_names,
        )
        efficiency.append({"available_base_images": size,
                           **per_image_metrics(decisions, deltas[selected])})
        audits[f"data_efficiency:{size}"] = audit

    display_banks = [bank_rows[0], bank_rows[1], bank_rows[-1]]
    output.mkdir(parents=True)
    atomic_text(output / "table2_router_controls.csv", csv_text(controls))
    atomic_text(output / "table3_nested_expert_bank_complete.csv", csv_text(bank_rows))
    atomic_text(output / "table3_ablations.csv", csv_text(display_banks))
    atomic_text(output / "data_efficiency.csv", csv_text(efficiency))
    report = {
        "schema_version": 1,
        "status": "completed_final_per_image_development_suite",
        "seed": SEED,
        "training_target": "20-image batch top-1 utility",
        "evaluation_unit": "individual out-of-fold image observation",
        "metric_definitions": {
            "fix_pct": "100 * count(identity wrong and selected expert correct) / all observations",
            "break_pct": "100 * count(identity correct and selected expert wrong) / all observations",
            "delta_accuracy_pp": "fix_pct - break_pct",
        },
        "fold_grouping": "connected components over base-image IDs and batch IDs",
        "gate_calibration": "inner OOF predictions inside each outer training fold",
        "dataset_sha256": sha256(dataset),
        "transitions_sha256": sha256(transitions),
        "rows": len(raw),
        "max_batch_target_reconstruction_error": maximum_error,
        "table2_router_controls": controls,
        "table3_nested_expert_bank_complete": bank_rows,
        "table3_display_rows": display_banks,
        "figure3_data_efficiency": efficiency,
        "fold_isolation_audit": isolation,
        "model_and_gate_audits": audits,
    }
    atomic_text(output / "report.json", json.dumps(report, indent=2) + "\n")
    manifest = {
        "status": report["status"],
        "dataset_sha256": sha256(dataset),
        "transitions_sha256": sha256(transitions),
        "outputs": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    atomic_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "table2_rows": len(controls),
        "table3_complete_rows": len(bank_rows),
        "data_efficiency_rows": len(efficiency),
        "folds": len(isolation),
    }, indent=2))


if __name__ == "__main__":
    main()
