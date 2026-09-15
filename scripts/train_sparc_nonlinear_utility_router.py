#!/usr/bin/env python3
"""Grouped-CV development training for nonlinear direct-utility routing."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import sklearn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streammint.nonlinear_utility_router import (  # noqa: E402
    NonlinearUtilityRouter,
)
from streammint.utility_router import (  # noqa: E402
    RISK_FEATURE_ORDER,
    augment_batch_statistics,
    seeded_group_folds,
)


SEED = 1109


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decisions_from_predictions(
    predicted: np.ndarray, threshold: float, margin: float
) -> np.ndarray:
    best = np.argmax(predicted, axis=1)
    best_value = predicted[np.arange(len(predicted)), best]
    second = (
        np.partition(predicted, -2, axis=1)[:, -2]
        if predicted.shape[1] > 1
        else np.zeros(len(predicted), dtype=np.float64)
    )
    accepted = (best_value > threshold) & (
        best_value - np.maximum(second, 0.0) >= margin
    )
    return np.where(accepted, best, -1)


def routed_metrics(decisions: np.ndarray, utilities: np.ndarray) -> dict[str, float]:
    rows = np.arange(len(decisions))
    routed = decisions >= 0
    realized = np.zeros(len(decisions), dtype=np.float64)
    realized[routed] = utilities[rows[routed], decisions[routed]]
    return {
        "mean_utility": float(realized.mean()),
        "coverage": float(routed.mean()),
        "harm_rate": float((realized < 0).mean()),
        "benefit_rate": float((realized > 0).mean()),
        "mean_utility_when_routed": float(realized[routed].mean()) if routed.any() else 0.0,
    }


def calibrate_gate(
    predicted: np.ndarray,
    utilities: np.ndarray,
    *,
    max_harm_rate: float,
    min_coverage: float,
) -> tuple[float, float, dict[str, float]]:
    finite = predicted[np.isfinite(predicted)]
    if not len(finite):
        raise ValueError("no finite utility predictions")
    thresholds = np.unique(
        np.maximum(
            0.0,
            np.concatenate(([0.0], np.quantile(finite, np.linspace(0, 1, 41)))),
        )
    )
    margins = (0.0, 0.01, 0.02, 0.05, 0.10)
    candidates = []
    for threshold in thresholds:
        for margin in margins:
            metrics = routed_metrics(
                decisions_from_predictions(predicted, float(threshold), margin),
                utilities,
            )
            if metrics["harm_rate"] <= max_harm_rate and metrics["coverage"] >= min_coverage:
                key = (
                    metrics["mean_utility"],
                    -metrics["harm_rate"],
                    metrics["coverage"],
                    -float(threshold),
                    -margin,
                )
                candidates.append((key, float(threshold), margin, metrics))
    if not candidates:
        return float(np.max(finite)), 0.0, routed_metrics(
            np.full(len(predicted), -1, dtype=np.int64), utilities
        )
    _key, threshold, margin, metrics = max(candidates, key=lambda item: item[0])
    return threshold, margin, metrics


def selection_distribution(
    decisions: np.ndarray, predicted: np.ndarray, expert_names: tuple[str, ...]
) -> dict:
    routed = decisions[decisions >= 0]
    counts = Counter("identity" if value < 0 else expert_names[int(value)] for value in decisions)
    predicted_best = np.argmax(predicted, axis=1)
    best_counts = Counter(expert_names[int(value)] for value in predicted_best)
    routed_counts = {name: int(counts.get(name, 0)) for name in expert_names}
    active = sum(value > 0 for value in routed_counts.values())
    maximum_share = (
        max(routed_counts.values()) / len(routed) if len(routed) else 0.0
    )
    return {
        "identity": int(counts.get("identity", 0)),
        "routed": routed_counts,
        "predicted_best_before_gate": {
            name: int(best_counts.get(name, 0)) for name in expert_names
        },
        "active_routed_experts": active,
        "maximum_routed_expert_share": maximum_share,
        "collapsed_to_one_expert": bool(len(routed) and active == 1),
    }


def train_mode(
    mode: str,
    raw_features: np.ndarray,
    utilities: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    batch_ids: np.ndarray,
    expert_names: tuple[str, ...],
    *,
    folds_count: int,
    max_harm_rate: float,
    min_coverage: float,
) -> tuple[NonlinearUtilityRouter, dict]:
    x = augment_batch_statistics(raw_features, batch_ids, mode=mode)
    folds = list(seeded_group_folds(groups, n_splits=folds_count, seed=SEED))
    oof = np.full_like(utilities, np.nan, dtype=np.float64)
    fold_audit = []
    for fold, (train, validation) in enumerate(folds):
        overlap = set(groups[train].tolist()) & set(groups[validation].tolist())
        if overlap:
            raise RuntimeError(f"base-image leakage in fold {fold}")
        router = NonlinearUtilityRouter.fit(
            x[train], utilities[train], observed[train], expert_names,
            feature_mode=mode, seed=SEED,
        )
        oof[validation] = router.predict_utility(x[validation])
        fold_audit.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "train_groups": int(len(np.unique(groups[train]))),
                "validation_groups": int(len(np.unique(groups[validation]))),
                "group_overlap": 0,
            }
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("grouped OOF prediction matrix is incomplete")
    threshold, margin, fit_metrics = calibrate_gate(
        oof, utilities,
        max_harm_rate=max_harm_rate,
        min_coverage=min_coverage,
    )
    cross_fitted = np.full(len(x), -1, dtype=np.int64)
    calibration_audit = []
    all_rows = np.arange(len(x))
    for fold, (_train, validation) in enumerate(folds):
        calibration = np.setdiff1d(all_rows, validation, assume_unique=True)
        fold_threshold, fold_margin, _ = calibrate_gate(
            oof[calibration], utilities[calibration],
            max_harm_rate=max_harm_rate,
            min_coverage=min_coverage,
        )
        cross_fitted[validation] = decisions_from_predictions(
            oof[validation], fold_threshold, fold_margin
        )
        calibration_audit.append(
            {
                "fold": fold,
                "threshold": fold_threshold,
                "margin": fold_margin,
                "calibration_rows": int(len(calibration)),
                "evaluation_rows": int(len(validation)),
            }
        )
    final = NonlinearUtilityRouter.fit(
        x, utilities, observed, expert_names, feature_mode=mode, seed=SEED
    )
    final.utility_threshold = threshold
    final.min_margin = margin
    report = {
        "feature_mode": mode,
        "input": (
            "18D masked-risk descriptor only"
            if mode == "none"
            else "18D masked-risk descriptor plus current-batch mean/std/log-count"
        ),
        "feature_dimensions": int(x.shape[1]),
        "grouped_cross_fitted_gate": routed_metrics(cross_fitted, utilities),
        "development_gate_fit_on_all_oof": fit_metrics,
        "selection_distribution": selection_distribution(
            cross_fitted, oof, expert_names
        ),
        "frozen_threshold": threshold,
        "frozen_margin": margin,
        "folds": fold_audit,
        "gate_cross_fit_folds": calibration_audit,
        "oof_prediction_sha256": hashlib.sha256(
            np.asarray(oof, dtype="<f8").tobytes()
        ).hexdigest(),
        "group_leakage_detected": False,
    }
    return final, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-modes", default="none,mean_std_count")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-harm-rate", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("SPARC nonlinear utility-router protocol requires seed 1109")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    modes = tuple(value for value in args.feature_modes.split(",") if value)
    if set(modes) != {"none", "mean_std_count"} or len(modes) != 2:
        raise ValueError("matched comparison requires exactly none,mean_std_count")
    dataset = args.dataset.resolve()
    dataset_report_path = args.dataset_report.resolve()
    dataset_report = json.loads(dataset_report_path.read_text())
    if dataset_report.get("status") != "development_utility_targets_not_confirmation_evidence":
        raise RuntimeError("only development utility evidence is allowed")
    if dataset_report.get("dataset_sha256") != sha256(dataset):
        raise RuntimeError("dataset/report hash mismatch")
    with np.load(dataset, allow_pickle=False) as stored:
        raw_features = stored["features"].astype(np.float64)
        utilities = stored["utilities"].astype(np.float64)
        observed = stored["observed"].astype(bool)
        groups = stored["groups"]
        batch_ids = stored["batch_ids"]
        expert_names = tuple(str(value) for value in stored["expert_names"].tolist())
        feature_order = tuple(str(value) for value in stored["feature_order"].tolist())
        dataset_seed = int(stored["seed"].item())
    if dataset_seed != SEED or feature_order != RISK_FEATURE_ORDER:
        raise RuntimeError("dataset seed or frozen 18D feature order mismatch")
    if raw_features.shape != (15000, 18) or utilities.shape != (15000, 12):
        raise RuntimeError("expected the frozen 15K x 12 development utility dataset")
    if observed.shape != utilities.shape or not observed.all():
        raise RuntimeError("nonlinear training requires the complete expert matrix")
    if not np.isfinite(raw_features).all() or not np.isfinite(utilities).all():
        raise RuntimeError("training arrays contain non-finite values")
    if len(np.unique(groups)) != 1000:
        raise RuntimeError("expected 1,000 disjoint base-image groups")

    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    reports = {}
    for mode in modes:
        router, report = train_mode(
            mode, raw_features, utilities, observed, groups, batch_ids,
            expert_names, folds_count=args.folds,
            max_harm_rate=args.max_harm_rate, min_coverage=args.min_coverage,
        )
        artifact = args.output_dir / f"histgb_direct_utility_{mode}.joblib"
        router.save(artifact)
        reloaded = NonlinearUtilityRouter.load(artifact)
        probe = augment_batch_statistics(raw_features[:20], batch_ids[:20], mode=mode)
        np.testing.assert_allclose(
            router.predict_utility(probe), reloaded.predict_utility(probe),
            rtol=0.0, atol=0.0,
        )
        report["model_artifact"] = str(artifact.resolve())
        report["model_sha256"] = sha256(artifact)
        reports[mode] = report
    # Primary criterion is cross-fitted mean utility. Ties prefer lower harm,
    # then higher coverage, then the simpler 18-D no-context representation.
    selected_mode = max(
        modes,
        key=lambda mode: (
            reports[mode]["grouped_cross_fitted_gate"]["mean_utility"],
            -reports[mode]["grouped_cross_fitted_gate"]["harm_rate"],
            reports[mode]["grouped_cross_fitted_gate"]["coverage"],
            mode == "none",
        ),
    )
    summary = {
        "schema_version": 1,
        "status": "development_only_nonlinear_direct_utility_router_not_confirmation_evidence",
        "seed": SEED,
        "model": "one sklearn HistGradientBoostingRegressor per expert",
        "model_parameters": NonlinearUtilityRouter.load(
            reports[selected_mode]["model_artifact"]
        ).model_parameters,
        "input_candidates": ["18D masked-risk", "18D masked-risk + batch mean/std/log-count (55D)"],
        "forbidden_inputs": ["corruption name", "image class label", "diagnostic A/B", "backend logits"],
        "target": "development matched expert utility relative to identity",
        "feature_mode_selection_rule": "highest grouped cross-fitted mean utility; ties lower harm, higher coverage, then none",
        "selected_feature_mode": selected_mode,
        "selected_model_artifact": reports[selected_mode]["model_artifact"],
        "selected_model_sha256": reports[selected_mode]["model_sha256"],
        "max_harm_rate_constraint": args.max_harm_rate,
        "min_coverage_constraint": args.min_coverage,
        "modes": reports,
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "dataset_report": str(dataset_report_path),
        "dataset_report_sha256": sha256(dataset_report_path),
        "trainer_sha256": sha256(Path(__file__)),
        "model_module_sha256": sha256(ROOT / "src/streammint/nonlinear_utility_router.py"),
        "fold_module_sha256": sha256(ROOT / "src/streammint/utility_router.py"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "diagnostic_or_confirmation_data_read": False,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
