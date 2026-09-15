#!/usr/bin/env python3
"""Retrain nonlinear utility routing after an audited expert-bank extension.

Unlike the frozen 12-expert trainer, this independent entry point accepts a
complete 15K x E development utility matrix.  It reuses the exact grouped-CV
training implementation and rejects every un-hashed input/source dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import sklearn


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINER = ROOT / "scripts/train_sparc_nonlinear_utility_router.py"
MODEL_MODULE = ROOT / "src/streammint/nonlinear_utility_router.py"
FOLD_MODULE = ROOT / "src/streammint/utility_router.py"
SEED = 1109


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {actual}")


def load_base():
    spec = importlib.util.spec_from_file_location("sparc_nonlinear_trainer_frozen", BASE_TRAINER)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_TRAINER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--dataset-report-sha256", required=True)
    parser.add_argument("--base-trainer-sha256", required=True)
    parser.add_argument("--model-module-sha256", required=True)
    parser.add_argument("--fold-module-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-modes", default="none,mean_std_count")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-harm-rate", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("extended nonlinear router requires seed 1109")
    modes = tuple(value for value in args.feature_modes.split(",") if value)
    if set(modes) != {"none", "mean_std_count"} or len(modes) != 2:
        raise ValueError("matched comparison requires none,mean_std_count")
    dataset = args.dataset.resolve()
    report_path = args.dataset_report.resolve()
    require_hash(dataset, args.dataset_sha256, "utility dataset")
    require_hash(report_path, args.dataset_report_sha256, "utility report")
    require_hash(BASE_TRAINER, args.base_trainer_sha256, "base trainer")
    require_hash(MODEL_MODULE, args.model_module_sha256, "model module")
    require_hash(FOLD_MODULE, args.fold_module_sha256, "fold module")
    report = json.loads(report_path.read_text())
    if report.get("status") != "development_utility_targets_not_confirmation_evidence":
        raise RuntimeError("only development utility evidence is allowed")
    if report.get("dataset_sha256") != args.dataset_sha256:
        raise RuntimeError("utility report does not bind dataset")
    with np.load(dataset, allow_pickle=False) as stored:
        features = stored["features"].astype(np.float64)
        utilities = stored["utilities"].astype(np.float64)
        observed = stored["observed"].astype(bool)
        groups = stored["groups"]
        batch_ids = stored["batch_ids"]
        expert_names = tuple(str(v) for v in stored["expert_names"].tolist())
        feature_order = tuple(str(v) for v in stored["feature_order"].tolist())
        dataset_seed = int(stored["seed"].item())
    base = load_base()
    if dataset_seed != SEED or feature_order != tuple(base.RISK_FEATURE_ORDER):
        raise RuntimeError("dataset seed or frozen 18D feature order mismatch")
    if features.shape != (15000, 18):
        raise RuntimeError(f"expected 15K x 18 features, got {features.shape}")
    if utilities.shape != (15000, len(expert_names)) or len(expert_names) < 2:
        raise RuntimeError("utility matrix/expert-name shape mismatch")
    if observed.shape != utilities.shape or not observed.all():
        raise RuntimeError("extended training requires a complete utility matrix")
    if len(set(expert_names)) != len(expert_names):
        raise RuntimeError("expert names must be unique")
    if len(np.unique(groups)) != 1000:
        raise RuntimeError("expected 1,000 base-image groups")
    if not np.isfinite(features).all() or not np.isfinite(utilities).all():
        raise RuntimeError("non-finite training arrays")
    validation = {
        "valid": True,
        "seed": SEED,
        "rows": len(features),
        "experts": list(expert_names),
        "feature_modes": list(modes),
        "dataset_sha256": args.dataset_sha256,
        "dataset_report_sha256": args.dataset_report_sha256,
        "source_sha256": {
            "base_trainer": args.base_trainer_sha256,
            "model_module": args.model_module_sha256,
            "fold_module": args.fold_module_sha256,
            "extended_trainer": sha256(Path(__file__).resolve()),
        },
        "diagnostic_or_confirmation_data_read": False,
    }
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    reports = {}
    for mode in modes:
        router, mode_report = base.train_mode(
            mode, features, utilities, observed, groups, batch_ids, expert_names,
            folds_count=args.folds, max_harm_rate=args.max_harm_rate,
            min_coverage=args.min_coverage,
        )
        artifact = args.output_dir / f"histgb_direct_utility_{mode}.joblib"
        router.save(artifact)
        reloaded = base.NonlinearUtilityRouter.load(artifact)
        probe = base.augment_batch_statistics(features[:20], batch_ids[:20], mode=mode)
        np.testing.assert_allclose(
            router.predict_utility(probe), reloaded.predict_utility(probe), rtol=0, atol=0
        )
        mode_report["model_artifact"] = str(artifact.resolve())
        mode_report["model_sha256"] = sha256(artifact)
        reports[mode] = mode_report
    selected = max(
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
        "status": "development_only_extended_nonlinear_router_not_confirmation_evidence",
        **{key: value for key, value in validation.items() if key != "valid"},
        "model": "one sklearn HistGradientBoostingRegressor per expert",
        "selected_feature_mode": selected,
        "selected_model_artifact": reports[selected]["model_artifact"],
        "selected_model_sha256": reports[selected]["model_sha256"],
        "selection_rule": "highest grouped-CV mean utility; ties lower harm, higher coverage, simpler mode",
        "max_harm_rate_constraint": args.max_harm_rate,
        "min_coverage_constraint": args.min_coverage,
        "modes": reports,
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
