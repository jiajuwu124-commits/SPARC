#!/usr/bin/env python3
"""Small dynamic smoke test for the current SPARC paper pipeline."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streammint.nonlinear_utility_router import NonlinearUtilityRouter  # noqa: E402
from streammint.utility_router import (  # noqa: E402
    augment_batch_statistics,
    connected_component_groups,
    seeded_group_folds,
)


def synthetic_router_smoke() -> dict:
    rng = np.random.default_rng(1109)
    features = rng.normal(size=(300, 18))
    utilities = np.stack([
        .08 * features[:, index % 18] + .02 * rng.normal(size=len(features))
        for index in range(13)
    ], axis=1)
    observed = np.ones_like(utilities, dtype=bool)
    batches = np.asarray([f"batch:{index // 20}" for index in range(len(features))])
    x55 = augment_batch_statistics(features, batches, mode="mean_std_count")
    router = NonlinearUtilityRouter.fit(
        x55, utilities, observed, [f"expert_{index}" for index in range(13)],
        seed=1109,
        model_parameters={"max_iter": 12, "min_samples_leaf": 10},
    )
    selected, predicted = router.select(x55[:40])
    if predicted.shape != (40, 13) or not np.isfinite(predicted).all():
        raise RuntimeError("synthetic router produced invalid predictions")
    return {
        "rows": 300, "descriptor_dimensions": int(x55.shape[1]),
        "experts": 13, "predictions_checked": int(predicted.size),
        "selected_or_abstained": int(len(selected)),
    }


def load_rebuilder():
    path = ROOT / "scripts/rebuild_sparc_paper_summary.py"
    spec = importlib.util.spec_from_file_location("sparc_rebuilder_smoke", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load raw-result auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def real_evidence_smoke(evidence: Path) -> dict:
    rebuild = load_rebuilder()
    runs = evidence / "main/runs"
    parent = rebuild.audit_result(
        runs / "clip_gaussian_noise_s5_parent", "clip", "gaussian_noise:5"
    )
    plugin = rebuild.audit_result(
        runs / "clip_gaussian_noise_s5_plugin", "clip", "gaussian_noise:5"
    )
    left, right = parent["records"], plugin["records"]
    if [[row.get(field) for field in rebuild.PAIR_FIELDS] for row in left] != [
        [row.get(field) for field in rebuild.PAIR_FIELDS] for row in right
    ]:
        raise RuntimeError("small real parent/plugin pair is not aligned")

    dataset = evidence / "development/utility_targets_dev15.npz"
    with np.load(dataset, allow_pickle=False) as stored:
        raw = stored["features"].astype(np.float64)
        utilities = stored["utilities"].astype(np.float64)
        observed = stored["observed"].astype(bool)
        groups = stored["groups"].astype(str)
        batches = stored["batch_ids"].astype(str)
        names = tuple(str(value) for value in stored["expert_names"])
    components = connected_component_groups(groups, batches)
    chosen = set(sorted(np.unique(components).tolist())[:5])
    keep = np.asarray([value in chosen for value in components])
    x55 = augment_batch_statistics(raw[keep], batches[keep], mode="mean_std_count")
    y, mask, component_subset = utilities[keep], observed[keep], components[keep]
    train, validation = next(seeded_group_folds(component_subset, n_splits=5, seed=1109))
    if set(component_subset[train]) & set(component_subset[validation]):
        raise RuntimeError("small development split leaked a component")
    router = NonlinearUtilityRouter.fit(
        x55[train], y[train], mask[train], names, seed=1109,
        model_parameters={"max_iter": 12, "min_samples_leaf": 10},
    )
    selected, predicted = router.select(x55[validation])
    if not np.isfinite(predicted).all() or len(selected) != len(validation):
        raise RuntimeError("small real development router output is invalid")
    return {
        "paired_test_records_checked": len(left) + len(right),
        "parent_accuracy": parent["accuracy"],
        "plugin_accuracy": plugin["accuracy"],
        "gain_pp": plugin["accuracy"] - parent["accuracy"],
        "development_rows": int(keep.sum()),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "joint_component_overlap": 0,
        "finite_utility_predictions": int(predicted.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path)
    args = parser.parse_args()
    report = {
        "status": "passed_current_paper_small_dynamic_smoke",
        "seed": 1109,
        "synthetic_component_smoke": synthetic_router_smoke(),
    }
    if args.evidence_root:
        report["real_evidence_smoke"] = real_evidence_smoke(args.evidence_root.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
