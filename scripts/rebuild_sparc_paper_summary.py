#!/usr/bin/env python3
"""Rebuild the paper's Full-15 summary from the released raw predictions.

This is deliberately independent of the GPU executor and its absolute frozen
paths.  It validates each result/protocol/completion triplet, reconstructs all
75 paired cells and five family/overall aggregates per backend, and checks the
result against the frozen paper summary without selecting or dropping cells.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np


BACKENDS = ("clip", "tent", "clipartt", "tda", "mint")
SCENARIOS = (
    "gaussian_noise:5", "shot_noise:5", "impulse_noise:5",
    "defocus_blur:5", "glass_blur:5", "motion_blur:5", "zoom_blur:5",
    "snow:5", "frost:5", "fog:5", "brightness:5", "contrast:5",
    "elastic_transform:5", "pixelate:5", "jpeg_compression:5",
)
FAMILIES = {
    "Noise": SCENARIOS[:3], "Blur": SCENARIOS[3:7],
    "Weather": SCENARIOS[7:11], "Digital": SCENARIOS[11:],
}
PAIR_FIELDS = (
    "image_id", "sample", "target", "raw_sha256", "descriptor_sha256",
    "stream_position", "batch",
)
PROTOCOL_PAIR_FIELDS = (
    "backend", "batch_size", "seed", "runner_sha256", "model_sha256",
    "prompt_sha256", "reset_policy", "native_reset_policy", "memory_fraction",
    "stream_keys_sha256", "stream_order_sha256", "test_ids_sequence_sha256",
)
RUN_RE = re.compile(
    r"^(clip|tent|clipartt|tda|mint)_(.+)_s5_(parent|plugin)$"
)


class ReproductionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReproductionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(rendered).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_run_name(path: Path) -> tuple[str, str, str]:
    match = RUN_RE.match(path.name)
    require(match is not None, f"unexpected run directory: {path.name}")
    backend, corruption, role = match.groups()
    return backend, f"{corruption}:5", role


def audit_result(path: Path, backend: str, scenario: str) -> dict[str, Any]:
    result_path = path / "result.json"
    protocol_path = path / "protocol.json"
    complete_path = path / "complete.json"
    require(result_path.is_file() and protocol_path.is_file() and complete_path.is_file(),
            f"missing result/protocol/complete triplet: {path}")
    result, protocol, complete = load(result_path), load(protocol_path), load(complete_path)
    protocol_hash, result_hash = sha256(protocol_path), sha256(result_path)
    require(complete.get("result_sha256") == result_hash, f"result hash mismatch: {path}")
    require(complete.get("protocol_sha256") == protocol_hash, f"protocol hash mismatch: {path}")
    require(result.get("protocol_sha256") == protocol_hash and result.get("protocol") == protocol,
            f"embedded protocol mismatch: {path}")
    records = result.get("records")
    require(isinstance(records, list) and len(records) == 2000,
            f"expected exactly 2,000 records: {path}")
    records_hash = canonical_sha256(records)
    require(result.get("records_sha256") == records_hash
            and complete.get("records_sha256") == records_hash,
            f"record hash mismatch: {path}")
    accuracy = 100.0 * sum(bool(row.get("correct")) for row in records) / len(records)
    require(math.isclose(accuracy, float(result.get("accuracy")), abs_tol=1e-12),
            f"accuracy is not reconstructible: {path}")
    require(protocol.get("status") == "frozen_five_backend_routed_replay",
            f"not a frozen paper run: {path}")
    require(protocol.get("backend") == backend and protocol.get("seed") == 1109,
            f"backend/seed mismatch: {path}")
    require(protocol.get("batch_size") == 20 and float(protocol.get("memory_fraction")) <= .70,
            f"batch or memory contract mismatch: {path}")
    require(bool(protocol.get("memory_gate_pass")), f"memory gate failed: {path}")
    require(all(str(row.get("audit_domain")) == scenario for row in records),
            f"scenario mismatch: {path}")
    require(all(int(row.get("stream_position", -1)) == index
                for index, row in enumerate(records)), f"stream order mismatch: {path}")
    return {"accuracy": accuracy, "records": records, "protocol": protocol}


def bootstrap_ci(matrix: np.ndarray, label: str, replicates: int = 10000,
                 seed: int = 99173) -> list[float]:
    derived = seed ^ int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(derived)
    values = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 250):
        stop = min(start + 250, replicates)
        selected = rng.integers(0, len(matrix), size=(stop - start, len(matrix)))
        values[start:stop] = matrix[selected].mean(axis=(1, 2)) * 100.0
    return [float(value) for value in np.quantile(values, (.025, .975))]


def compare_rows(actual: list[dict[str, Any]], expected: list[dict[str, Any]],
                 keys: tuple[str, str], label: str) -> None:
    right = {(str(row[keys[0]]), str(row[keys[1]])): row for row in expected}
    require(len(actual) == len(expected) == len(right), f"{label} inventory mismatch")
    for row in actual:
        key = (str(row[keys[0]]), str(row[keys[1]]))
        require(key in right, f"missing reference {label}: {key}")
        reference = right[key]
        for field in ("parent_accuracy", "plugin_accuracy", "gain_pp"):
            require(math.isclose(float(row[field]), float(reference[field]), abs_tol=1e-12),
                    f"{label} mismatch: {key}/{field}")
        require(np.allclose(row["ci95_pp"], reference["ci95_pp"], rtol=0, atol=1e-12),
                f"{label} CI mismatch: {key}")
        require(row["expert_counts"] == reference["expert_counts"],
                f"{label} routing mismatch: {key}")


def rebuild(runs: Path, reference_path: Path, output: Path) -> dict[str, Any]:
    reference = load(reference_path)
    directories = sorted(path for path in runs.iterdir() if path.is_dir())
    require(len(directories) == 150, f"expected 150 run directories, found {len(directories)}")
    inventory: dict[tuple[str, str, str], dict[str, Any]] = {}
    for directory in directories:
        backend, scenario, role = parse_run_name(directory)
        require(scenario in SCENARIOS, f"unexpected scenario: {scenario}")
        key = (backend, scenario, role)
        require(key not in inventory, f"duplicate run: {key}")
        inventory[key] = audit_result(directory, backend, scenario)

    conditions: list[dict[str, Any]] = []
    by_cell: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    route_reference: dict[str, list[tuple[Any, ...]]] = {}
    for backend in BACKENDS:
        for scenario in SCENARIOS:
            parent, plugin = inventory[backend, scenario, "parent"], inventory[backend, scenario, "plugin"]
            for field in PROTOCOL_PAIR_FIELDS:
                require(parent["protocol"].get(field) == plugin["protocol"].get(field),
                        f"protocol pairing mismatch: {backend}/{scenario}/{field}")
            left, right = parent["records"], plugin["records"]
            require([[row.get(field) for field in PAIR_FIELDS] for row in left]
                    == [[row.get(field) for field in PAIR_FIELDS] for row in right],
                    f"record pairing mismatch: {backend}/{scenario}")
            route = [(row.get("key"), row.get("image_id"), row.get("descriptor_sha256"),
                      row.get("expert"), row.get("expert_index"), row.get("decision"))
                     for row in right]
            if scenario in route_reference:
                require(route == route_reference[scenario],
                        f"backend-dependent routing: {backend}/{scenario}")
            else:
                route_reference[scenario] = route
            delta = np.asarray([float(b["correct"]) - float(a["correct"])
                                for a, b in zip(left, right, strict=True)], dtype=np.float64)
            conditions.append({
                "split": "test", "backend": backend, "scope": scenario,
                "parent_accuracy": parent["accuracy"], "plugin_accuracy": plugin["accuracy"],
                "gain_pp": float(delta.mean() * 100.0),
                "ci95_pp": bootstrap_ci(delta[:, None], f"{backend}|{scenario}"),
                "expert_counts": dict(sorted(Counter(str(row["expert"]) for row in right).items())),
            })
            by_cell[backend, scenario] = (parent, plugin)

    aggregates: list[dict[str, Any]] = []
    for backend in BACKENDS:
        ids = [str(row["image_id"]) for row in by_cell[backend, SCENARIOS[0]][0]["records"]]
        matrices: dict[str, np.ndarray] = {}
        for scenario in SCENARIOS:
            parent, plugin = by_cell[backend, scenario]
            left = {str(row["image_id"]): row for row in parent["records"]}
            right = {str(row["image_id"]): row for row in plugin["records"]}
            require(set(left) == set(ids) == set(right),
                    f"base-image inventory mismatch: {backend}/{scenario}")
            matrices[scenario] = np.asarray([
                float(right[key]["correct"]) - float(left[key]["correct"]) for key in ids
            ], dtype=np.float64)
        for scope, scenarios in (*FAMILIES.items(), ("Avg", SCENARIOS)):
            matrix = np.stack([matrices[scenario] for scenario in scenarios], axis=1)
            parent_values = [by_cell[backend, scenario][0]["accuracy"] for scenario in scenarios]
            plugin_values = [by_cell[backend, scenario][1]["accuracy"] for scenario in scenarios]
            counts: Counter[str] = Counter()
            for scenario in scenarios:
                counts.update(str(row["expert"]) for row in by_cell[backend, scenario][1]["records"])
            aggregates.append({
                "split": "test", "backend": backend, "scope": scope,
                "parent_accuracy": float(np.mean(parent_values)),
                "plugin_accuracy": float(np.mean(plugin_values)),
                "gain_pp": float(matrix.mean() * 100.0),
                "ci95_pp": bootstrap_ci(matrix, f"{backend}|{scope}"),
                "expert_counts": dict(sorted(counts.items())),
            })

    gains = [float(row["gain_pp"]) for row in conditions]
    payload = {
        "schema_version": 1,
        "status": "strictly_paired_frozen_evaluation_summary",
        "manifest_sha256": reference.get("manifest_sha256"),
        "runner_sha256": reference.get("runner_sha256"),
        "pairing_fields": list(PAIR_FIELDS),
        "development_test_overlap": 0,
        "statistics": {"bootstrap_replicates": 10000, "bootstrap_seed": 99173,
                       "cluster": "base image_id across corruptions"},
        "acceptance": {
            "core_cells": 75,
            "positive_core_cells": sum(value > 0 for value in gains),
            "preferred_gt_1_5pp_cells": sum(value > 1.5 for value in gains),
            "all_core_cells_positive": all(value > 0 for value in gains),
            "minimum_observed_gain_pp": min(gains),
            "reporting_policy": "retain every preregistered cell, including zero or negative gains",
        },
        "conditions": conditions,
        "aggregates": aggregates,
    }
    compare_rows(conditions, reference.get("conditions", []), ("backend", "scope"), "condition")
    compare_rows(aggregates, reference.get("aggregates", []), ("backend", "scope"), "aggregate")
    require(payload["acceptance"] == reference.get("acceptance"), "acceptance summary mismatch")
    atomic_json(output, payload)
    return {
        "status": "passed_raw_prediction_reproduction",
        "conditions": len(conditions), "result_files": len(directories),
        "records_checked": 300000, "backend_independent_routing": True,
        "reference_matched": True, "output": str(output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = rebuild(args.runs.resolve(), args.reference_summary.resolve(), args.output.resolve())
    except (OSError, KeyError, ValueError, ReproductionError) as error:
        print(json.dumps({"status": "failed_raw_prediction_reproduction", "error": str(error)}, indent=2))
        raise SystemExit(2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
