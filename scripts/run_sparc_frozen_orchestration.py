#!/usr/bin/env python3
"""Freeze, preflight, execute, and summarize a matched SPARC evaluation.

No final evaluation is launched unless an external lock binds the complete
orchestration manifest and every declared artifact.  Parent and plugin use the
same routed runner and identical input manifests except for the router.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np


SCENARIOS = (
    "gaussian_noise:5", "shot_noise:5", "impulse_noise:5",
    "defocus_blur:5", "glass_blur:5", "motion_blur:5", "zoom_blur:5",
    "snow:5", "frost:5", "fog:5", "brightness:5",
    "contrast:5", "elastic_transform:5", "pixelate:5", "jpeg_compression:5",
)
FAMILIES = {
    "Noise": SCENARIOS[:3], "Blur": SCENARIOS[3:7],
    "Weather": SCENARIOS[7:11], "Digital": SCENARIOS[11:],
}
SPLITS = ("test",)
BACKENDS = ("clip", "tent", "clipartt", "tda", "mint")
PAIR_FIELDS = (
    "image_id", "sample", "target", "raw_sha256", "stream_position", "batch"
)


class GateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def resolve(root: Path, raw: str, *, must_exist: bool = True) -> Path:
    candidate = Path(raw).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise GateError(f"path escapes project root: {raw}") from error
    if must_exist and not path.exists():
        raise GateError(f"declared path is missing: {path}")
    return path


def artifact(root: Path, entry: Any, label: str) -> Path:
    if not isinstance(entry, dict) or set(("path", "sha256")) - set(entry):
        raise GateError(f"{label} must contain path and sha256")
    path = resolve(root, str(entry["path"]))
    if not path.is_file() or sha256(path) != entry["sha256"]:
        raise GateError(f"{label} is missing or hash-mismatched: {path}")
    return path


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise GateError(f"ID file is empty or contains duplicates: {path}")
    return values


def sequence_sha256(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise GateError(f"JSON root must be an object: {path}")
    return value


def condition_key(value: dict[str, Any]) -> tuple[str, str, str]:
    return str(value.get("split")), str(value.get("backend")), str(value.get("scenario"))


def stripped_run_manifest(value: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(value))
    copied.pop("router", None)
    copied.pop("evidence_scope", None)
    return copied


def validate_run_pair(
    root: Path, condition: dict[str, Any], contract: dict[str, Any], artifacts: dict[str, str]
) -> dict[str, Any]:
    split, backend, scenario = condition_key(condition)
    if split not in SPLITS or backend not in BACKENDS or scenario not in SCENARIOS:
        raise GateError(f"invalid condition key: {(split, backend, scenario)}")
    parent_entry = condition.get("parent", {})
    plugin_entry = condition.get("plugin", {})
    parent_path = artifact(root, parent_entry.get("manifest"), f"{split}/{backend}/{scenario} parent manifest")
    plugin_path = artifact(root, plugin_entry.get("manifest"), f"{split}/{backend}/{scenario} plugin manifest")
    artifacts[str(parent_path)] = sha256(parent_path)
    artifacts[str(plugin_path)] = sha256(plugin_path)
    parent = load_json(parent_path)
    plugin = load_json(plugin_path)
    if stripped_run_manifest(parent) != stripped_run_manifest(plugin):
        raise GateError(f"parent/plugin run manifests differ beyond router/scope: {(split, backend, scenario)}")
    for label, value in (("parent", parent), ("plugin", plugin)):
        if value.get("status") != "frozen_for_routed_replay":
            raise GateError(f"{label} run manifest is not frozen")
        if value.get("seed") != contract["seed"]:
            raise GateError(f"{label} seed mismatch: {(split, backend, scenario)}")
        if value.get("batch_size") != contract["batch_size"]:
            raise GateError(f"{label} batch mismatch: {(split, backend, scenario)}")
        if value.get("backend", {}).get("kind") != backend:
            raise GateError(f"{label} backend mismatch: {(split, backend, scenario)}")
        backend_model_sha = value.get("backend", {}).get("clip_checkpoint", {}).get("sha256")
        if backend_model_sha != contract["model_sha256"]:
            raise GateError(f"{label} model checkpoint mismatch: {(split, backend, scenario)}")
        if value.get("backend", {}).get("clip_checkpoint") != contract["_model_checkpoint_entry"]:
            raise GateError(f"{label} model checkpoint path/hash differs from global freeze")
        reset = value.get("routing_information_contract", {}).get("mint_reset")
        expected_reset = contract["reset_policy"][backend]
        if reset != expected_reset:
            raise GateError(f"{label} reset policy mismatch: {(split, backend, scenario)}")
        evaluation = value.get("evaluation_contract", {})
        expected_evaluation = {
            "split": split,
            "scenario": scenario,
            "test_ids_sequence_sha256": contract["test_ids_sequence_sha256"][split],
            "prompt_sha256": contract["prompt_sha256"],
            "model_sha256": contract["model_sha256"],
        }
        if evaluation != expected_evaluation:
            raise GateError(f"{label} evaluation_contract mismatch: {(split, backend, scenario)}")
        index = artifact(root, value.get("input_stream", {}).get("index"), f"{label} input index")
        artifacts[str(index)] = sha256(index)
        index_payload = load_json(index)
        if index_payload.get("split") != split or index_payload.get("scenario") != scenario:
            raise GateError(f"{label} input index split/scenario mismatch")
        if index_payload.get("rows") != contract["n_per_scenario"]:
            raise GateError(f"{label} input index row count mismatch")
        index_rows = index_payload.get("records")
        if not isinstance(index_rows, list) or len(index_rows) != contract["n_per_scenario"]:
            raise GateError(f"{label} input index records are incomplete")
        image_ids = [row.get("image_id") for row in index_rows]
        if image_ids != contract["_test_ids"][split]:
            raise GateError(f"{label} input index image-ID sequence differs from frozen split")
        if len({row.get("sample") for row in index_rows}) != len(index_rows):
            raise GateError(f"{label} input index sample IDs are duplicated")
        if any(row.get("audit_domain") != scenario for row in index_rows):
            raise GateError(f"{label} input index scenario mismatch")
        router_path = artifact(root, value.get("router", {}).get("artifact"), f"{label} router")
        artifacts[str(router_path)] = sha256(router_path)
        feature_path = artifact(root, value.get("features", {}).get("artifact"), f"{label} descriptors")
        artifacts[str(feature_path)] = sha256(feature_path)
        for source_index, source_entry in enumerate(value.get("input_stream", {}).get("source_artifacts", [])):
            source_path = artifact(root, source_entry, f"{label} input source {source_index}")
            artifacts[str(source_path)] = sha256(source_path)
        for expert_index, expert in enumerate(value.get("expert_bank", [])):
            if "checkpoint" in expert:
                checkpoint = artifact(root, expert["checkpoint"], f"{label} expert checkpoint {expert_index}")
                artifacts[str(checkpoint)] = sha256(checkpoint)
    if parent["input_stream"]["index"] != plugin["input_stream"]["index"]:
        raise GateError(f"parent/plugin input index differs: {(split, backend, scenario)}")
    if parent["features"] != plugin["features"]:
        raise GateError(f"parent/plugin descriptors differ: {(split, backend, scenario)}")
    if parent["backend"] != plugin["backend"]:
        raise GateError(f"parent/plugin model/backend/prompt configuration differs")
    if parent["expert_bank"] != plugin["expert_bank"]:
        raise GateError(f"parent/plugin expert bank differs")
    parent_output = resolve(root, str(parent_entry.get("output_dir")), must_exist=False)
    plugin_output = resolve(root, str(plugin_entry.get("output_dir")), must_exist=False)
    if parent_output == plugin_output:
        raise GateError(f"parent/plugin output directories collide")
    return {
        "split": split, "backend": backend, "scenario": scenario,
        "parent_manifest": str(parent_path), "plugin_manifest": str(plugin_path),
        "parent_manifest_sha256": sha256(parent_path),
        "plugin_manifest_sha256": sha256(plugin_path),
        "parent_output": str(parent_output), "plugin_output": str(plugin_output),
        "input_index_sha256": parent["input_stream"]["index"]["sha256"],
        "descriptor_sha256": parent["features"]["artifact"]["sha256"],
        "parent_router_sha256": parent["router"]["artifact"]["sha256"],
        "plugin_router_sha256": plugin["router"]["artifact"]["sha256"],
    }


def preflight(
    manifest_path: Path, *, native_validate: bool = True
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    root = manifest_path.resolve().parents[1]
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise GateError("unsupported orchestration schema")
    if manifest.get("status") != "frozen_before_evaluation":
        return ({
            "schema_version": 1, "status": "blocked_draft_not_frozen", "ready": False,
            "manifest": str(manifest_path.resolve()), "gpu_used": False,
            "results_read": False,
            "required_scope": "15 corruptions x 5 backends x parent/plugin = 150 runs",
            "known_capability_gaps": manifest.get("known_capability_gaps", []),
        }, None)
    if manifest.get("evidence_scope") != "new_untouched_confirmation_not_used_for_selection":
        raise GateError("final orchestration requires a new untouched evidence scope")
    contract = manifest.get("protocol", {})
    if contract.get("seed") != 1109 or contract.get("batch_size") != 20:
        raise GateError("global seed/batch must be 1109/20")
    if tuple(contract.get("scenarios", ())) != SCENARIOS:
        raise GateError("orchestration requires all 15 canonical corruptions")
    if tuple(contract.get("backends", ())) != BACKENDS:
        raise GateError("orchestration requires CLIP, TENT, CLIPArTT, TDA, and Mint")
    if contract.get("reset_policy") != {
        "clip": "not_applicable",
        "tent": "once_before_this_corruption_stream",
        "clipartt": "once_before_this_corruption_stream",
        "tda": "once_before_this_corruption_stream",
        "mint": "once_before_this_corruption_stream",
    }:
        raise GateError("reset policy must be frozen and backend-specific")
    if manifest.get("acceptance") != {
        "core_scope": "all_75_backend_corruption_cells",
        "strict_minimum_gain_pp": 0.0,
        "comparison": "strictly_greater_than",
        "preferred_gain_pp": 1.5,
    }:
        raise GateError("acceptance rule must require positive gain in all 75 core cells")

    artifacts: dict[str, str] = {}
    runner = artifact(root, manifest.get("runner"), "runner")
    artifacts[str(runner)] = sha256(runner)
    capability_path = artifact(
        root, manifest.get("runner_capability_report"), "runner capability report"
    )
    capability = load_json(capability_path)
    if capability.get("status") != "validated_unified_nonlinear_routed_runner":
        raise GateError("runner capability report is not a validated unified runner")
    if capability.get("runner_sha256") != sha256(runner):
        raise GateError("runner capability report does not bind the declared runner")
    if tuple(capability.get("supported_backends", ())) != BACKENDS:
        raise GateError("runner capability report does not cover all five backends")
    if capability.get("supports_nonlinear_router") is not True:
        raise GateError("runner does not attest nonlinear-router support")
    if capability.get("same_runner_for_parent_and_plugin") is not True:
        raise GateError("runner does not attest matched parent/plugin execution")
    artifacts[str(capability_path)] = sha256(capability_path)
    prompt = artifact(root, manifest.get("prompt_contract"), "prompt contract")
    artifacts[str(prompt)] = sha256(prompt)
    if contract.get("prompt_sha256") != sha256(prompt):
        raise GateError("protocol prompt hash mismatch")
    model = artifact(root, manifest.get("model_checkpoint"), "model checkpoint")
    artifacts[str(model)] = sha256(model)
    if contract.get("model_sha256") != sha256(model):
        raise GateError("protocol model hash mismatch")
    contract["_model_checkpoint_entry"] = manifest["model_checkpoint"]
    environment = artifact(root, manifest.get("environment_lock"), "environment lock")
    artifacts[str(environment)] = sha256(environment)
    python_path = Path(str(manifest.get("execution", {}).get("python", ""))).expanduser().resolve()
    if not python_path.is_file():
        raise GateError("execution.python must be an existing absolute executable")
    artifacts[str(python_path)] = sha256(python_path)

    development_ids: set[str] = set()
    for index, entry in enumerate(manifest.get("development_id_files", [])):
        path = artifact(root, entry, f"development ID file {index}")
        values = read_ids(path)
        if development_ids & set(values):
            raise GateError("development ID files overlap each other")
        development_ids.update(values)
        artifacts[str(path)] = sha256(path)
    if not development_ids:
        raise GateError("at least one development ID file is required")
    test_entries = manifest.get("test_id_files", {})
    if set(test_entries) != set(SPLITS):
        raise GateError("test_id_files must contain exactly the frozen test split")
    test_sets = {}
    observed_sequence_hashes = {}
    for split in SPLITS:
        path = artifact(root, test_entries[split], f"test ID file {split}")
        values = read_ids(path)
        test_sets[split] = set(values)
        observed_sequence_hashes[split] = sequence_sha256(values)
        artifacts[str(path)] = sha256(path)
        if development_ids & test_sets[split]:
            raise GateError(f"development/test ID overlap detected for split {split}")
    if observed_sequence_hashes != contract.get("test_ids_sequence_sha256"):
        raise GateError("test ID sequence hash mismatch")
    if any(len(test_sets[split]) != contract.get("n_per_scenario") for split in SPLITS):
        raise GateError("test split size differs from n_per_scenario")
    contract["_test_ids"] = {
        split: read_ids(artifact(root, test_entries[split], f"test ID file {split}"))
        for split in SPLITS
    }

    conditions = manifest.get("conditions")
    expected_keys = {(split, backend, scenario) for split in SPLITS for backend in BACKENDS for scenario in SCENARIOS}
    if not isinstance(conditions, list) or {condition_key(value) for value in conditions} != expected_keys:
        raise GateError("conditions must contain the exact 75 backend/scenario cells")
    if len(conditions) != 75:
        raise GateError("conditions contain duplicates")
    audited = [validate_run_pair(root, value, contract, artifacts) for value in conditions]

    native_validations = 0
    if native_validate:
        python = str(manifest["execution"]["python"])
        for condition in audited:
            for side in ("parent", "plugin"):
                command = [
                    python, str(runner), "--manifest", condition[f"{side}_manifest"],
                    "--output-dir", condition[f"{side}_output"], "--validate-only",
                ]
                completed = subprocess.run(
                    command, cwd=root, text=True, capture_output=True, check=False
                )
                if completed.returncode:
                    raise GateError(
                        f"native runner validation failed for {condition_key(condition)}/{side}: "
                        f"{completed.stderr.strip() or completed.stdout.strip()}"
                    )
                native_validations += 1

    parent_hashes = {row["parent_router_sha256"] for row in audited}
    plugin_hashes = {row["plugin_router_sha256"] for row in audited}
    if len(parent_hashes) != 1 or len(plugin_hashes) != 1 or parent_hashes == plugin_hashes:
        raise GateError("all cells require one shared parent router and one distinct shared plugin router")
    control_report = artifact(root, manifest.get("identity_control_report"), "identity control report")
    control = load_json(control_report)
    if control.get("status") != "frozen_identity_control_router_all_abstain":
        raise GateError("identity control report status mismatch")
    if control.get("artifact_sha256") != next(iter(parent_hashes)):
        raise GateError("identity control report does not bind parent router")
    artifacts[str(control_report)] = sha256(control_report)
    report = {
        "schema_version": 1, "status": "ready_frozen_preflight", "ready": True,
        "manifest": str(manifest_path.resolve()), "manifest_sha256": sha256(manifest_path),
        "runner": str(runner), "runner_sha256": sha256(runner),
        "conditions": audited, "condition_count": 75,
        "development_id_count": len(development_ids),
        "test_id_counts": {key: len(value) for key, value in test_sets.items()},
        "development_test_overlap": 0,
        "prompt_sha256": sha256(prompt), "model_sha256": sha256(model),
        "environment_lock_sha256": sha256(environment),
        "native_run_manifests_validated": native_validations,
        "artifact_hashes": dict(sorted(artifacts.items())),
        "gpu_used": False, "results_read": False,
    }
    return report, manifest


def freeze_lock(manifest_path: Path, report: dict[str, Any], lock_path: Path) -> None:
    for condition in report["conditions"]:
        for side in ("parent_output", "plugin_output"):
            if (Path(condition[side]) / "result.json").exists():
                raise GateError("refusing to freeze after an evaluation result already exists")
    payload = {
        "schema_version": 1, "status": "frozen_orchestration_lock",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": report["manifest_sha256"],
        "artifact_hashes": report["artifact_hashes"],
        "frozen_before_any_result": True,
    }
    if lock_path.exists():
        if load_json(lock_path) != payload:
            raise GateError("existing orchestration lock differs")
    else:
        atomic_json(lock_path, payload)


def validate_lock(lock_path: Path, report: dict[str, Any]) -> None:
    lock = load_json(lock_path)
    if lock.get("status") != "frozen_orchestration_lock":
        raise GateError("invalid orchestration lock status")
    if lock.get("manifest_sha256") != report["manifest_sha256"]:
        raise GateError("manifest changed after freeze")
    if lock.get("artifact_hashes") != report["artifact_hashes"]:
        raise GateError("artifact set changed after freeze")
    for raw, expected in lock["artifact_hashes"].items():
        path = Path(raw)
        if not path.is_file() or sha256(path) != expected:
            raise GateError(f"artifact changed after freeze: {path}")


def output_records(path: Path, expected: dict[str, Any], side: str) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    result_path = path / "result.json"
    protocol_path = path / "protocol.json"
    complete_path = path / "complete.json"
    if not all(value.is_file() for value in (result_path, protocol_path, complete_path)):
        raise GateError(f"incomplete {side} output: {path}")
    result, protocol, complete = load_json(result_path), load_json(protocol_path), load_json(complete_path)
    if result.get("protocol") != protocol or result.get("protocol_sha256") != sha256(protocol_path):
        raise GateError(f"{side} embedded protocol mismatch")
    if complete.get("result_sha256") != sha256(result_path):
        raise GateError(f"{side} result completion hash mismatch")
    if protocol.get("runner_sha256") != expected["runner_sha256"]:
        raise GateError(f"{side} used a different runner")
    for field in ("seed", "batch_size", "backend"):
        if protocol.get(field) != expected[field]:
            raise GateError(f"{side} output {field} mismatch")
    for field in (
        "manifest_sha256", "router_sha256", "model_sha256", "prompt_sha256",
        "test_ids_sequence_sha256", "reset_policy",
    ):
        if protocol.get(field) != expected[field]:
            raise GateError(f"{side} output {field} mismatch")
    rows = result.get("records")
    if not isinstance(rows, list) or len(rows) != expected["n"]:
        raise GateError(f"{side} output record count mismatch")
    for position, row in enumerate(rows):
        if any(field not in row for field in PAIR_FIELDS):
            raise GateError(f"{side} output missing a pairing field at {position}")
        if row.get("stream_position") != position or row.get("batch") != position // expected["batch_size"]:
            raise GateError(f"{side} output order/batch mismatch at {position}")
        if row.get("audit_domain") != expected["scenario"]:
            raise GateError(f"{side} output scenario mismatch")
        if not isinstance(row.get("correct"), bool):
            raise GateError(f"{side} output lacks boolean correctness")
    if result.get("records_sha256") != canonical_sha256(rows):
        raise GateError(f"{side} records hash mismatch")
    accuracy = 100.0 * sum(row["correct"] for row in rows) / len(rows)
    if abs(float(result.get("accuracy", -1)) - accuracy) > 1e-12:
        raise GateError(f"{side} accuracy is not reconstructible")
    return result, tuple(rows)


def audit_outputs(report: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = []
    for condition in report["conditions"]:
        common = {
            "runner_sha256": report["runner_sha256"],
            "seed": protocol["seed"], "batch_size": protocol["batch_size"],
            "backend": condition["backend"], "scenario": condition["scenario"],
            "n": protocol["n_per_scenario"],
            "model_sha256": protocol["model_sha256"],
            "prompt_sha256": protocol["prompt_sha256"],
            "test_ids_sequence_sha256": protocol["test_ids_sequence_sha256"][condition["split"]],
            "reset_policy": protocol["reset_policy"][condition["backend"]],
        }
        parent_expected = {
            **common, "manifest_sha256": condition["parent_manifest_sha256"],
            "router_sha256": condition["parent_router_sha256"],
        }
        plugin_expected = {
            **common, "manifest_sha256": condition["plugin_manifest_sha256"],
            "router_sha256": condition["plugin_router_sha256"],
        }
        parent, parent_rows = output_records(
            Path(condition["parent_output"]), parent_expected, "parent"
        )
        plugin, plugin_rows = output_records(
            Path(condition["plugin_output"]), plugin_expected, "plugin"
        )
        if [tuple(row.get(field) for field in PAIR_FIELDS) for row in parent_rows] != [
            tuple(row.get(field) for field in PAIR_FIELDS) for row in plugin_rows
        ]:
            raise GateError(f"parent/plugin records are not exactly paired: {condition_key(condition)}")
        if any(row.get("expert") != "identity" or row.get("decision") != "abstain" for row in parent_rows):
            raise GateError(f"identity-control parent did not abstain everywhere: {condition_key(condition)}")
        outputs.append({**condition, "parent": parent, "plugin": plugin,
                        "parent_records": parent_rows, "plugin_records": plugin_rows})
    return outputs


def bootstrap_ci(matrix: np.ndarray, label: str, replicates: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed ^ int(hashlib.sha256(label.encode()).hexdigest()[:8], 16))
    values = np.empty(replicates)
    for index in range(replicates):
        sample = rng.integers(0, len(matrix), size=len(matrix))
        values[index] = matrix[sample].mean() * 100.0
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def make_summary(outputs: list[dict[str, Any]], statistics: dict[str, Any]) -> dict[str, Any]:
    rows = []
    aggregates = []
    for item in outputs:
        differences = np.asarray([
            float(plugin["correct"]) - float(parent["correct"])
            for parent, plugin in zip(item["parent_records"], item["plugin_records"])
        ])
        counts = Counter(str(row["expert"]) for row in item["plugin_records"])
        rows.append({
            "split": item["split"], "backend": item["backend"], "scope": item["scenario"],
            "parent_accuracy": float(item["parent"]["accuracy"]),
            "plugin_accuracy": float(item["plugin"]["accuracy"]),
            "gain_pp": float(differences.mean() * 100.0),
            "ci95_pp": bootstrap_ci(differences[:, None], "|".join(condition_key(item)),
                                     statistics["bootstrap_replicates"], statistics["bootstrap_seed"]),
            "expert_counts": dict(sorted(counts.items())),
        })
    for split in SPLITS:
        for backend in BACKENDS:
            selected_all = [item for item in outputs if item["split"] == split and item["backend"] == backend]
            base_samples = sorted({row["sample"] for row in selected_all[0]["parent_records"]})
            matrices = {}
            for item in selected_all:
                parent = {row["sample"]: row for row in item["parent_records"]}
                plugin = {row["sample"]: row for row in item["plugin_records"]}
                if set(parent) != set(base_samples) or set(plugin) != set(base_samples):
                    raise GateError(f"base-image IDs differ across corruptions: {split}/{backend}")
                matrices[item["scenario"]] = np.asarray([
                    float(plugin[sample]["correct"]) - float(parent[sample]["correct"])
                    for sample in base_samples
                ])
            for scope, scenarios in (*FAMILIES.items(), ("Avg", SCENARIOS)):
                chosen = [item for item in selected_all if item["scenario"] in scenarios]
                matrix = np.stack([matrices[scenario] for scenario in scenarios], axis=1)
                counts: Counter[str] = Counter()
                for item in chosen:
                    counts.update(str(row["expert"]) for row in item["plugin_records"])
                aggregates.append({
                    "split": split, "backend": backend, "scope": scope,
                    "parent_accuracy": float(np.mean([item["parent"]["accuracy"] for item in chosen])),
                    "plugin_accuracy": float(np.mean([item["plugin"]["accuracy"] for item in chosen])),
                    "gain_pp": float(matrix.mean() * 100.0),
                    "ci95_pp": bootstrap_ci(matrix, f"{split}|{backend}|{scope}",
                                             statistics["bootstrap_replicates"], statistics["bootstrap_seed"]),
                    "expert_counts": dict(sorted(counts.items())),
                })
    return {"conditions": rows, "aggregates": aggregates}


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "summary.json", payload)
    fields = ("split", "backend", "scope", "parent_accuracy", "plugin_accuracy",
              "gain_pp", "ci95_low_pp", "ci95_high_pp", "expert_counts")
    temporary = output_dir / "summary.csv.tmp"
    with temporary.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in payload["conditions"] + payload["aggregates"]:
            writer.writerow({**{key: row[key] for key in fields[:-3]},
                             "ci95_low_pp": row["ci95_pp"][0],
                             "ci95_high_pp": row["ci95_pp"][1],
                             "expert_counts": json.dumps(row["expert_counts"], sort_keys=True)})
    temporary.replace(output_dir / "summary.csv")
    lines = ["% Auto-generated; do not edit.", "\\begin{tabular}{lllrrr}",
             "Split & Backend & Scope & Parent & SPARC & Gain \\\\", "\\hline"]
    for row in payload["aggregates"]:
        lines.append(
            f"{row['split'].upper()} & {row['backend']} & {row['scope']} & "
            f"{row['parent_accuracy']:.2f} & {row['plugin_accuracy']:.2f} & {row['gain_pp']:+.2f} \\\\"
        )
    lines.extend(("\\end{tabular}", ""))
    (output_dir / "summary_table.tex").write_text("\n".join(lines))


def validate_execution_environment(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    expected_path = artifact(root, manifest["environment_lock"], "environment lock")
    expected = load_json(expected_path)
    python = str(manifest["execution"]["python"])
    probe = (
        "import json,platform,joblib,numpy,sklearn,torch;"
        "print(json.dumps({'python':platform.python_version(),"
        "'numpy':numpy.__version__,'sklearn':sklearn.__version__,"
        "'joblib':joblib.__version__,'torch':torch.__version__,"
        "'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
        "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},sort_keys=True))"
    )
    completed = subprocess.run(
        [python, "-c", probe], cwd=root, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise GateError(f"environment probe failed: {completed.stderr.strip()}")
    try:
        observed = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise GateError("environment probe did not emit valid JSON") from error
    if not observed.get("cuda_available"):
        raise GateError("frozen execution requires CUDA")
    if expected.get("status") != "frozen_execution_environment" or expected.get("values") != observed:
        raise GateError(f"active execution environment differs from lock: {observed}")
    return observed


def execute(report: dict[str, Any], manifest: dict[str, Any], root: Path) -> None:
    python = str(manifest["execution"]["python"])
    runner = report["runner"]
    report["active_environment"] = validate_execution_environment(manifest, root)
    log_path = resolve(root, manifest["execution"]["command_log"], must_exist=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for condition in report["conditions"]:
        for side in ("parent", "plugin"):
            output = condition[f"{side}_output"]
            command = [python, runner, "--manifest", condition[f"{side}_manifest"],
                       "--output-dir", output]
            with log_path.open("a") as log:
                log.write(json.dumps({"condition": condition_key(condition), "side": side,
                                      "command": command}) + "\n")
            completed = subprocess.run(command, cwd=root, check=False)
            if completed.returncode:
                raise GateError(f"runner failed ({completed.returncode}); stopping: {command}")
    audited = audit_outputs(report, manifest["protocol"])
    calculated = make_summary(audited, manifest["statistics"])
    core_gains = [float(row["gain_pp"]) for row in calculated["conditions"]]
    acceptance = {
        **manifest["acceptance"],
        "core_cells": len(core_gains),
        "positive_core_cells": sum(value > 0.0 for value in core_gains),
        "preferred_gt_1_5pp_cells": sum(value > 1.5 for value in core_gains),
        "all_core_cells_positive": all(value > 0.0 for value in core_gains),
        "minimum_observed_gain_pp": min(core_gains),
    }
    payload = {
        "schema_version": 1, "status": "strictly_paired_frozen_evaluation_summary",
        "manifest_sha256": report["manifest_sha256"], "runner_sha256": report["runner_sha256"],
        "pairing_fields": list(PAIR_FIELDS), "development_test_overlap": 0,
        "statistics": manifest["statistics"], "acceptance": acceptance,
        **calculated,
    }
    output_dir = resolve(root, manifest["outputs"]["summary_dir"], must_exist=False)
    write_outputs(output_dir, payload)
    if not acceptance["all_core_cells_positive"]:
        raise GateError(
            "scientific acceptance gate failed: one or more of the 75 predeclared core "
            "cells has non-positive paired gain; full results were retained"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument(
        "--run-all", action="store_true",
        help="atomically freeze if needed, then execute all 150 runs and summarize",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    root = manifest_path.parents[1]
    try:
        report, manifest = preflight(manifest_path, native_validate=True)
        if not report["ready"]:
            print(json.dumps(report, indent=2))
            raise SystemExit(2)
        lock_path = args.lock.resolve() if args.lock else manifest_path.with_suffix(".lock.json")
        if args.freeze:
            freeze_lock(manifest_path, report, lock_path)
            report["lock"] = str(lock_path)
            report["lock_sha256"] = sha256(lock_path)
        elif args.run_all and not lock_path.exists():
            freeze_lock(manifest_path, report, lock_path)
            validate_lock(lock_path, report)
            report["lock"] = str(lock_path)
            report["lock_sha256"] = sha256(lock_path)
        else:
            if not lock_path.is_file():
                raise GateError(f"missing pre-evaluation lock: {lock_path}")
            validate_lock(lock_path, report)
            report["lock"] = str(lock_path)
            report["lock_sha256"] = sha256(lock_path)
        if args.execute or args.run_all:
            assert manifest is not None
            execute(report, manifest, root)
            report["status"] = "execution_and_summary_complete"
            report["gpu_used"] = True
        print(json.dumps(report, indent=2))
    except (GateError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        failure = {"schema_version": 1, "status": "blocked_preflight", "ready": False,
                   "error": str(error), "gpu_used": False}
        print(json.dumps(failure, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
