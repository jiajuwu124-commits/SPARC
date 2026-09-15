#!/usr/bin/env python3
"""Strict one-click executor for the matched five-backend SPARC matrix.

Prepared input-only matrices may be audited but never executed. A future
frozen matrix requires a ready capability report and external lock. Across
restarts, valid completed conditions are hash-audited and skipped, valid
progress directories are resumed by the unified runner, and malformed output
directories fail closed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("clip", "tent", "clipartt", "tda", "mint")
RESET = {
    "clip": "not_applicable", "tent": "every_batch_episodic",
    "clipartt": "every_batch_episodic", "tda": "once_before_stream",
    "mint": "once_before_stream",
}
SCENARIOS = (
    "gaussian_noise:5", "shot_noise:5", "impulse_noise:5",
    "defocus_blur:5", "glass_blur:5", "motion_blur:5", "zoom_blur:5",
    "snow:5", "frost:5", "fog:5", "brightness:5",
    "contrast:5", "elastic_transform:5", "pixelate:5", "jpeg_compression:5",
)
PAIR_FIELDS = ("image_id", "sample", "target", "raw_sha256", "descriptor_sha256",
               "stream_position", "batch")
FAMILIES = {
    "Noise": SCENARIOS[:3], "Blur": SCENARIOS[3:7],
    "Weather": SCENARIOS[7:11], "Digital": SCENARIOS[11:],
}
SUMMARY_DIR = ROOT / "results/sparc_untouched2k_five_backend_summary_v1"
TABLE_DIR = ROOT / "results/sparc_untouched2k_five_backend_table1_v1"
TABLE_SCRIPT = ROOT / "scripts/generate_sparc_full15_table1.py"
UNIFIED_RUNNER = ROOT / "scripts/run_sparc_five_backend_routed_replay.py"
ENVIRONMENT_PROBE_VERSION = 2


class GateError(RuntimeError):
    pass


_LEDGER_CACHE: dict[tuple[str, str, bool], tuple[list[dict[str, Any]], str, str]] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise GateError(f"JSON root is not an object: {path}")
    return value


def verified(entry: dict[str, Any], label: str) -> Path:
    path = Path(str(entry.get("path", ""))).resolve()
    if not path.is_file() or sha256(path) != entry.get("sha256"):
        raise GateError(f"{label} is missing or hash-mismatched: {path}")
    return path


def strip_pair_difference(value: dict[str, Any]) -> dict[str, Any]:
    answer = json.loads(json.dumps(value))
    answer.pop("router", None)
    answer.pop("evidence_scope", None)
    return answer


def audit_matrix(path: Path) -> dict[str, Any]:
    matrix = load(path)
    if matrix.get("status") not in {
        "prepared_inputs_only_requires_native_gpu_qualification_and_external_lock",
        "frozen_before_evaluation",
    }:
        raise GateError("matrix is neither prepared nor frozen")
    protocol = matrix.get("protocol", {})
    expected = {
        "seed": 1109, "batch_size": 20, "memory_fraction": 0.68,
        "n_per_scenario": 2000, "scenarios": list(SCENARIOS),
        "backends": list(BACKENDS), "reset_policy": RESET,
    }
    if any(protocol.get(name) != value for name, value in expected.items()):
        raise GateError("global seed/batch/memory/scope/reset contract mismatch")
    if not 0 < float(protocol["memory_fraction"]) <= 0.70:
        raise GateError("memory fraction exceeds 70%")
    runner = verified(matrix["runner"], "unified runner")
    if runner != UNIFIED_RUNNER.resolve():
        raise GateError("matrix does not use the current unified runner entry point")
    verified(matrix["prompt_contract"], "prompt contract")
    verified(matrix["identity_control_report"], "identity control report")
    input_readiness_path = verified(matrix["input_readiness"], "untouched input readiness")
    input_readiness = load(input_readiness_path)
    if input_readiness.get("status") != "prepared_untouched_input_only_no_outcomes" \
            or input_readiness.get("descriptor_uses_labels") is not False \
            or input_readiness.get("recognition_model_loaded") is not False \
            or input_readiness.get("prediction_outcomes_read") is not False \
            or set(input_readiness.get("artifacts", {})) != set(SCENARIOS):
        raise GateError("untouched descriptor readiness evidence boundary failed")
    lineage_path = verified(matrix["training_test_lineage_report"], "training/test lineage report")
    lineage_auditor = verified(matrix["training_test_lineage_auditor"], "training/test lineage auditor")
    lineage = load(lineage_path)
    contract = lineage.get("lineage_contract", {})
    if lineage.get("status") != "passed_training_test_lineage_audit" \
            or lineage.get("ready") is not True \
            or lineage.get("auditor_sha256") != sha256(lineage_auditor) \
            or lineage.get("training_protocol_sha256") is None \
            or contract.get("development_selection_union_count") != 3000 \
            or contract.get("development_test_overlap") != 0 \
            or contract.get("untouched_test_sequence_sha256") != protocol["test_ids_sequence_sha256"]["test"] \
            or lineage.get("untouched_test_ids_sha256") != input_readiness.get("ids", {}).get("sha256") \
            or contract.get("training_rows") != 2000 or contract.get("validation_rows") != 1000 \
            or any(contract.get("split_overlaps", {}).get(key) != 0 for key in (
                "development__diagnostic_a", "development__diagnostic_b", "diagnostic_a__diagnostic_b"
            )):
        raise GateError("training/test lineage hard gate failed")
    conditions = matrix.get("conditions", [])
    expected_keys = {(backend, scenario) for scenario in SCENARIOS for backend in BACKENDS}
    observed_keys = {(row.get("backend"), row.get("scenario")) for row in conditions}
    if len(conditions) != 75 or observed_keys != expected_keys:
        raise GateError("matrix does not contain the exact 75 backend/scenario cells")
    bank_hashes, plugin_hashes, parent_hashes = set(), set(), set()
    deployment_protocol_hashes: set[str] = set()
    output_dirs: set[Path] = set()
    artifact_hash_cache: dict[Path, str] = {}

    def verify_nested(entry: dict[str, Any], label: str) -> Path:
        nested = Path(str(entry.get("path", ""))).resolve()
        if not nested.is_file():
            raise GateError(f"{label} is missing: {nested}")
        if nested not in artifact_hash_cache:
            artifact_hash_cache[nested] = sha256(nested)
        digest = artifact_hash_cache[nested]
        if digest != entry.get("sha256"):
            raise GateError(f"{label} hash mismatch: {nested}")
        return nested
    backend_configs: dict[str, str] = {}
    audited = []
    for condition in conditions:
        backend, scenario = condition["backend"], condition["scenario"]
        pair = {}
        for side in ("parent", "plugin"):
            entry = condition[side]
            manifest_path = Path(entry["manifest"]).resolve()
            if not manifest_path.is_file() or sha256(manifest_path) != entry["manifest_sha256"]:
                raise GateError(f"{backend}/{scenario}/{side} manifest hash mismatch")
            manifest = load(manifest_path)
            output_dir = Path(entry["output_dir"]).resolve()
            if output_dir in output_dirs:
                raise GateError(f"duplicate output directory: {output_dir}")
            output_dirs.add(output_dir)
            if manifest.get("status") != "frozen_for_routed_replay":
                raise GateError("run manifest is not frozen")
            verify_nested(manifest["router"]["artifact"], "router")
            verify_nested(manifest["features"]["artifact"], "descriptor")
            if manifest["features"]["artifact"] != input_readiness["artifacts"][scenario]["descriptor"]:
                raise GateError("manifest descriptor does not match the audited scenario readiness")
            index_path = verify_nested(manifest["input_stream"]["index"], "stream index")
            index_payload = load(index_path)
            rows = index_payload.get("records")
            if not isinstance(rows, list) or len(rows) != 2000:
                raise GateError("final stream index is not exactly 2,000 rows")
            if hashlib.sha256("\n".join(str(row.get("image_id")) for row in rows).encode()).hexdigest() \
                    != protocol["test_ids_sequence_sha256"]["test"]:
                raise GateError("stream index image-ID sequence differs from matrix")
            for nested in manifest["input_stream"].get("source_artifacts", []):
                verify_nested(nested, "input source")
            for name in ("clip_checkpoint", "meta"):
                verify_nested(manifest["backend"][name], f"backend {name}")
            for nested in manifest["backend"].get("native_source_artifacts", []):
                verify_nested(nested, "backend native source")
            for nested in manifest.get("runtime_dependency_closure", []):
                verify_nested(nested, "runtime dependency")
            for expert in manifest.get("expert_bank", []):
                if "checkpoint" in expert:
                    verify_nested(expert["checkpoint"], f"expert {expert['name']} checkpoint")
                if "training_protocol" in expert:
                    verify_nested(expert["training_protocol"], f"expert {expert['name']} protocol")
            if (manifest.get("seed"), manifest.get("batch_size"), manifest.get("memory_fraction")) \
                    != (1109, 20, 0.68):
                raise GateError("run manifest seed/batch/memory mismatch")
            if manifest["backend"]["kind"] != backend \
                    or manifest["backend"]["reset"] != RESET[backend]:
                raise GateError("run manifest backend/reset mismatch")
            backend_hash = canonical_sha256(manifest["backend"])
            if backend in backend_configs and backend_configs[backend] != backend_hash:
                raise GateError(f"{backend} configuration changes across scenarios")
            backend_configs[backend] = backend_hash
            evaluation = manifest.get("evaluation_contract")
            if evaluation != {
                "split": "test", "scenario": scenario,
                "test_ids_sequence_sha256": protocol["test_ids_sequence_sha256"]["test"],
                "prompt_sha256": protocol["prompt_sha256_by_backend"][backend],
                "model_sha256": protocol["model_sha256"],
            }:
                raise GateError("run manifest evaluation contract mismatch")
            if evaluation["prompt_sha256"] != canonical_sha256(manifest["backend"].get("prompt")):
                raise GateError("evaluation prompt hash is self-reported rather than actual")
            global_contract = manifest.get("global_matrix_contract", {})
            if canonical_sha256(manifest.get("expert_bank", [])) != global_contract.get("expert_bank_sha256"):
                raise GateError("expert-bank content differs from its global hash")
            bank = manifest.get("expert_bank", [])
            if len(bank) != 13 or sum(expert.get("kind") == "deployment_brightness" for expert in bank) != 1:
                raise GateError("final bank must have 13 experts and exactly one deployment_brightness")
            deployment_protocol_hashes.add(next(
                expert["training_protocol"]["sha256"] for expert in bank
                if expert.get("kind") == "deployment_brightness"
            ))
            if manifest["router"]["artifact"]["sha256"] not in {
                global_contract.get("parent_router_sha256"), global_contract.get("plugin_router_sha256")
            }:
                raise GateError("manifest router is outside the frozen global pair")
            bank_hashes.add(global_contract.get("expert_bank_sha256"))
            plugin_hashes.add(global_contract.get("plugin_router_sha256"))
            parent_hashes.add(global_contract.get("parent_router_sha256"))
            if global_contract.get("native_reset_policy") != RESET:
                raise GateError("run manifest global reset ledger mismatch")
            pair[side] = (manifest_path, manifest)
        if strip_pair_difference(pair["parent"][1]) != strip_pair_difference(pair["plugin"][1]):
            raise GateError(f"parent/plugin differ beyond router/scope: {backend}/{scenario}")
        if pair["parent"][1]["router"]["artifact"]["sha256"] != protocol["parent_router_sha256"]:
            raise GateError("parent router differs across conditions")
        if pair["plugin"][1]["router"]["artifact"]["sha256"] != protocol["plugin_router_sha256"]:
            raise GateError("plugin router differs across conditions")
        audited.append({**condition, "parent_manifest": pair["parent"][1],
                        "plugin_manifest": pair["plugin"][1]})
    if bank_hashes != {protocol["expert_bank_sha256"]} \
            or plugin_hashes != {protocol["plugin_router_sha256"]} \
            or parent_hashes != {protocol["parent_router_sha256"]}:
        raise GateError("75-condition global bank/router hard gate failed")
    if len(output_dirs) != 150:
        raise GateError("all 150 output directories must be globally unique")
    if deployment_protocol_hashes != {lineage["training_protocol_sha256"]}:
        raise GateError("deployment checkpoint lineage differs across the final bank")
    if matrix["status"] == "frozen_before_evaluation" and matrix.get("expert_count") != 13:
        raise GateError("final frozen matrix must use the complete 13-expert bank")
    return {"matrix": matrix, "runner": runner, "conditions": audited,
            "expert_bank_sha256": next(iter(bank_hashes)),
            "plugin_router_sha256": next(iter(plugin_hashes)),
            "verified_nested_artifacts": len(artifact_hash_cache)}


def audit_output(manifest_path: Path, output: Path, *, require_parent: bool) -> dict[str, Any]:
    manifest = load(manifest_path)
    protocol_path, result_path, complete_path = (output / name for name in
                                                 ("protocol.json", "result.json", "complete.json"))
    if not all(path.is_file() for path in (protocol_path, result_path, complete_path)):
        raise GateError(f"incomplete output cannot be verified-skip: {output}")
    protocol, result, complete = load(protocol_path), load(result_path), load(complete_path)
    if result.get("protocol") != protocol or result.get("protocol_sha256") != sha256(protocol_path):
        raise GateError("result/protocol binding mismatch")
    if complete.get("result_sha256") != sha256(result_path) \
            or complete.get("protocol_sha256") != sha256(protocol_path):
        raise GateError("complete/result/protocol binding mismatch")
    evaluation = manifest["evaluation_contract"]
    declared_runner = Path(manifest["bridge_contract"]["runner"]["path"]).resolve()
    if declared_runner != UNIFIED_RUNNER.resolve():
        raise GateError("output manifest does not bind the current unified runner path")
    required_protocol = {
        "manifest_sha256": sha256(manifest_path),
        "router_sha256": manifest["router"]["artifact"]["sha256"],
        "model_sha256": evaluation["model_sha256"],
        "prompt_sha256": evaluation["prompt_sha256"],
        "test_ids_sequence_sha256": evaluation["test_ids_sequence_sha256"],
        "reset_policy": manifest["backend"]["reset"],
        "runner_sha256": sha256(UNIFIED_RUNNER),
        "seed": 1109, "batch_size": 20, "memory_fraction": 0.68,
    }
    if any(protocol.get(name) != value for name, value in required_protocol.items()):
        raise GateError("output protocol differs from frozen run manifest")
    if required_protocol["runner_sha256"] != manifest["bridge_contract"]["runner"]["sha256"]:
        raise GateError("output is not bound to the current unified runner")
    if protocol.get("status") != "frozen_five_backend_routed_replay" \
            or result.get("environment", {}).get("mode") != "cuda":
        raise GateError("mock/non-CUDA output cannot enter the final matrix")
    peak, fraction = protocol.get("observed_peak_gpu_gib"), protocol.get("observed_peak_fraction")
    if not isinstance(peak, (int, float)) or not isinstance(fraction, (int, float)) \
            or fraction > 0.68 or protocol.get("memory_gate_pass") is not True:
        raise GateError("output lacks a passing measured GPU-memory peak")
    if result.get("timing", {}).get("peak_gpu_gib") != peak:
        raise GateError("protocol/result peak-memory mismatch")
    records = result.get("records")
    if not isinstance(records, list) or len(records) != 2000:
        raise GateError("output does not contain 2,000 records")
    if len({str(row.get("image_id")) for row in records}) != 2000 \
            or len({int(row.get("sample", -1)) for row in records}) != 2000:
        raise GateError("output image_id/sample inventory is not unique")
    if any(any(name not in row for name in PAIR_FIELDS) for row in records):
        raise GateError("output lacks an exact pairing field")
    expected_rows, order_sha, keys_sha = frozen_output_ledger(manifest)
    if protocol.get("stream_order_sha256") != order_sha or protocol.get("stream_keys_sha256") != keys_sha:
        raise GateError("output protocol stream hashes differ from the frozen input order")
    protocol_sha = sha256(protocol_path)
    for index, (row, expected) in enumerate(zip(records, expected_rows, strict=True)):
        if any(row.get(name) != value for name, value in expected.items()):
            raise GateError(f"output row differs from frozen stream/descriptor ledger at {index}")
        expert_index = row.get("expert_index")
        if not isinstance(expert_index, int) or not -1 <= expert_index < len(manifest["expert_bank"]):
            raise GateError(f"output expert index is invalid at {index}")
        expected_expert = "identity" if expert_index == -1 else manifest["expert_bank"][expert_index]["name"]
        expected_decision = "abstain" if expert_index == -1 else "route"
        if row.get("expert") != expected_expert or row.get("decision") != expected_decision \
                or row.get("protocol_sha256") != protocol_sha:
            raise GateError(f"output route/protocol ledger is inconsistent at {index}")
        if bool(row.get("correct")) != (int(row.get("prediction")) == int(row.get("target"))):
            raise GateError(f"output correctness is not reconstructible at {index}")
    if result.get("records_sha256") != canonical_sha256(records) \
            or complete.get("records_sha256") != result.get("records_sha256"):
        raise GateError("output record hash mismatch")
    accuracy = 100.0 * sum(bool(row["correct"]) for row in records) / len(records)
    if abs(float(result.get("accuracy", -1)) - accuracy) > 1e-12:
        raise GateError("output accuracy is not reconstructible")
    if require_parent and any(row.get("expert_index") != -1 or row.get("expert") != "identity"
                              or row.get("decision") != "abstain"
                              for row in records):
        raise GateError("parent output did not abstain to identity on every row")
    return result


def frozen_output_ledger(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    index_entry, descriptor_entry = manifest["input_stream"]["index"], manifest["features"]["artifact"]
    cache_key = (index_entry["sha256"], descriptor_entry["sha256"],
                 bool(manifest["input_stream"].get("shuffle", True)))
    if cache_key in _LEDGER_CACHE:
        return _LEDGER_CACHE[cache_key]
    index_path, descriptor_path = Path(index_entry["path"]), Path(descriptor_entry["path"])
    if sha256(index_path) != index_entry["sha256"] or sha256(descriptor_path) != descriptor_entry["sha256"]:
        raise GateError("frozen ledger input artifact hash mismatch")
    index_payload = load(index_path)
    rows = index_payload.get("records")
    if not isinstance(rows, list) or len(rows) != 2000:
        raise GateError("frozen ledger index does not contain exactly 2,000 rows")
    with np.load(descriptor_path, allow_pickle=False) as data:
        keys = [str(value) for value in data["keys"].tolist()]
        features = np.asarray(data["features"], dtype=np.float64)
    if len(keys) != 2000 or len(set(keys)) != 2000 or features.shape != (2000, 18):
        raise GateError("frozen descriptor ledger shape/key inventory mismatch")
    feature_by_key = {key: features[position] for position, key in enumerate(keys)}
    order = (np.random.default_rng(1109).permutation(len(rows))
             if cache_key[2] else np.arange(len(rows)))
    ledger = []
    ordered_keys = []
    for position, source_position in enumerate(order):
        source = rows[int(source_position)]
        key = str(source["key"])
        if key not in feature_by_key:
            raise GateError(f"descriptor ledger lacks stream key {key}")
        ordered_keys.append(key)
        ledger.append({
            "key": key, "image_id": str(source.get("image_id", source["sample"])),
            "sample": source["sample"], "target": int(source["target"]),
            "raw_sha256": str(source["raw_sha256"]),
            "descriptor_sha256": hashlib.sha256(
                np.asarray(feature_by_key[key], dtype="<f8").tobytes()
            ).hexdigest(),
            "stream_position": position, "batch": position // 20,
            "batch_position": position % 20, "audit_domain": source.get("audit_domain"),
        })
    order_sha = hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest()
    keys_sha = hashlib.sha256("\n".join(ordered_keys).encode()).hexdigest()
    _LEDGER_CACHE[cache_key] = (ledger, order_sha, keys_sha)
    return _LEDGER_CACHE[cache_key]


def audit_protocol_only(manifest_path: Path, protocol_path: Path) -> None:
    manifest, protocol = load(manifest_path), load(protocol_path)
    evaluation, backend = manifest["evaluation_contract"], manifest["backend"]
    runner_entry = manifest.get("bridge_contract", {}).get("runner", {})
    required = {
        "status": "frozen_five_backend_routed_replay",
        "manifest_sha256": sha256(manifest_path),
        "runner_sha256": sha256(Path(runner_entry["path"]).resolve()),
        "router_sha256": manifest["router"]["artifact"]["sha256"],
        "backend": backend["kind"], "reset_policy": backend["reset"],
        "seed": 1109, "batch_size": 20, "memory_fraction": 0.68,
        "model_sha256": evaluation["model_sha256"],
        "prompt_sha256": evaluation["prompt_sha256"],
        "test_ids_sequence_sha256": evaluation["test_ids_sequence_sha256"],
    }
    if runner_entry.get("sha256") != required["runner_sha256"] \
            or any(protocol.get(name) != value for name, value in required.items()):
        raise GateError("protocol-only partial is not bound to the current frozen run")
    forbidden_final = {"observed_peak_gpu_gib", "observed_peak_fraction", "memory_gate_pass"}
    if forbidden_final.intersection(protocol):
        raise GateError("protocol-only partial unexpectedly contains finalized fields")


def output_action(manifest_path: Path, output: Path, *, require_parent: bool) -> str:
    if not output.exists():
        return "run"
    names = {path.name for path in output.iterdir()}
    final = {"protocol.json", "result.json", "complete.json"}
    if "complete.json" in names:
        if not final.issubset(names):
            raise GateError(f"malformed completed output: {output}")
        audit_output(manifest_path, output, require_parent=require_parent)
        return "verified_skip"
    if "result.json" in names:
        if {"protocol.json", "progress.json"}.issubset(names):
            return "resume"
        raise GateError(f"uncommitted result lacks resumable provenance: {output}")
    if "protocol.json" in names and "progress.json" in names:
        return "resume"
    first_batch_orphan = re.compile(
        r"(?:records\.slot[01]\.jsonl|state\.slot[01]\.pt)(?:\.partial-[0-9]+)?"
        r"|progress\.json\.partial-[0-9]+"
    )
    if "protocol.json" in names and all(
        first_batch_orphan.fullmatch(name) for name in names - {"protocol.json"}
    ):
        # Safe first-batch crash windows include protocol-only and records/state
        # slots atomically written before progress.json was committed. The
        # runner re-derives the immutable protocol and, because there is no
        # committed progress, overwrites/ignores these orphans from batch zero.
        audit_protocol_only(manifest_path, output / "protocol.json")
        return "resume"
    if names:
        raise GateError(f"unrecognized partial output: {output}")
    return "run"


def probe_environment(python: str) -> dict[str, Any]:
    code = (
        "import importlib.metadata as m,json,platform,joblib,numpy,sklearn,torch;"
        "v=lambda n:m.version(n);"
        "print(json.dumps({'probe_version':2,'python':platform.python_version(),'numpy':numpy.__version__,"
        "'sklearn':sklearn.__version__,'joblib':joblib.__version__,'torch':torch.__version__,"
        "'torchvision':v('torchvision'),'PIL':v('Pillow'),'imagecorruptions':v('imagecorruptions'),"
        "'ftfy':v('ftfy'),'regex':v('regex'),"
        "'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
        "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},sort_keys=True))"
    )
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"}
    completed = subprocess.run([python, "-c", code], cwd=ROOT, env=env,
                               capture_output=True, text=True, check=False)
    if completed.returncode:
        raise GateError(f"environment probe failed: {completed.stderr.strip()}")
    try:
        observed = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise GateError("environment probe emitted invalid JSON") from error
    if observed.get("cuda_available") is not True:
        raise GateError("frozen execution environment has no CUDA device")
    if observed.get("probe_version") != ENVIRONMENT_PROBE_VERSION:
        raise GateError("environment probe schema mismatch")
    return observed


def lock_artifact_hashes(matrix: dict[str, Any]) -> dict[str, str]:
    entries = {
        "prompt_contract": matrix["prompt_contract"],
        "identity_control_report": matrix["identity_control_report"],
        "input_readiness": matrix["input_readiness"],
        "runner_capability_report": matrix["runner_capability_report"],
        "training_test_lineage_report": matrix["training_test_lineage_report"],
        "training_test_lineage_auditor": matrix["training_test_lineage_auditor"],
    }
    return {name: entry["sha256"] for name, entry in entries.items()}


def validate_lock(lock_path: Path, matrix_path: Path, runner: Path, python: str) -> None:
    matrix = load(matrix_path)
    lock = load(lock_path)
    expected = {
        "schema_version": 1, "status": "frozen_before_any_evaluation",
        "matrix_sha256": sha256(matrix_path), "runner_sha256": sha256(runner),
        "executor_sha256": sha256(Path(__file__).resolve()),
        "table_script_sha256": sha256(TABLE_SCRIPT),
        "artifact_hashes": lock_artifact_hashes(matrix),
        "environment": probe_environment(python),
    }
    if lock != expected:
        raise GateError("external matrix lock mismatch")


def freeze_lock(audit: dict[str, Any], matrix_path: Path, lock_path: Path, python: str) -> None:
    matrix, runner = audit["matrix"], audit["runner"]
    if matrix.get("status") != "frozen_before_evaluation":
        raise GateError("cannot freeze a lock for a merely prepared matrix")
    capability_path = verified(matrix["runner_capability_report"], "runner capability")
    capability = load(capability_path)
    if capability.get("status") != "validated_unified_nonlinear_routed_runner" \
            or capability.get("orchestration_ready") is not True \
            or capability.get("runner_sha256") != sha256(runner) \
            or capability.get("supports_deployment_brightness") is not True \
            or capability.get("expert_bank_sha256") != matrix["protocol"]["expert_bank_sha256"] \
            or capability.get("plugin_router_sha256") != matrix["protocol"]["plugin_router_sha256"] \
            or capability.get("known_gaps") not in (None, []):
        raise GateError("cannot freeze without exact native runner qualification")
    forbidden = {
        "protocol.json", "progress.json", "records.slot0.jsonl", "records.slot1.jsonl",
        "state.slot0.pt", "state.slot1.pt", "result.json", "complete.json",
    }
    resolved_lock = lock_path.resolve()
    for condition in audit["conditions"]:
        for side in ("parent", "plugin"):
            output = Path(condition[side]["output_dir"])
            if resolved_lock == output.resolve() or resolved_lock.is_relative_to(output.resolve()):
                raise GateError("external lock path cannot be inside a final output directory")
            if output.exists() and forbidden.intersection(path.name for path in output.iterdir()):
                raise GateError(f"pre-evaluation output already exists: {output}")
    payload = {
        "schema_version": 1, "status": "frozen_before_any_evaluation",
        "matrix_sha256": sha256(matrix_path), "runner_sha256": sha256(runner),
        "executor_sha256": sha256(Path(__file__).resolve()),
        "table_script_sha256": sha256(TABLE_SCRIPT),
        "artifact_hashes": lock_artifact_hashes(matrix),
        "environment": probe_environment(python),
    }
    if lock_path.exists() and load(lock_path) != payload:
        raise GateError("refusing to overwrite a different external lock")
    atomic_json(lock_path, payload)


def execute(audit: dict[str, Any], matrix_path: Path, lock_path: Path, python: str) -> None:
    matrix, runner = audit["matrix"], audit["runner"]
    if matrix["status"] != "frozen_before_evaluation":
        raise GateError("prepared input-only matrix is not authorized for execution")
    capability_path = verified(matrix["runner_capability_report"], "runner capability")
    capability = load(capability_path)
    if capability.get("status") != "validated_unified_nonlinear_routed_runner" \
            or capability.get("orchestration_ready") is not True \
            or capability.get("runner_sha256") != sha256(runner) \
            or capability.get("supports_deployment_brightness") is not True \
            or capability.get("expert_bank_sha256") != matrix["protocol"]["expert_bank_sha256"] \
            or capability.get("plugin_router_sha256") != matrix["protocol"]["plugin_router_sha256"] \
            or capability.get("known_gaps") not in (None, []):
        raise GateError("runner has no final native capability qualification")
    validate_lock(lock_path, matrix_path, runner, python)
    log_path = ROOT / "results/sparc_untouched2k_five_backend_runs/command_log.json"
    log = load(log_path).get("commands", []) if log_path.exists() else []
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "4"
    for condition in audit["conditions"]:
        pair_results = {}
        for side in ("parent", "plugin"):
            manifest_path = Path(condition[side]["manifest"])
            output = Path(condition[side]["output_dir"])
            action = output_action(manifest_path, output, require_parent=side == "parent")
            command = [python, str(runner), "--manifest", str(manifest_path),
                       "--output-dir", str(output)]
            entry = {"backend": condition["backend"], "scenario": condition["scenario"],
                     "side": side, "action": action, "command": command,
                     "started_unix": time.time(),
                     "status": "verified_skip" if action == "verified_skip" else "launched"}
            log.append(entry)
            atomic_json(log_path, {"schema_version": 1, "commands": log})
            if action != "verified_skip":
                completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
                if completed.returncode:
                    entry["status"] = "failed"
                    entry["returncode"] = completed.returncode
                    entry["completed_unix"] = time.time()
                    atomic_json(log_path, {"schema_version": 1, "commands": log})
                    raise GateError(f"runner failed for {condition['backend']}/{condition['scenario']}/{side}")
                entry["status"] = "completed"
            pair_results[side] = audit_output(manifest_path, output, require_parent=side == "parent")
            entry["completed_unix"] = time.time()
            atomic_json(log_path, {"schema_version": 1, "commands": log})
        left, right = pair_results["parent"]["records"], pair_results["plugin"]["records"]
        if [[row[name] for name in PAIR_FIELDS] for row in left] != [
            [row[name] for name in PAIR_FIELDS] for row in right
        ]:
            raise GateError("completed parent/plugin pair is not row-aligned")


def bootstrap_ci(matrix: np.ndarray, label: str, replicates: int = 10000,
                 seed: int = 99173) -> list[float]:
    rng = np.random.default_rng(seed ^ int(hashlib.sha256(label.encode()).hexdigest()[:8], 16))
    values = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 250):
        stop = min(start + 250, replicates)
        selected = rng.integers(0, len(matrix), size=(stop - start, len(matrix)))
        values[start:stop] = matrix[selected].mean(axis=(1, 2)) * 100.0
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def summarize_and_generate_table(audit: dict[str, Any], matrix_path: Path, python: str) -> None:
    by_cell: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    conditions = []
    for item in audit["conditions"]:
        backend, scenario = item["backend"], item["scenario"]
        parent = audit_output(Path(item["parent"]["manifest"]), Path(item["parent"]["output_dir"]),
                              require_parent=True)
        plugin = audit_output(Path(item["plugin"]["manifest"]), Path(item["plugin"]["output_dir"]),
                              require_parent=False)
        left, right = parent["records"], plugin["records"]
        if [[row[field] for field in PAIR_FIELDS] for row in left] != [
            [row[field] for field in PAIR_FIELDS] for row in right
        ]:
            raise GateError(f"summary pairing mismatch: {backend}/{scenario}")
        delta = np.asarray([float(b["correct"]) - float(a["correct"])
                            for a, b in zip(left, right, strict=True)], dtype=np.float64)
        conditions.append({
            "split": "test", "backend": backend, "scope": scenario,
            "parent_accuracy": float(parent["accuracy"]),
            "plugin_accuracy": float(plugin["accuracy"]),
            "gain_pp": float(delta.mean() * 100.0),
            "ci95_pp": bootstrap_ci(delta[:, None], f"{backend}|{scenario}"),
            "expert_counts": dict(sorted(Counter(str(row["expert"]) for row in right).items())),
        })
        by_cell[backend, scenario] = (parent, plugin)
    aggregates = []
    for backend in BACKENDS:
        ordered_ids = [str(row["image_id"]) for row in by_cell[backend, SCENARIOS[0]][0]["records"]]
        matrices = {}
        for scenario in SCENARIOS:
            parent, plugin = by_cell[backend, scenario]
            left = {str(row["image_id"]): row for row in parent["records"]}
            right = {str(row["image_id"]): row for row in plugin["records"]}
            if set(left) != set(ordered_ids) or set(right) != set(ordered_ids):
                raise GateError(f"base-image cluster inventory mismatch: {backend}/{scenario}")
            matrices[scenario] = np.asarray([
                float(right[key]["correct"]) - float(left[key]["correct"]) for key in ordered_ids
            ], dtype=np.float64)
        for scope, scenarios in (*FAMILIES.items(), ("Avg", SCENARIOS)):
            parent_values = [float(by_cell[backend, scenario][0]["accuracy"]) for scenario in scenarios]
            plugin_values = [float(by_cell[backend, scenario][1]["accuracy"]) for scenario in scenarios]
            matrix = np.stack([matrices[scenario] for scenario in scenarios], axis=1)
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
        "schema_version": 1, "status": "strictly_paired_frozen_evaluation_summary",
        "manifest_sha256": sha256(matrix_path), "runner_sha256": sha256(audit["runner"]),
        "pairing_fields": list(PAIR_FIELDS), "development_test_overlap": 0,
        "statistics": {"bootstrap_replicates": 10000, "bootstrap_seed": 99173,
                       "cluster": "base image_id across corruptions"},
        "acceptance": {
            "core_cells": 75, "positive_core_cells": sum(value > 0 for value in gains),
            "preferred_gt_1_5pp_cells": sum(value > 1.5 for value in gains),
            "all_core_cells_positive": all(value > 0 for value in gains),
            "minimum_observed_gain_pp": min(gains),
            "reporting_policy": "retain every preregistered cell, including zero or negative gains",
        },
        "conditions": conditions, "aggregates": aggregates,
    }
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / "summary.json"
    atomic_json(summary_path, payload)
    completed = subprocess.run(
        [python, str(TABLE_SCRIPT), "--summary", str(summary_path), "--output-dir", str(TABLE_DIR)],
        cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                      "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"},
        check=False,
    )
    if completed.returncode:
        raise GateError("strict summary passed but Table 1 generation failed")
    if not payload["acceptance"]["all_core_cells_positive"]:
        atomic_json(SUMMARY_DIR / "acceptance_failure.json", {
            "schema_version": 1, "status": "failed_all_core_cells_positive",
            "summary_sha256": sha256(summary_path),
            "minimum_observed_gain_pp": payload["acceptance"]["minimum_observed_gain_pp"],
            "negative_or_zero_cells_retained": True,
        })
        raise GateError("all outputs and Table 1 were retained, but at least one core gain is non-positive")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--python", default="python")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--freeze-lock", action="store_true")
    args = parser.parse_args()
    matrix_path = args.manifest.resolve()
    audit = audit_matrix(matrix_path)
    if args.freeze_lock:
        if args.lock is None:
            raise GateError("--freeze-lock requires --lock")
        freeze_lock(audit, matrix_path, args.lock.resolve(), args.python)
    if args.run_all:
        if args.lock is None:
            raise GateError("--run-all requires an external --lock")
        if not args.lock.resolve().exists():
            freeze_lock(audit, matrix_path, args.lock.resolve(), args.python)
        execute(audit, matrix_path, args.lock.resolve(), args.python)
        summarize_and_generate_table(audit, matrix_path, args.python)
    print(json.dumps({
        "status": "matrix_execution_complete" if args.run_all else "matrix_static_audit_pass",
        "matrix_status": audit["matrix"]["status"], "conditions": len(audit["conditions"]),
        "expert_bank_sha256": audit["expert_bank_sha256"],
        "plugin_router_sha256": audit["plugin_router_sha256"],
        "gpu_used": bool(args.run_all),
        "lock_frozen": bool(args.freeze_lock),
    }, indent=2))


if __name__ == "__main__":
    main()
