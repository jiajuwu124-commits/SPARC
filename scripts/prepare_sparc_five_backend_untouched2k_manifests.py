#!/usr/bin/env python3
"""Build the 75 matched parent/plugin manifest pairs for frozen untouched 2K.

Only frozen input artifacts, routers, expert-bank configuration, and source
hashes are read. No model-result or accuracy artifact is accepted or inspected.
Generated manifests remain execution-blocked until the unified runner receives
native GPU parity qualification and the external orchestration lock is frozen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_sparc_five_backend_routed_replay.py"
INPUT_REPORT = ROOT / "results/sparc_untouched_replay_inputs_20260915/readiness.json"
IDENTITY_REPORT = ROOT / "results/sparc_utility_router_development_20260915/nonlinear_identity_control_v1/report.json"
OUTPUT = ROOT / "configs/sparc_untouched2k_five_backend"
READINESS_PREPARED = ROOT / "results/sparc_untouched2k_five_backend_manifest_prep/readiness.json"
READINESS_FROZEN = ROOT / "results/sparc_untouched2k_five_backend_manifest_prep/readiness.frozen.json"
MATRIX_PREPARED = ROOT / "configs/sparc_untouched2k_five_backend_matrix.prepared.json"
MATRIX_FROZEN = ROOT / "configs/sparc_untouched2k_five_backend_matrix.frozen.json"
PROMPTS = ROOT / "configs/sparc_five_backend_prompt_contract.json"
LINEAGE_REPORT = ROOT / "results/sparc_deployment_brightness_router_v1/training_test_lineage_audit.json"
LINEAGE_AUDITOR = ROOT / "scripts/audit_sparc_training_test_lineage.py"
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
RUNTIME_DEPENDENCIES_COMMON = (
    ROOT / "scripts/run_sparc_routed_replay.py", ROOT / "src/streammint/data.py",
    ROOT / "src/streammint/risk_route_fast.py", ROOT / "src/streammint/utility_router.py",
    ROOT / "src/streammint/nonlinear_utility_router.py", ROOT / "src/streammint/conditional_restorer.py",
    ROOT / "src/streammint/snow_specialist.py", ROOT / "src/streammint/brightness_specialist.py",
    ROOT / "src/streammint/deployment_brightness.py", ROOT / "src/streammint/methods.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_generated(path: Path, value: Any, *, validate_only: bool) -> None:
    if path.exists() and json.loads(path.read_text()) == value:
        return
    if validate_only:
        raise RuntimeError(f"generated artifact missing or stale: {path}")
    atomic_json(path, value)


def source_manifest(backend: str) -> dict[str, Any]:
    path = ROOT / f"configs/sparc_five_backend_bridge_{backend}_diagnostic_a_defocus_blur_s5.json"
    value = json.loads(path.read_text())
    if value["backend"]["kind"] != backend or value["backend"]["reset"] != RESET[backend]:
        raise RuntimeError(f"source backend/reset mismatch: {backend}")
    return value


def runtime_dependency_closure(backend: dict[str, Any]) -> list[dict[str, str]]:
    roots = {Path(entry["path"]).resolve().parent for entry in backend["native_source_artifacts"]
             if Path(entry["path"]).name in {"__init__.py", "clip.py", "model.py"}
             and Path(entry["path"]).resolve().parent.name == "clip"}
    if len(roots) != 1:
        raise RuntimeError("backend does not identify one exact CLIP package root")
    clip_root = next(iter(roots))
    paths = [*RUNTIME_DEPENDENCIES_COMMON, *(clip_root / name for name in (
        "__init__.py", "clip.py", "model.py", "simple_tokenizer.py", "bpe_simple_vocab_16e6.txt.gz"
    ))]
    if backend["kind"] in {"tent", "clipartt"}:
        paths.append(ROOT / "third_party/CLIPArTT/models/tent.py")
    return [artifact(path) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--bank-manifest", type=Path,
        default=ROOT / "configs/sparc_five_backend_bridge_clip_diagnostic_a_defocus_blur_s5.json",
        help="frozen routed manifest supplying the common expert bank and plugin router",
    )
    parser.add_argument("--identity-report", type=Path, default=IDENTITY_REPORT)
    parser.add_argument(
        "--capability-report", type=Path,
        help="native five-backend qualification; when valid, freeze the matrix for an external lock",
    )
    args = parser.parse_args()
    inputs = json.loads(INPUT_REPORT.read_text())
    if inputs.get("status") != "prepared_untouched_input_only_no_outcomes":
        raise RuntimeError("untouched replay inputs are not ready")
    if tuple(inputs.get("scenarios", ())) != SCENARIOS or inputs.get("n_per_scenario") != 2000:
        raise RuntimeError("untouched input scope mismatch")
    if inputs.get("prediction_outcomes_read") is not False or inputs.get("gpu_used") is not False:
        raise RuntimeError("input preparation evidence boundary is invalid")
    expected_descriptor_sources = {
        "masked_risk_extractor": ROOT / "src/streammint/risk_route_fast.py",
        "descriptor_transform": ROOT / "src/streammint/utility_router.py",
    }
    for name, source in expected_descriptor_sources.items():
        entry = inputs.get("source_artifacts", {}).get(name, {})
        if entry != {"path": str(source.resolve()), "sha256": sha256(source)}:
            raise RuntimeError(f"untouched descriptor provenance is missing/stale: {name}")
    identity_report_path = args.identity_report.resolve()
    identity = json.loads(identity_report_path.read_text())
    if identity.get("status") != "frozen_identity_control_router_all_abstain":
        raise RuntimeError("identity-control router is not frozen")
    parent_router = artifact(Path(identity["artifact"]))
    if parent_router["sha256"] != identity["artifact_sha256"]:
        raise RuntimeError("identity-control router/report hash mismatch")

    templates = {backend: source_manifest(backend) for backend in BACKENDS}
    templates["clip"]["backend"]["prompt"] = "a photo of a {}."
    bank_source = json.loads(args.bank_manifest.resolve().read_text())
    bank = bank_source["expert_bank"]
    bank_hash = canonical_sha256(bank)
    plugin_router = bank_source["router"]["artifact"]
    if len(bank) not in {12, 13} or len({entry["name"] for entry in bank}) != len(bank):
        raise RuntimeError("the frozen bank must contain 12 or 13 unique nonidentity experts")
    deployment = [entry for entry in bank if entry.get("kind") == "deployment_brightness"]
    if len(bank) == 13 and len(deployment) != 1:
        raise RuntimeError("a 13-expert bank requires exactly one deployment_brightness expert")
    if len(bank) == 12 and deployment:
        raise RuntimeError("the 12-expert preparation bank cannot contain an unfrozen deployment expert")
    if identity.get("expert_names") != [entry["name"] for entry in bank] \
            or identity.get("source_router_sha256") != plugin_router["sha256"]:
        raise RuntimeError("identity control is not derived from this exact plugin router/bank")
    model_hashes = {value["backend"]["clip_checkpoint"]["sha256"] for value in templates.values()}
    if len(model_hashes) != 1:
        raise RuntimeError("all five backends must use one exact CLIP checkpoint")
    if any(value["seed"] != 1109 or value["batch_size"] != 20
           or not 0 < value["memory_fraction"] <= 0.70 for value in templates.values()):
        raise RuntimeError("seed/batch/memory hard gate failed")

    prompt_contract = {
        "schema_version": 1, "status": "frozen_five_backend_native_prompt_contract",
        "within_backend_parent_plugin_identical": True,
        "cross_backend_prompt_controlled_comparison": False,
        "prompts": {backend: templates[backend]["backend"]["prompt"] for backend in BACKENDS},
        "prompt_sha256_by_backend": {
            backend: canonical_sha256(templates[backend]["backend"]["prompt"])
            for backend in BACKENDS
        },
        "note": "author-native prompts differ across backend families and are fixed before evaluation",
    }
    write_generated(PROMPTS, prompt_contract, validate_only=args.validate_only)
    prompt_contract_hash = sha256(PROMPTS)
    prompt_hashes = prompt_contract["prompt_sha256_by_backend"]
    ids_sequence_hash = inputs["ids"]["sequence_sha256"]
    lineage = json.loads(LINEAGE_REPORT.read_text())
    lineage_contract = lineage.get("lineage_contract", {})
    deployment_entry = deployment[0] if deployment else None
    if deployment_entry is not None and (
        lineage.get("status") != "passed_training_test_lineage_audit"
        or lineage.get("ready") is not True
        or lineage.get("auditor_sha256") != sha256(LINEAGE_AUDITOR)
        or lineage.get("training_protocol_sha256") != deployment_entry["training_protocol"]["sha256"]
        or lineage.get("untouched_test_ids_sha256") != inputs["ids"]["sha256"]
        or lineage_contract.get("untouched_test_sequence_sha256") != ids_sequence_hash
        or lineage_contract.get("development_selection_union_count") != 3000
        or lineage_contract.get("development_test_overlap") != 0
        or any(lineage_contract.get("split_overlaps", {}).get(key) != 0 for key in (
            "development__diagnostic_a", "development__diagnostic_b", "diagnostic_a__diagnostic_b"
        ))
        or lineage_contract.get("training_rows") != 2000
        or lineage_contract.get("validation_rows") != 1000
        or lineage.get("untouched_prediction_results_read") is not False
    ):
        raise RuntimeError("training/test lineage report is missing, stale, or failed")
    conditions, manifest_hashes = [], {}
    for scenario in SCENARIOS:
        input_entry = inputs["artifacts"][scenario]
        source_array = input_entry["source_array"]
        descriptor = input_entry["descriptor"]
        index = input_entry["stream_index"]
        for backend in BACKENDS:
            pair = {}
            for side, router in (("parent", parent_router), ("plugin", plugin_router)):
                value = copy.deepcopy(templates[backend])
                value["status"] = "frozen_for_routed_replay"
                value["evidence_scope"] = "new_untouched_confirmation_not_used_for_selection"
                value["router"] = {"artifact": router}
                value["expert_bank"] = copy.deepcopy(bank)
                value["runtime_dependency_closure"] = runtime_dependency_closure(value["backend"])
                value["features"] = {"kind": "precomputed_descriptors", "artifact": descriptor}
                value["input_stream"] = {
                    "index": index, "source_artifacts": [source_array], "shuffle": True,
                }
                value["routing_information_contract"]["backend_reset"] = RESET[backend]
                value["routing_information_contract"]["mint_reset"] = RESET[backend]
                value["routing_information_contract"]["one_corruption_per_stream"] = True
                value["evaluation_contract"] = {
                    "split": "test", "scenario": scenario,
                    "test_ids_sequence_sha256": ids_sequence_hash,
                    "prompt_sha256": prompt_hashes[backend],
                    "model_sha256": next(iter(model_hashes)),
                }
                value["global_matrix_contract"] = {
                    "expert_bank_sha256": bank_hash,
                    "plugin_router_sha256": plugin_router["sha256"],
                    "parent_router_sha256": parent_router["sha256"],
                    "seed": 1109, "batch_size": 20, "memory_fraction": 0.68,
                    "native_reset_policy": RESET,
                }
                value["bridge_contract"]["runner"] = artifact(RUNNER)
                value["bridge_contract"]["native_reset_policy"] = RESET[backend]
                value["freeze_provenance"] = {
                    "input_readiness": artifact(INPUT_REPORT),
                    "identity_control_report": artifact(identity_report_path),
                    "prompt_contract": artifact(PROMPTS),
                    "training_test_lineage_report": artifact(LINEAGE_REPORT),
                    "training_test_lineage_auditor": artifact(LINEAGE_AUDITOR),
                    "result_or_accuracy_artifacts_read": [],
                }
                name = scenario.replace(":", "_s")
                path = OUTPUT / f"{backend}_{name}_{side}.json"
                write_generated(path, value, validate_only=args.validate_only)
                pair[side] = {
                    "manifest": str(path.resolve()),
                    "manifest_sha256": sha256(path),
                    "output_dir": str((ROOT / "results/sparc_untouched2k_five_backend_runs" /
                                       f"{backend}_{name}_{side}").resolve()),
                }
                manifest_hashes[str(path.resolve())] = sha256(path)
            conditions.append({"split": "test", "backend": backend, "scenario": scenario,
                               "parent": pair["parent"], "plugin": pair["plugin"]})
    if len(conditions) != 75 or len(manifest_hashes) != 150:
        raise RuntimeError("matrix cardinality is not 75 pairs / 150 manifests")
    capability_entry = None
    matrix_status = "prepared_inputs_only_requires_native_gpu_qualification_and_external_lock"
    if args.capability_report is not None:
        capability_path = args.capability_report.resolve()
        capability = json.loads(capability_path.read_text())
        if capability.get("status") != "validated_unified_nonlinear_routed_runner" \
                or capability.get("orchestration_ready") is not True:
            raise RuntimeError("capability report is not orchestration-ready")
        if capability.get("runner_sha256") != sha256(RUNNER) \
                or capability.get("expert_bank_sha256") != bank_hash \
                or capability.get("plugin_router_sha256") != plugin_router["sha256"]:
            raise RuntimeError("capability report is not bound to this runner/bank/router")
        capability_entry = artifact(capability_path)
        matrix_status = "frozen_before_evaluation"
    matrix = {
        "schema_version": 1,
        "status": matrix_status,
        "evidence_scope": "new_untouched_confirmation_not_used_for_selection",
        "runner": artifact(RUNNER), "prompt_contract": artifact(PROMPTS),
        "identity_control_report": artifact(identity_report_path),
        "input_readiness": artifact(INPUT_REPORT),
        "training_test_lineage_report": artifact(LINEAGE_REPORT),
        "training_test_lineage_auditor": artifact(LINEAGE_AUDITOR),
        "protocol": {
            "seed": 1109, "batch_size": 20, "memory_fraction": 0.68,
            "n_per_scenario": 2000, "scenarios": list(SCENARIOS),
            "backends": list(BACKENDS), "reset_policy": RESET,
            "test_ids_sequence_sha256": {"test": ids_sequence_hash},
            "prompt_contract_sha256": prompt_contract_hash,
            "prompt_sha256_by_backend": prompt_hashes,
            "model_sha256": next(iter(model_hashes)),
            "expert_bank_sha256": bank_hash,
            "plugin_router_sha256": plugin_router["sha256"],
            "parent_router_sha256": parent_router["sha256"],
        },
        "conditions": conditions,
        "condition_count": 75, "run_invocations": 150, "expert_count": len(bank),
        "result_or_accuracy_artifacts_read": [], "gpu_used": False,
    }
    if capability_entry is not None:
        matrix["runner_capability_report"] = capability_entry
    matrix_path = MATRIX_FROZEN if capability_entry is not None else MATRIX_PREPARED
    readiness_path = READINESS_FROZEN if capability_entry is not None else READINESS_PREPARED
    write_generated(matrix_path, matrix, validate_only=args.validate_only)
    report = {
        "schema_version": 1, "status": "prepared_75_pairs_150_manifests_inputs_only",
        "matrix": artifact(matrix_path), "runner": artifact(RUNNER),
        "input_readiness": artifact(INPUT_REPORT), "prompt_contract": artifact(PROMPTS),
        "condition_count": 75, "manifest_count": 150,
        "manifest_sha256": manifest_hashes, "expert_bank_sha256": bank_hash,
        "plugin_router_sha256": plugin_router["sha256"],
        "parent_router_sha256": parent_router["sha256"], "expert_count": len(bank),
        "global_contract": {"seed": 1109, "batch_size": 20, "memory_fraction": 0.68,
                            "reset_policy": RESET, "ids_sequence_sha256": ids_sequence_hash},
        "cross_condition_expert_bank_exact": True,
        "cross_condition_plugin_router_exact": True,
        "prediction_outcomes_read": False, "gpu_used": False,
        "execution_authorized": capability_entry is not None,
        "remaining_gate": (
            "completed deployment-brightness protocol, 13-expert router/control, native GPU parity, and external lock"
            if len(bank) == 12 else "native GPU parity qualification plus external pre-evaluation lock"
        ),
    }
    if capability_entry is not None:
        report["status"] = "frozen_75_pairs_150_manifests_requires_external_lock"
        report["runner_capability_report"] = capability_entry
        report["remaining_gate"] = "external pre-evaluation lock"
    write_generated(readiness_path, report, validate_only=args.validate_only)
    print(json.dumps({"status": report["status"], "conditions": 75, "manifests": 150,
                      "matrix": str(matrix_path), "report": str(readiness_path),
                      "sha256": sha256(readiness_path)}, indent=2))


if __name__ == "__main__":
    main()
