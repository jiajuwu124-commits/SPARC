#!/usr/bin/env python3
"""Stage a hash-gated deployment-brightness router integration.

The program is deliberately a state machine: it creates only artifacts whose
dependencies already exist and pass SHA-256 validation.  It never opens routed
diagnostic/final *results*.  ``--validate-only`` performs the same checks and
prints the next commands without writing anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SEED = 1109
SCENARIOS = (
    "gaussian_noise:5", "shot_noise:5", "impulse_noise:5",
    "defocus_blur:5", "glass_blur:5", "motion_blur:5", "zoom_blur:5",
    "snow:5", "frost:5", "fog:5", "brightness:5", "contrast:5",
    "elastic_transform:5", "pixelate:5", "jpeg_compression:5",
)
REQUIRED_SOURCES = {
    "integrator": ROOT / "scripts/prepare_sparc_deployment_brightness_integration.py",
    "matrix_extension": ROOT / "scripts/run_sparc_deployment_brightness_expert_matrix.py",
    "routed_extension": ROOT / "scripts/run_sparc_deployment_brightness_routed_replay.py",
    "label_builder": ROOT / "scripts/build_sparc_utility_labels_multi_supplement.py",
    "base_label_builder": ROOT / "scripts/build_sparc_utility_labels.py",
    "utility_contract": ROOT / "src/streammint/utility_pipeline_contract.py",
    "extended_trainer": ROOT / "scripts/train_sparc_nonlinear_utility_router_extended.py",
    "base_trainer": ROOT / "scripts/train_sparc_nonlinear_utility_router.py",
    "model_module": ROOT / "src/streammint/nonlinear_utility_router.py",
    "fold_module": ROOT / "src/streammint/utility_router.py",
    "deployment_model": ROOT / "src/streammint/deployment_brightness.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path)}


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def verify_artifact(entry: dict[str, Any], base: Path, label: str) -> Path:
    if not isinstance(entry, dict) or not {"path", "sha256"}.issubset(entry):
        raise ValueError(f"{label} must contain path and sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(entry["sha256"])):
        raise ValueError(f"{label}.sha256 must be lowercase SHA-256")
    path = resolve(base, str(entry["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != entry["sha256"]:
        raise RuntimeError(f"{label} hash mismatch: expected {entry['sha256']}, got {actual}")
    return path


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"refusing to overwrite changed artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def freeze_spec(args: argparse.Namespace) -> None:
    if args.checkpoint is None or args.training_protocol is None or args.base_label_manifest is None:
        raise ValueError("--freeze-spec requires checkpoint, training-protocol, base-label-manifest")
    checkpoint = args.checkpoint.resolve()
    protocol = args.training_protocol.resolve()
    base_labels = args.base_label_manifest.resolve()
    spec = {
        "schema_version": 1,
        "status": "frozen_deployment_brightness_integration_inputs",
        "seed": SEED,
        "checkpoint": artifact(checkpoint),
        "training_protocol": artifact(protocol),
        "base_label_manifest": artifact(base_labels),
        "source_artifacts": {name: artifact(path) for name, path in REQUIRED_SOURCES.items()},
        "expert": {"name": args.expert_name, "residual_scale": args.residual_scale},
        "output_root": str(args.output_root.resolve()),
        "diagnostic_base_manifests": [artifact(path) for path in args.diagnostic_base_manifest],
        "final_base_manifests": [artifact(path) for path in args.final_base_manifest],
        "scope": {
            "dev15_is_router_training_evidence": True,
            "diagnostic_a_b_are_redesign_seen_and_not_confirmation_evidence": True,
            "untouched_result_files_read": [],
            "final_manifests_require_explicit_pre_result_input_manifests": True,
        },
    }
    atomic_json(args.freeze_spec.resolve(), spec)
    print(json.dumps({"spec": str(args.freeze_spec.resolve()), "sha256": sha256(args.freeze_spec.resolve())}, indent=2))


def validate_spec(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    path = path.resolve()
    spec = json.loads(path.read_text())
    if spec.get("schema_version") != 1 or spec.get("status") != "frozen_deployment_brightness_integration_inputs":
        raise ValueError("invalid integration spec")
    if int(spec.get("seed", -1)) != SEED:
        raise ValueError("integration seed must be 1109")
    resolved = {
        "checkpoint": verify_artifact(spec["checkpoint"], path.parent, "checkpoint"),
        "training_protocol": verify_artifact(spec["training_protocol"], path.parent, "training_protocol"),
        "base_label_manifest": verify_artifact(spec["base_label_manifest"], path.parent, "base_label_manifest"),
    }
    sources = spec.get("source_artifacts", {})
    if set(sources) != set(REQUIRED_SOURCES):
        raise ValueError("source_artifacts must exactly match the integration dependency set")
    for name, expected_path in REQUIRED_SOURCES.items():
        actual_path = verify_artifact(sources[name], path.parent, f"source_artifacts.{name}")
        if actual_path != expected_path.resolve():
            raise RuntimeError(f"source_artifacts.{name} path is not the project source")
        resolved[f"source:{name}"] = actual_path
    checkpoint = torch.load(resolved["checkpoint"], map_location="cpu", weights_only=False)
    protocol = json.loads(resolved["training_protocol"].read_text())
    if checkpoint.get("format") != "sparc-deployment-supervised-brightness-v1":
        raise RuntimeError("unsupported deployment checkpoint format")
    if int(checkpoint.get("seed", -1)) != SEED:
        raise RuntimeError("checkpoint seed mismatch")
    if checkpoint.get("uses_imagenet_class_labels_during_training") is not True:
        raise RuntimeError("checkpoint supervision ledger missing")
    if checkpoint.get("inference_requires_class_labels") is not False:
        raise RuntimeError("checkpoint inference is not label-free")
    if checkpoint.get("inference_requires_corruption_identity") is not False:
        raise RuntimeError("checkpoint inference is not corruption-ID-free")
    if protocol.get("status") != "redesign_complete_not_final_tested":
        raise RuntimeError("training protocol status mismatch")
    if protocol.get("checkpoint_sha256") != spec["checkpoint"]["sha256"]:
        raise RuntimeError("training protocol/checkpoint hash mismatch")
    if protocol.get("seed") != SEED:
        raise RuntimeError("training protocol seed mismatch")
    if protocol.get("source_sha256", {}).get("src/streammint/deployment_brightness.py") != sources["deployment_model"]["sha256"]:
        raise RuntimeError("training protocol does not bind the frozen deployment model source")
    for relative, expected in protocol.get("source_sha256", {}).items():
        source_path = (ROOT / relative).resolve()
        if not source_path.is_file() or sha256(source_path) != expected:
            raise RuntimeError(f"checkpoint training source is missing or changed: {relative}")
    expert = spec.get("expert", {})
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(expert.get("name", ""))):
        raise ValueError("unsafe expert name")
    scale = float(expert.get("residual_scale", -1))
    if not 0 <= scale <= 2:
        raise ValueError("residual_scale must be in [0,2]")
    for group in ("diagnostic_base_manifests", "final_base_manifests"):
        for index, entry in enumerate(spec.get(group, [])):
            resolved[f"{group}:{index}"] = verify_artifact(entry, path.parent, f"{group}[{index}]")
    return spec, resolved


def matrix_manifest(spec: dict[str, Any], resolved: dict[str, Path]) -> tuple[dict[str, Any], Path, Path]:
    labels = json.loads(resolved["base_label_manifest"].read_text())
    base_matrix = resolve(resolved["base_label_manifest"].parent, labels["matrix_dir"])
    matrix_protocol = json.loads((base_matrix / "protocol.json").read_text())
    if matrix_protocol.get("scenarios") != list(SCENARIOS) or matrix_protocol.get("n") != 1000:
        raise RuntimeError("base utility matrix is not the frozen dev15/1K protocol")
    output_root = Path(spec["output_root"]).resolve()
    manifest_path = output_root / "configs/sparc_expert_matrix.deployment_brightness.dev15.json"
    result_dir = output_root / "results/expert_matrix_deployment_brightness_dev15"
    expert = spec["expert"]
    manifest = {
        "schema_version": 1, "status": "development", "backend": "stateless_clip",
        "seed": SEED, "memory_fraction_limit": 0.68,
        "ids": matrix_protocol["ids_source"], "n": 1000,
        "offset": int(matrix_protocol.get("offset", 0)),
        "cache": matrix_protocol["cache"], "meta": matrix_protocol["meta"],
        "clip_checkpoint": matrix_protocol["clip_checkpoint"],
        "batch_size": int(matrix_protocol["batch_size"]),
        "workers": int(matrix_protocol["workers"]), "scenarios": list(SCENARIOS),
        "experts": [
            {"name": "identity", "type": "identity"},
            {
                "name": expert["name"], "type": "deployment_brightness",
                "checkpoint": str(resolved["checkpoint"]),
                "checkpoint_sha256": spec["checkpoint"]["sha256"],
                "training_protocol": str(resolved["training_protocol"]),
                "training_protocol_sha256": spec["training_protocol"]["sha256"],
                "source_sha256": spec["source_artifacts"]["deployment_model"]["sha256"],
                "extension_runner_sha256": spec["source_artifacts"]["matrix_extension"]["sha256"],
                "residual_scale": float(expert["residual_scale"]),
            },
        ],
        "integration_contract": {
            "complete_cross_product": True, "expected_jobs": 30,
            "required_output_dir": str(result_dir),
            "dev15_role": "router_training_only_not_confirmation",
            "class_labels_are_never_expert_inputs": True,
            "untouched_results_read": False,
        },
    }
    return manifest, manifest_path, result_dir


def build_label_manifest(
    spec: dict[str, Any], resolved: dict[str, Path], supplemental_dir: Path,
) -> tuple[dict[str, Any], Path] | None:
    protocol_path = supplemental_dir / "protocol.json"
    if not protocol_path.is_file():
        return None
    supplemental = json.loads(protocol_path.read_text())
    expected_matrix_path = (
        Path(spec["output_root"]).resolve()
        / "configs/sparc_expert_matrix.deployment_brightness.dev15.json"
    )
    if not expected_matrix_path.is_file():
        raise RuntimeError("supplemental protocol exists without its frozen source manifest")
    expected_names = ["identity", spec["expert"]["name"]]
    observed_names = [entry.get("name") for entry in supplemental.get("experts", [])]
    if (
        supplemental.get("status") != "development_expert_utility_matrix_not_confirmation_evidence"
        or supplemental.get("seed") != SEED
        or supplemental.get("scenarios") != list(SCENARIOS)
        or supplemental.get("n") != 1000
        or observed_names != expected_names
        or supplemental.get("complete_cross_product") is not True
    ):
        raise RuntimeError("supplemental matrix protocol schema/scope mismatch")
    if supplemental.get("manifest_sha256") != sha256(expected_matrix_path):
        raise RuntimeError("supplemental protocol/source-manifest hash mismatch")
    if supplemental.get("source_sha256", {}).get(
        "scripts/run_sparc_deployment_brightness_expert_matrix.py"
    ) != spec["source_artifacts"]["matrix_extension"]["sha256"]:
        raise RuntimeError("supplemental protocol matrix-extension hash mismatch")
    if supplemental.get("source_sha256", {}).get(
        "src/streammint/deployment_brightness.py"
    ) != spec["source_artifacts"]["deployment_model"]["sha256"]:
        raise RuntimeError("supplemental protocol deployment-source hash mismatch")
    base = json.loads(resolved["base_label_manifest"].read_text())
    base_dir = resolved["base_label_manifest"].parent
    name = spec["expert"]["name"]
    if name in base["experts"] or name in base.get("supplemental_experts", {}):
        raise RuntimeError("new expert name already exists in base label manifest")
    output_root = Path(spec["output_root"]).resolve()
    output = output_root / "results/utility_targets_dev15_with_deployment_brightness.npz"
    report = output_root / "results/utility_targets_dev15_with_deployment_brightness.manifest.json"
    manifest = {
        **base,
        # The new manifest lives under a separate output root, so inherited
        # relative paths must be frozen as absolute paths before relocation.
        "matrix_dir": str(resolve(base_dir, base["matrix_dir"])),
        "descriptor_cache": str(resolve(base_dir, base["descriptor_cache"])),
        "experts": [*base["experts"], name],
        "supplemental_experts": {
            **{
                expert_name: str(resolve(base_dir, directory))
                for expert_name, directory in base.get("supplemental_experts", {}).items()
            },
            name: str(supplemental_dir),
        },
        "output": str(output), "report": str(report),
        "expected_sha256": {
            **base["expected_sha256"],
            f"supplemental_matrix_protocol.{name}": sha256(protocol_path),
        },
        "integration_scope": {
            "new_expert_was_supervised_on_declared_redesign_splits": True,
            "dev15_utility_is_training_evidence_not_confirmation": True,
            "diagnostic_a_b_are_not_held_out_after_redesign": True,
        },
    }
    path = output_root / "configs/sparc_utility_labels_dev15_with_deployment_brightness.json"
    return manifest, path


def command(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(value)) for value in parts)


def routed_manifest(
    original_path: Path, spec: dict[str, Any], router_path: Path,
    router_sha: str, *, final: bool,
) -> tuple[dict[str, Any], Path]:
    original = json.loads(original_path.read_text())
    if original.get("status") != "frozen_for_routed_replay":
        raise RuntimeError(f"base routed manifest is not frozen: {original_path}")
    names = [entry["name"] for entry in original["expert_bank"]]
    name = spec["expert"]["name"]
    if name in names:
        raise RuntimeError("deployment brightness expert already exists in base routed manifest")
    deployment = {
        "name": name, "kind": "deployment_brightness",
        "checkpoint": spec["checkpoint"], "training_protocol": spec["training_protocol"],
        "source_sha256": spec["source_artifacts"]["deployment_model"]["sha256"],
        "checkpoint_source_sha256": spec["source_artifacts"]["deployment_model"]["sha256"],
        "extension_runner_sha256": spec["source_artifacts"]["routed_extension"]["sha256"],
        "residual_scale": float(spec["expert"]["residual_scale"]),
    }
    revised = {
        **original,
        "router": {"artifact": {"path": str(router_path), "sha256": router_sha}},
        "expert_bank": [*original["expert_bank"], deployment],
        "deployment_brightness_integration": {
            "source_manifest": artifact(original_path),
            "expert_training_uses_labels_but_inference_is_image_only": True,
            "evidence_role": "untouched_final" if final else "redesign_seen_split_debug_only",
            "must_not_be_reported_as_confirmation": not final,
            "untouched_results_read_during_manifest_generation": False,
        },
    }
    output_root = Path(spec["output_root"]).resolve()
    group = "final" if final else "diagnostic_debug"
    path = output_root / f"configs/{group}/{original_path.stem}_deployment_brightness.json"
    return revised, path


def advance(spec_path: Path, validate_only: bool) -> dict[str, Any]:
    spec, resolved = validate_spec(spec_path)
    manifest, matrix_path, supplemental_dir = matrix_manifest(spec, resolved)
    output_root = Path(spec["output_root"]).resolve()
    actions: list[dict[str, Any]] = []
    if not validate_only:
        atomic_json(matrix_path, manifest)
    actions.append({
        "stage": "dev15_supplemental_matrix",
        "ready": True, "manifest": str(matrix_path), "output_dir": str(supplemental_dir),
        "command": command([
            "python", REQUIRED_SOURCES["matrix_extension"], "--manifest", matrix_path,
            "--output-dir", supplemental_dir,
        ]),
        "validate_command": command([
            "python", REQUIRED_SOURCES["matrix_extension"], "--manifest", matrix_path,
            "--output-dir", supplemental_dir, "--validate-only",
        ]),
    })
    label_stage = build_label_manifest(spec, resolved, supplemental_dir)
    if label_stage is None:
        actions.append({"stage": "utility_label_merge", "ready": False,
                        "blocked_on": str(supplemental_dir / "protocol.json")})
        return {"valid": True, "validate_only": validate_only, "actions": actions,
                "untouched_result_files_read": []}
    label_manifest, label_path = label_stage
    if not validate_only:
        atomic_json(label_path, label_manifest)
    utility_path = Path(label_manifest["output"])
    utility_report = Path(label_manifest["report"])
    actions.append({
        "stage": "utility_label_merge", "ready": True, "manifest": str(label_path),
        "command": command(["python", REQUIRED_SOURCES["label_builder"], "--manifest", label_path]),
        "validate_command": command([
            "python", REQUIRED_SOURCES["label_builder"], "--manifest", label_path, "--validate-only"
        ]),
    })
    if not utility_path.is_file() or not utility_report.is_file():
        actions.append({"stage": "nonlinear_router_retrain", "ready": False,
                        "blocked_on": [str(utility_path), str(utility_report)]})
        return {"valid": True, "validate_only": validate_only, "actions": actions,
                "untouched_result_files_read": []}
    # The utility report is the authoritative binding for the freshly built NPZ.
    utility_report_payload = json.loads(utility_report.read_text())
    if utility_report_payload.get("dataset_sha256") != sha256(utility_path):
        raise RuntimeError("new utility report/dataset hash mismatch")
    if not label_path.is_file() or utility_report_payload.get("source_manifest_sha256") != sha256(label_path):
        raise RuntimeError("new utility report does not bind the generated label manifest")
    router_dir = output_root / "results/nonlinear_router_with_deployment_brightness"
    trainer_args = [
        "python", REQUIRED_SOURCES["extended_trainer"],
        "--dataset", utility_path, "--dataset-sha256", sha256(utility_path),
        "--dataset-report", utility_report, "--dataset-report-sha256", sha256(utility_report),
        "--base-trainer-sha256", spec["source_artifacts"]["base_trainer"]["sha256"],
        "--model-module-sha256", spec["source_artifacts"]["model_module"]["sha256"],
        "--fold-module-sha256", spec["source_artifacts"]["fold_module"]["sha256"],
        "--output-dir", router_dir,
    ]
    actions.append({"stage": "nonlinear_router_retrain", "ready": True,
                    "command": command(trainer_args),
                    "validate_command": command([*trainer_args, "--validate-only"])})
    router_report_path = router_dir / "report.json"
    if not router_report_path.is_file():
        actions.append({"stage": "routed_manifest_generation", "ready": False,
                        "blocked_on": str(router_report_path)})
        return {"valid": True, "validate_only": validate_only, "actions": actions,
                "untouched_result_files_read": []}
    router_report = json.loads(router_report_path.read_text())
    if router_report.get("status") != "development_only_extended_nonlinear_router_not_confirmation_evidence":
        raise RuntimeError("extended router report status mismatch")
    router_path = Path(router_report["selected_model_artifact"]).resolve()
    if sha256(router_path) != router_report["selected_model_sha256"]:
        raise RuntimeError("extended router artifact/report mismatch")
    if router_report.get("experts") != label_manifest["experts"]:
        raise RuntimeError("extended router expert order differs from utility labels")
    generated = []
    for group, final in (("diagnostic_base_manifests", False), ("final_base_manifests", True)):
        for index, _entry in enumerate(spec.get(group, [])):
            revised, path = routed_manifest(
                resolved[f"{group}:{index}"], spec, router_path,
                router_report["selected_model_sha256"], final=final,
            )
            if not validate_only:
                atomic_json(path, revised)
            generated.append({"path": str(path), "role": "final" if final else "debug_seen_split"})
    actions.append({"stage": "routed_manifest_generation", "ready": True,
                    "generated": generated,
                    "runner": str(REQUIRED_SOURCES["routed_extension"])})
    return {"valid": True, "validate_only": validate_only, "actions": actions,
            "untouched_result_files_read": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--freeze-spec", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-protocol", type=Path)
    parser.add_argument("--base-label-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/deployment_brightness_integration")
    parser.add_argument("--expert-name", default="deployment_brightness")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--diagnostic-base-manifest", type=Path, action="append", default=[])
    parser.add_argument("--final-base-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.freeze_spec:
        freeze_spec(args)
        return
    if args.spec is None:
        raise ValueError("--spec is required unless --freeze-spec is used")
    print(json.dumps(advance(args.spec, args.validate_only), indent=2))


if __name__ == "__main__":
    main()
