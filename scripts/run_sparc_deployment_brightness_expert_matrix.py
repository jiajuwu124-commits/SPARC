#!/usr/bin/env python3
"""Run a dev15 supplemental matrix for a supervised brightness expert.

This is a strict extension shim around ``run_sparc_expert_matrix.py``.  It is
kept separate so an actively running/frozen matrix runner is never edited.
The manifest hash binds this shim, the deployment model source, the checkpoint,
and its training protocol.  The underlying runner still provides paired
identity records, deterministic ordering, and cache validation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = ROOT / "scripts/run_sparc_expert_matrix.py"
MODEL_SOURCE = ROOT / "src/streammint/deployment_brightness.py"
SEED = 1109


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_base():
    spec = importlib.util.spec_from_file_location("sparc_expert_matrix_frozen", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()
_original_checkpoint_metadata = base.validate_checkpoint_metadata
_original_build_protocol = base.build_protocol
_original_load_expert = base.load_expert
_original_apply_expert = base.apply_expert


def deployment_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in manifest["experts"] if entry.get("type") == "deployment_brightness"]


def validate_deployment_entry(entry: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    required = {
        "checkpoint_sha256", "training_protocol", "training_protocol_sha256",
        "source_sha256", "extension_runner_sha256",
    }
    missing = sorted(required - set(entry))
    if missing:
        raise ValueError(f"deployment brightness expert is missing {missing}")
    for field in required - {"training_protocol"}:
        value = str(entry[field])
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{field} must be a lowercase SHA-256")
    checkpoint_path = resolve(base_dir, entry["checkpoint"])
    protocol_path = resolve(base_dir, entry["training_protocol"])
    gates = {
        "checkpoint_sha256": (checkpoint_path, entry["checkpoint_sha256"]),
        "training_protocol_sha256": (protocol_path, entry["training_protocol_sha256"]),
        "source_sha256": (MODEL_SOURCE, entry["source_sha256"]),
        "extension_runner_sha256": (Path(__file__).resolve(), entry["extension_runner_sha256"]),
    }
    for label, (path, expected) in gates.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{label} mismatch: expected {expected}, got {actual}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    protocol = json.loads(protocol_path.read_text())
    if checkpoint.get("format") != "sparc-deployment-supervised-brightness-v1":
        raise RuntimeError("unsupported deployment brightness checkpoint")
    if int(checkpoint.get("seed", -1)) != SEED:
        raise RuntimeError("deployment checkpoint seed mismatch")
    if checkpoint.get("uses_imagenet_class_labels_during_training") is not True:
        raise RuntimeError("supervision ledger is missing from deployment checkpoint")
    if checkpoint.get("inference_requires_class_labels") is not False:
        raise RuntimeError("deployment expert inference must be label-free")
    if checkpoint.get("inference_requires_corruption_identity") is not False:
        raise RuntimeError("deployment expert inference must be corruption-ID-free")
    if protocol.get("checkpoint_sha256") != entry["checkpoint_sha256"]:
        raise RuntimeError("training protocol does not bind the checkpoint")
    if protocol.get("status") != "redesign_complete_not_final_tested":
        raise RuntimeError("training protocol has unexpected status")
    scale = float(entry.get("residual_scale", 1.0))
    if not 0.0 <= scale <= 2.0:
        raise ValueError("deployment brightness residual_scale must be in [0,2]")
    return {
        "format": checkpoint["format"],
        "residual_scale": scale,
        "uses_imagenet_class_labels_during_training": True,
        "inference_requires_labels_or_corruption_id": False,
        "training_protocol_sha256": entry["training_protocol_sha256"],
    }


def patched_checkpoint_metadata(manifest: dict[str, Any], base_dir: Path):
    ordinary = dict(manifest)
    ordinary["experts"] = [
        entry for entry in manifest["experts"] if entry.get("type") != "deployment_brightness"
    ]
    metadata = _original_checkpoint_metadata(ordinary, base_dir)
    for entry in deployment_entries(manifest):
        metadata[entry["name"]] = validate_deployment_entry(entry, base_dir)
    return metadata


def patched_build_protocol(manifest_path: Path, manifest: dict[str, Any], base_dir: Path):
    protocol, ids, cache = _original_build_protocol(manifest_path, manifest, base_dir)
    if deployment_entries(manifest):
        protocol["source_sha256"]["scripts/run_sparc_deployment_brightness_expert_matrix.py"] = sha256(Path(__file__).resolve())
        protocol["source_sha256"]["src/streammint/deployment_brightness.py"] = sha256(MODEL_SOURCE)
        protocol["supervision_scope"] = (
            "deployment brightness was trained with declared redesign labels; "
            "dev15 matrix is router-training evidence, not confirmation evidence"
        )
    return protocol, ids, cache


def patched_load_expert(entry: dict[str, Any], base_dir: Path, device: str):
    if entry.get("type") != "deployment_brightness":
        return _original_load_expert(entry, base_dir, device)
    from streammint.deployment_brightness import DeploymentBrightnessRestorer

    validate_deployment_entry(entry, base_dir)
    checkpoint = torch.load(resolve(base_dir, entry["checkpoint"]), map_location="cpu", weights_only=False)
    model = DeploymentBrightnessRestorer(**checkpoint["model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, {}


def patched_apply_expert(images, entry, model, metadata, scenario):
    if entry.get("type") != "deployment_brightness":
        return _original_apply_expert(images, entry, model, metadata, scenario)
    with torch.no_grad(), torch.autocast(
        device_type=images.device.type, enabled=images.is_cuda
    ):
        restored = model(images)
        scale = float(entry.get("residual_scale", 1.0))
        return (images + scale * (restored - images)).clamp(0.0, 1.0)


def install_extension() -> None:
    base.SUPPORTED_TYPES = set(base.SUPPORTED_TYPES) | {"deployment_brightness"}
    base.validate_checkpoint_metadata = patched_checkpoint_metadata
    base.build_protocol = patched_build_protocol
    base.load_expert = patched_load_expert
    base.apply_expert = patched_apply_expert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    install_extension()
    # Reuse the frozen runner's CLI after replacing argv with the exact subset
    # it accepts. No files are written in validate-only mode.
    sys.argv = [str(BASE_RUNNER), "--manifest", str(args.manifest), "--output-dir", str(args.output_dir)]
    if args.validate_only:
        sys.argv.append("--validate-only")
    base.main()


if __name__ == "__main__":
    main()
