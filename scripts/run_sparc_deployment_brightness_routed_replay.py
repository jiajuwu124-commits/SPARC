#!/usr/bin/env python3
"""Strict routed-replay extension for a supervised brightness expert.

The frozen base runner remains byte-for-byte untouched.  This shim adds one
explicit expert kind and verifies its checkpoint, training ledger, and both
source files before delegating the replay loop to the base implementation.
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = ROOT / "scripts/run_sparc_routed_replay.py"
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
    spec = importlib.util.spec_from_file_location("sparc_routed_replay_frozen", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base()
_original_validate_manifest = base.validate_manifest
_OriginalRealExperts = base.RealExperts
_original_apply_mock = base.apply_mock_experts


def validate_deployment_entry(entry: dict[str, Any], manifest_base: Path) -> None:
    import torch

    for field in (
        "checkpoint", "training_protocol", "source_sha256", "checkpoint_source_sha256",
        "extension_runner_sha256",
    ):
        if field not in entry:
            raise ValueError(f"deployment brightness expert requires {field}")
    for field in (
        "source_sha256", "checkpoint_source_sha256", "extension_runner_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry[field])):
            raise ValueError(f"{field} must be a lowercase SHA-256")
    checkpoint_path = base.verify_artifact(
        manifest_base, entry["checkpoint"], f"{entry['name']} checkpoint"
    )
    protocol_path = base.verify_artifact(
        manifest_base, entry["training_protocol"], f"{entry['name']} training protocol"
    )
    if sha256(MODEL_SOURCE) != entry["source_sha256"]:
        raise RuntimeError("deployment brightness model source hash mismatch")
    if sha256(Path(__file__).resolve()) != entry["extension_runner_sha256"]:
        raise RuntimeError("deployment routed extension source hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    protocol = json.loads(protocol_path.read_text())
    if checkpoint.get("format") != "sparc-deployment-supervised-brightness-v1":
        raise RuntimeError("unsupported deployment brightness checkpoint")
    if int(checkpoint.get("seed", -1)) != SEED:
        raise RuntimeError("deployment brightness checkpoint seed mismatch")
    if checkpoint.get("uses_imagenet_class_labels_during_training") is not True:
        raise RuntimeError("deployment brightness supervision ledger is missing")
    if checkpoint.get("inference_requires_class_labels") is not False:
        raise RuntimeError("deployment brightness inference must be label-free")
    if checkpoint.get("inference_requires_corruption_identity") is not False:
        raise RuntimeError("deployment brightness inference must be corruption-ID-free")
    if protocol.get("checkpoint_sha256") != entry["checkpoint"]["sha256"]:
        raise RuntimeError("training protocol/checkpoint hash mismatch")
    if protocol.get("status") != "redesign_complete_not_final_tested":
        raise RuntimeError("deployment training protocol status mismatch")
    sources = protocol.get("source_sha256", {})
    if sources.get("src/streammint/deployment_brightness.py") != entry["checkpoint_source_sha256"]:
        raise RuntimeError("training protocol does not bind the declared model source")
    if entry["checkpoint_source_sha256"] != entry["source_sha256"]:
        raise RuntimeError("checkpoint-time and replay-time deployment source differ")
    scale = float(entry.get("residual_scale", 1.0))
    if not np.isfinite(scale) or not 0.0 <= scale <= 2.0:
        raise ValueError("deployment brightness residual_scale must be in [0,2]")


def patched_validate_manifest(path: Path, *, mock: bool):
    manifest = _original_validate_manifest(path, mock=mock)
    manifest_base = path.resolve().parent
    for entry in manifest["expert_bank"]:
        if entry.get("kind") == "deployment_brightness":
            validate_deployment_entry(entry, manifest_base)
    return manifest


class RealExperts:
    """Base real-expert bank plus the separately audited deployment model."""

    def __init__(self, manifest: dict[str, Any], device: Any) -> None:
        import torch
        from streammint.brightness_specialist import BrightnessClippingSpecialist
        from streammint.conditional_restorer import ConditionalResidualRestorer
        from streammint.deployment_brightness import DeploymentBrightnessRestorer
        from streammint.snow_specialist import SnowSpecialist

        self.device = device
        self.specifications = manifest["expert_bank"]
        self.models: dict[int, Any] = {}
        manifest_base = Path(manifest["_path"]).parent
        for index, entry in enumerate(self.specifications):
            if entry["kind"] == "fixed":
                continue
            checkpoint_path = base.verify_artifact(
                manifest_base, entry["checkpoint"], f"{entry['name']} checkpoint"
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            kind = entry["kind"]
            if kind == "conditional_restorer":
                model = ConditionalResidualRestorer(**checkpoint["model"])
                if "condition_index" not in entry:
                    raise ValueError("conditional expert requires condition_index")
            elif kind == "snow_specialist":
                if checkpoint.get("format") != "sparc-snow-specialist-v1":
                    raise RuntimeError("unsupported snow specialist checkpoint")
                model = SnowSpecialist(**checkpoint["model"])
            elif kind == "brightness_specialist":
                if checkpoint.get("format") != "sparc-brightness-clipping-specialist-v1":
                    raise RuntimeError("unsupported brightness specialist checkpoint")
                if checkpoint.get("uses_imagenet_class_labels") is not False:
                    raise RuntimeError("brightness specialist supervision ledger is missing")
                model = BrightnessClippingSpecialist(**checkpoint["model"])
            elif kind == "deployment_brightness":
                validate_deployment_entry(entry, manifest_base)
                model = DeploymentBrightnessRestorer(**checkpoint["model"])
            else:  # pragma: no cover - guarded by validation
                raise AssertionError(kind)
            model.load_state_dict(checkpoint["state_dict"])
            self.models[index] = model.to(device).eval()

    def apply(self, images: list[np.ndarray], decisions: np.ndarray):
        import torch
        from PIL import Image
        from streammint.data import preprocess_image

        cpu = []
        for image, decision in zip(images, decisions):
            if decision >= 0 and self.specifications[int(decision)]["kind"] == "fixed":
                image = np.asarray(preprocess_image(
                    Image.fromarray(image), self.specifications[int(decision)]["operator"]
                ))
            cpu.append(torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255)
        output = torch.stack(cpu).to(self.device, non_blocking=True)
        for index, model in self.models.items():
            rows = np.flatnonzero(decisions == index)
            if not len(rows):
                continue
            positions = torch.as_tensor(rows, device=self.device)
            subset = output.index_select(0, positions)
            entry = self.specifications[index]
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type, enabled=self.device.type == "cuda"
            ):
                if entry["kind"] == "conditional_restorer":
                    condition = torch.full(
                        (len(rows),), int(entry["condition_index"]),
                        device=self.device, dtype=torch.long,
                    )
                    restored = model(subset, condition)
                    scale = float(entry.get("residual_scale", 1.0))
                    restored = (subset + scale * (restored - subset)).clamp(0, 1)
                elif entry["kind"] in {"brightness_specialist", "deployment_brightness"}:
                    restored = model(subset)
                    scale = float(entry.get("residual_scale", 1.0))
                    restored = (subset + scale * (restored - subset)).clamp(0, 1)
                else:
                    restored = model(subset)
            output.index_copy_(0, positions, restored)
        return output


def patched_apply_mock(images, decisions, expert_bank):
    translated = []
    for entry in expert_bank:
        if entry.get("kind") == "deployment_brightness":
            translated.append({**entry, "kind": "brightness_specialist"})
        else:
            translated.append(entry)
    return _original_apply_mock(images, decisions, translated)


def install_extension() -> None:
    base.ALLOWED_EXPERT_KINDS = set(base.ALLOWED_EXPERT_KINDS) | {"deployment_brightness"}
    base.validate_manifest = patched_validate_manifest
    base.RealExperts = RealExperts
    base.apply_mock_experts = patched_apply_mock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    install_extension()
    sys.argv = [str(BASE_RUNNER), "--manifest", str(args.manifest), "--output-dir", str(args.output_dir)]
    if args.mock:
        sys.argv.append("--mock")
    if args.validate_only:
        sys.argv.append("--validate-only")
    base.main()


if __name__ == "__main__":
    main()
