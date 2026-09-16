#!/usr/bin/env python3
"""Unified nonlinear routed replay bridge for five native CLIP backends.

The routed image stage is identical for CLIP, TENT, CLIPArTT, TDA, and Mint.
Backend construction happens only after the manifest is parsed so the bundled
CLIPArTT `clip` package cannot be silently mixed with OpenAI CLIP. CPU mock and
validate-only modes are plumbing checks and never scientific evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
from typing import Any

# Keep CPU preflight and data plumbing bounded even when the caller forgets to
# set thread limits.  Real CUDA execution still uses the explicit 68% VRAM
# allocator gate below.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_ROOT = ROOT / "third_party/CLIPArTT"
BACKENDS = ("clip", "tent", "clipartt", "tda", "mint")
RESET_POLICY = {
    "clip": "not_applicable",
    "tent": "every_batch_episodic",
    "clipartt": "every_batch_episodic",
    "tda": "once_before_stream",
    "mint": "once_before_stream",
}
PROMPT_POLICY = {
    "clip": "a photo of a {}.",
    "tent": "a photo of a {class}", "clipartt": "a photo of a {class}",
    "tda": "seven-template ensemble from streammint.methods.TEMPLATES",
    "mint": "seven-template ensemble from streammint.methods.TEMPLATES",
}
SEED, BATCH_SIZE, MEMORY_FRACTION = 1109, 20, 0.68
DEPLOYMENT_MODEL_SOURCE = ROOT / "src/streammint/deployment_brightness.py"
DEPLOYMENT_TRAINER_SOURCE = ROOT / "scripts/train_sparc_deployment_brightness.py"
RUNTIME_DEPENDENCIES_COMMON = (
    ROOT / "scripts/run_sparc_routed_replay.py",
    ROOT / "src/streammint/data.py",
    ROOT / "src/streammint/risk_route_fast.py",
    ROOT / "src/streammint/utility_router.py",
    ROOT / "src/streammint/nonlinear_utility_router.py",
    ROOT / "src/streammint/conditional_restorer.py",
    ROOT / "src/streammint/snow_specialist.py",
    ROOT / "src/streammint/brightness_specialist.py",
    ROOT / "src/streammint/deployment_brightness.py",
    ROOT / "src/streammint/methods.py",
)


def expected_runtime_dependencies(kind: str, backend: dict[str, Any]) -> set[Path]:
    roots = {Path(entry["path"]).resolve().parent for entry in backend.get("native_source_artifacts", [])
             if Path(entry["path"]).name in {"__init__.py", "clip.py", "model.py"}
             and Path(entry["path"]).resolve().parent.name == "clip"}
    if len(roots) != 1:
        raise ValueError(f"{kind} requires exactly one hash-bound CLIP package root")
    clip_root = next(iter(roots))
    expected = {path.resolve() for path in RUNTIME_DEPENDENCIES_COMMON}
    expected.update(clip_root / name for name in (
        "__init__.py", "clip.py", "model.py", "simple_tokenizer.py", "bpe_simple_vocab_16e6.txt.gz"
    ))
    if kind in {"tent", "clipartt"}:
        expected.add((AUTHOR_ROOT / "models/tent.py").resolve())
        if clip_root != (AUTHOR_ROOT / "clip").resolve():
            raise RuntimeError("vendor backend is not bound to the vendored CLIP package")
    elif clip_root == (AUTHOR_ROOT / "clip").resolve():
        raise RuntimeError("standard backend unexpectedly resolves to vendored CLIP")
    return {path.resolve() for path in expected}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rendered_json_sha256(value: Any) -> str:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(rendered.encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_records(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    with temporary.open("w") as target:
        for row in records:
            target.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def atomic_torch_save(path: Path, value: Any) -> None:
    import torch

    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def bootstrap(backend: str):
    """Import exactly one CLIP implementation before importing streammint."""
    vendor = None
    if backend in {"tent", "clipartt"}:
        sys.path.insert(0, str(AUTHOR_ROOT))
        import clip as vendor_clip  # type: ignore
        from models import tent as vendor_tent  # type: ignore
        vendor = (vendor_clip, vendor_tent)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sparc_routed_replay as common
    return common, vendor


def resolve(base: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def verify(common: Any, base: Path, entry: dict[str, Any], label: str) -> Path:
    return common.verify_artifact(base, entry, label)


def validate_deployment_brightness(
    entry: dict[str, Any], base: Path, common: Any,
) -> None:
    """Accept only a completed, source-bound, image-only deployment expert."""
    import torch

    required = (
        "checkpoint", "training_protocol", "source_sha256", "checkpoint_source_sha256",
        "training_source_sha256", "unified_runner_sha256", "inference_contract",
    )
    if any(name not in entry for name in required):
        raise ValueError(f"deployment brightness requires fields {required}")
    for name in ("source_sha256", "checkpoint_source_sha256", "training_source_sha256",
                 "unified_runner_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry[name])):
            raise ValueError(f"deployment brightness {name} is not a lowercase SHA-256")
    if entry["inference_contract"] != {
        "inputs": ["rgb_image"], "class_labels": False, "corruption_identity": False,
    }:
        raise RuntimeError("deployment brightness must declare image-only inference")
    checkpoint_path = verify(common, base, entry["checkpoint"], "deployment brightness checkpoint")
    protocol_path = verify(common, base, entry["training_protocol"], "deployment training protocol")
    if sha256(DEPLOYMENT_MODEL_SOURCE) != entry["source_sha256"]:
        raise RuntimeError("deployment brightness replay model source changed")
    if sha256(DEPLOYMENT_TRAINER_SOURCE) != entry["training_source_sha256"]:
        raise RuntimeError("deployment brightness training source changed")
    if sha256(Path(__file__).resolve()) != entry["unified_runner_sha256"]:
        raise RuntimeError("deployment brightness entry is bound to another unified runner")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    protocol = json.loads(protocol_path.read_text())
    if checkpoint.get("format") != "sparc-deployment-supervised-brightness-v1":
        raise RuntimeError("unsupported deployment brightness checkpoint format")
    if checkpoint.get("seed") != SEED or protocol.get("seed") != SEED:
        raise RuntimeError("deployment brightness seed mismatch")
    if checkpoint.get("uses_imagenet_class_labels_during_training") is not True:
        raise RuntimeError("deployment supervision ledger is missing")
    if checkpoint.get("inference_requires_class_labels") is not False \
            or checkpoint.get("inference_requires_corruption_identity") is not False:
        raise RuntimeError("deployment checkpoint is not image-only at inference")
    if protocol.get("status") != "redesign_complete_not_final_tested":
        raise RuntimeError("deployment training is not complete and frozen")
    if protocol.get("checkpoint_sha256") != entry["checkpoint"]["sha256"]:
        raise RuntimeError("deployment training protocol/checkpoint mismatch")
    sources = protocol.get("source_sha256", {})
    if sources.get("src/streammint/deployment_brightness.py") != entry["checkpoint_source_sha256"]:
        raise RuntimeError("deployment checkpoint-time model source mismatch")
    if sources.get("scripts/train_sparc_deployment_brightness.py") != entry["training_source_sha256"]:
        raise RuntimeError("deployment training source/protocol mismatch")
    if entry["checkpoint_source_sha256"] != entry["source_sha256"]:
        raise RuntimeError("deployment model source differs between train and replay")
    scale = float(entry.get("residual_scale", 1.0))
    if not np.isfinite(scale) or not 0.0 <= scale <= 2.0:
        raise ValueError("deployment brightness residual_scale must be in [0,2]")


def validate_manifest(path: Path, *, mock: bool, common: Any) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported five-backend routed schema")
    if (manifest.get("seed"), manifest.get("batch_size"), manifest.get("memory_fraction")) != (
        SEED, BATCH_SIZE, MEMORY_FRACTION,
    ):
        raise ValueError("five-backend replay requires seed1109/batch20/memory0.68")
    if not mock and manifest.get("status") != "frozen_for_routed_replay":
        raise RuntimeError("real replay requires a frozen manifest")
    backend = manifest.get("backend", {})
    kind = backend.get("kind")
    if kind not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if backend.get("reset", "not_applicable") != RESET_POLICY[kind]:
        raise ValueError(f"native reset mismatch for {kind}: expected {RESET_POLICY[kind]}")
    if backend.get("prompt") != PROMPT_POLICY[kind]:
        raise ValueError(f"native prompt mismatch for {kind}: expected {PROMPT_POLICY[kind]}")
    if kind in {"tent", "clipartt"}:
        expected = {"steps": 10, "lr": 0.001, "K": 3, "episodic": True}
        if {key: backend.get(key) for key in expected} != expected:
            raise ValueError(f"{kind} must use the frozen author-rule settings {expected}")
    if kind == "tda":
        expected_tda = {
            "positive_capacity": 3, "positive_alpha": 2.0, "positive_beta": 5.0,
            "negative_capacity": 2, "negative_alpha": 0.117, "negative_beta": 1.0,
            "entropy_bounds": [0.2, 0.5], "mask_bounds": [0.03, 1.0],
        }
        if backend.get("native_hyperparameters") != expected_tda:
            raise ValueError("TDA manifest does not declare the exact native ImageNet settings")
    if kind == "mint" and (float(backend.get("lr", -1)) != 0.015
                            or float(backend.get("prior", -1)) != 10000.0):
        raise ValueError("Mint manifest does not declare lr=0.015 and prior=10000")
    base = path.resolve().parent
    router_path = verify(common, base, manifest["router"]["artifact"], "nonlinear router")
    index_path = verify(common, base, manifest["input_stream"]["index"], "stream index")
    rows = common.load_json_records(index_path)
    verified_sources = {
        verify(common, base, entry, "input source")
        for entry in manifest["input_stream"].get("source_artifacts", [])
    }
    referenced_sources = {
        resolve(index_path.parent, str(row["image"]["path"])) for row in rows
    }
    if verified_sources != referenced_sources:
        raise RuntimeError("source_artifacts do not exactly cover stream-index sources")
    features = manifest.get("features", {})
    if features.get("kind") != "precomputed_descriptors":
        raise ValueError("final bridge requires frozen precomputed 18-D descriptors")
    verify(common, base, features["artifact"], "descriptor artifact")
    verify(common, base, backend["clip_checkpoint"], "CLIP checkpoint")
    verify(common, base, backend["meta"], "ImageNet metadata")
    experts = manifest.get("expert_bank", [])
    names = [str(entry.get("name")) for entry in experts]
    if not names or len(names) != len(set(names)) or "identity" in names:
        raise ValueError("expert bank must be a unique non-identity list")
    for entry in experts:
        kind_name = entry.get("kind")
        if kind_name not in set(common.ALLOWED_EXPERT_KINDS) | {"deployment_brightness"}:
            raise ValueError(f"unsupported expert kind {kind_name}")
        if kind_name == "fixed":
            if not isinstance(entry.get("operator"), str):
                raise ValueError("fixed expert requires operator")
        else:
            verify(common, base, entry["checkpoint"], f"{entry['name']} checkpoint")
            if kind_name == "deployment_brightness":
                validate_deployment_brightness(entry, base, common)
    native_sources = backend.get("native_source_artifacts", [])
    if kind != "clip" and not native_sources:
        raise ValueError(f"{kind} requires native source provenance")
    for entry in native_sources:
        verify(common, base, entry, f"{kind} native source")
    closure = manifest.get("runtime_dependency_closure")
    if not isinstance(closure, list):
        raise ValueError("manifest requires a complete runtime_dependency_closure")
    declared = {resolve(base, entry["path"]): entry for entry in closure}
    if set(declared) != expected_runtime_dependencies(kind, backend):
        raise RuntimeError("runtime dependency closure is incomplete or contains extras")
    for source, entry in declared.items():
        if not source.is_file() or sha256(source) != entry.get("sha256"):
            raise RuntimeError(f"runtime dependency changed: {source}")
    routing_contract = manifest.get("routing_information_contract", {})
    forbidden = set(routing_contract.get("forbidden_router_inputs", ()))
    if not {"target", "audit_domain", "corruption_name", "backend_logits"}.issubset(forbidden):
        raise ValueError("routing forbidden-input ledger is incomplete")
    if routing_contract.get("backend_reset") != RESET_POLICY[kind]:
        raise ValueError("routing contract/backend reset mismatch")
    evaluation = manifest.get("evaluation_contract")
    if evaluation is not None:
        actual_prompt_sha = canonical_sha256(backend.get("prompt"))
        actual_ids_sha = hashlib.sha256(
            "\n".join(str(row.get("image_id", row["sample"])) for row in rows).encode()
        ).hexdigest()
        if evaluation.get("prompt_sha256") != actual_prompt_sha:
            raise RuntimeError("evaluation prompt hash is not bound to the actual backend prompt")
        if evaluation.get("model_sha256") != backend["clip_checkpoint"]["sha256"]:
            raise RuntimeError("evaluation model hash is not bound to the actual checkpoint")
        if evaluation.get("test_ids_sequence_sha256") != actual_ids_sha:
            raise RuntimeError("evaluation ID-sequence hash is not bound to the actual stream rows")
    manifest["_path"] = str(path.resolve())
    manifest["_index_path"] = str(index_path)
    manifest["_rows"] = rows
    router = common.load_router(router_path)
    if tuple(router.expert_names) != tuple(names):
        raise RuntimeError("router and manifest expert order differ")
    if router.feature_mode not in {"none", "mean_std_count"}:
        raise RuntimeError("unsupported nonlinear router feature mode")
    return manifest


class MockFiveBackend:
    def __init__(self, kind: str):
        self.kind, self.seen = kind, 0

    def predict(self, images: Any) -> np.ndarray:
        values = np.rint(np.asarray(images).mean(axis=(1, 2, 3))).astype(np.int64)
        if self.kind in {"tda", "mint"}:
            values += self.seen // BATCH_SIZE
            self.seen += len(values)
        elif self.kind == "tent":
            values += 3
        elif self.kind == "clipartt":
            values += 5
        return values % 1000

    def snapshot(self) -> dict[str, Any]:
        return {"kind": self.kind, "seen": self.seen}

    def restore(self, state: dict[str, Any]) -> None:
        if set(state) != {"kind", "seen"} or state.get("kind") != self.kind \
                or int(state.get("seen", -1)) < 0:
            raise RuntimeError("invalid mock backend resume state")
        self.seen = int(state["seen"])


def apply_mock_experts(
    common: Any, images: list[np.ndarray], decisions: np.ndarray,
    expert_bank: list[dict[str, Any]],
) -> np.ndarray:
    translated = [
        ({**entry, "kind": "brightness_specialist"}
         if entry.get("kind") == "deployment_brightness" else entry)
        for entry in expert_bank
    ]
    return common.apply_mock_experts(images, decisions, translated)


class UnifiedExperts:
    """One native bank loader for legacy and deployment-supervised experts."""

    def __init__(self, manifest: dict[str, Any], device: Any, common: Any) -> None:
        import torch
        from streammint.brightness_specialist import BrightnessClippingSpecialist
        from streammint.conditional_restorer import ConditionalResidualRestorer
        from streammint.deployment_brightness import DeploymentBrightnessRestorer
        from streammint.snow_specialist import SnowSpecialist

        self.device, self.specifications = device, manifest["expert_bank"]
        self.models: dict[int, Any] = {}
        base = Path(manifest["_path"]).parent
        for index, entry in enumerate(self.specifications):
            if entry["kind"] == "fixed":
                continue
            checkpoint_path = verify(common, base, entry["checkpoint"], f"{entry['name']} checkpoint")
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
                validate_deployment_brightness(entry, base, common)
                model = DeploymentBrightnessRestorer(**checkpoint["model"])
            else:  # pragma: no cover - guarded during manifest validation
                raise AssertionError(kind)
            model.load_state_dict(checkpoint["state_dict"])
            self.models[index] = model.to(device).eval()

    def apply(self, images: list[np.ndarray], decisions: np.ndarray) -> Any:
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
                    condition = torch.full((len(rows),), int(entry["condition_index"]),
                                           device=self.device, dtype=torch.long)
                    restored = model(subset, condition)
                else:
                    restored = model(subset)
                scale = float(entry.get("residual_scale", 1.0))
                restored = (subset + scale * (restored - subset)).clamp(0, 1)
            output.index_copy_(0, positions, restored)
        return output


class StandardBackend:
    def __init__(self, manifest: dict[str, Any], common: Any):
        import clip
        import torch
        from streammint.data import CLIP_MEAN, CLIP_STD
        from streammint.methods import StreamingAdapter, TDAAdapter, build_text_weights

        config = manifest["backend"]
        base = Path(manifest["_path"]).parent
        meta, _ = torch.load(resolve(base, config["meta"]["path"]), weights_only=False)
        checkpoint = resolve(base, config["clip_checkpoint"]["path"])
        self.model, _ = clip.load(str(checkpoint), device="cuda")
        self.model.eval()
        class_names = [meta[wnid][0] for wnid in sorted(meta)]
        templates = (config["prompt"],) if config["kind"] == "clip" else None
        self.text = (build_text_weights(self.model, class_names, templates=templates)
                     if templates is not None else build_text_weights(self.model, class_names))
        self.kind, self.common = config["kind"], common
        self.adapter = None
        if self.kind == "tda":
            self.model.requires_grad_(False)
            self.adapter = TDAAdapter(self.model, self.text)
        elif self.kind == "mint":
            self.adapter = StreamingAdapter(
                self.model, self.text, method="mint", lr=float(config.get("lr", 0.015)),
                prior=float(config.get("prior", 10000.0)),
            )
        self.mean = torch.tensor(CLIP_MEAN, device="cuda")[None, :, None, None]
        self.std = torch.tensor(CLIP_STD, device="cuda")[None, :, None, None]

    def predict(self, images: Any) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        normalized = (images - self.mean) / self.std
        if self.adapter is not None:
            return self.adapter(normalized).argmax(-1).cpu().numpy()
        with torch.no_grad(), torch.autocast("cuda"):
            features = F.normalize(self.model.encode_image(normalized).float(), dim=-1)
            return (100.0 * features @ self.text.T).argmax(-1).cpu().numpy()

    def snapshot(self) -> dict[str, Any]:
        if self.kind == "mint":
            return {"kind": self.kind, "adapter": self.common.adapter_snapshot(self.adapter)}
        if self.kind == "tda":
            names = (
                "positive_features", "positive_entropies", "positive_valid",
                "negative_features", "negative_entropies", "negative_valid", "negative_masks",
            )
            return {
                "kind": self.kind,
                "tensors": {name: getattr(self.adapter, name).detach().cpu() for name in names},
                "diagnostics": vars(self.adapter.diagnostics).copy(),
                "batch_trace": list(self.adapter.batch_trace),
                "last_grid_logits": {
                    name: value.detach().cpu() for name, value in self.adapter.last_grid_logits.items()
                },
            }
        return {"kind": self.kind}

    def restore(self, state: dict[str, Any]) -> None:
        if state.get("kind") != self.kind:
            raise RuntimeError("backend resume-state kind mismatch")
        if self.kind == "mint":
            self.common.restore_adapter(self.adapter, state["adapter"])
        elif self.kind == "tda":
            for name, value in state["tensors"].items():
                target = getattr(self.adapter, name)
                target.copy_(value.to(target.device))
            for name, value in state["diagnostics"].items():
                setattr(self.adapter.diagnostics, name, value)
            self.adapter.batch_trace = list(state["batch_trace"])
            self.adapter.last_grid_logits = {
                name: value.to(self.text.device) for name, value in state["last_grid_logits"].items()
            }


class VendorBackend:
    def __init__(self, manifest: dict[str, Any], vendor: tuple[Any, Any]):
        import torch

        vendor_clip, tent = vendor
        self.clip, self.tent = vendor_clip, tent
        config = manifest["backend"]
        base = Path(manifest["_path"]).parent
        meta, _ = torch.load(resolve(base, config["meta"]["path"]), weights_only=False)
        classes = [meta[wnid][0] for wnid in sorted(meta)]
        self.inventory = type("ClassInventory", (), {"classes": classes})()
        self.text_tokens = torch.cat([
            vendor_clip.tokenize(f"a photo of a {name}") for name in classes
        ]).cuda()
        checkpoint = resolve(base, config["clip_checkpoint"]["path"])
        model, _ = vendor_clip.load(str(checkpoint), device="cuda")
        model.visual = tent.configure_model(model.visual, "ViT-L/14")
        params, _ = tent.collect_params(model.visual, "ViT-L/14")
        optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0)
        self.kind = config["kind"]
        if self.kind == "clipartt":
            self.adapter = tent.Tent(model, optimizer, steps=10, method="clipartt", episodic=True)
        else:
            self.adapter = self._memory_bounded_tent(model, optimizer)
        self.mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device="cuda")[None, :, None, None]
        self.std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device="cuda")[None, :, None, None]

    def _memory_bounded_tent(self, model: Any, optimizer: Any):
        import torch
        import torch.nn.functional as F

        tent = self.tent
        text_tokens = self.text_tokens

        class Adapter:
            def __init__(self):
                self.model, self.optimizer, self.steps = model, optimizer, 10
                self.model_state, self.optimizer_state = tent.copy_model_and_optimizer(model, optimizer)
                with torch.no_grad():
                    self.text_features = F.normalize(model.encode_text(text_tokens), dim=-1).detach()

            def __call__(self, images: Any, *_args: Any, **_kwargs: Any) -> None:
                tent.load_model_and_optimizer(
                    self.model, self.optimizer, self.model_state, self.optimizer_state
                )
                for _ in range(self.steps):
                    image_features = F.normalize(self.model.encode_image(images), dim=-1)
                    logits = self.model.logit_scale.exp() * image_features @ self.text_features.T
                    tent.softmax_entropy(logits).mean().backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        return Adapter()

    def predict(self, images: Any) -> np.ndarray:
        import torch

        normalized = (images - self.mean) / self.std
        self.adapter(
            normalized, self.text_tokens, self.inventory, "cuda", K=3, target_method=1
        )
        with torch.no_grad():
            image_features = self.adapter.model.encode_image(normalized)
            text_features = self.adapter.model.encode_text(self.text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return (100.0 * image_features @ text_features.T).argmax(-1).cpu().numpy()

    def snapshot(self) -> dict[str, Any]:
        # Both vendor backends are episodic. Every batch starts by restoring
        # this process's source model/optimizer state, so no cross-batch model
        # state is part of the protocol.
        return {"kind": self.kind, "episodic_batch_restart": True}

    def restore(self, state: dict[str, Any]) -> None:
        if state != {"kind": self.kind, "episodic_batch_restart": True}:
            raise RuntimeError("invalid episodic backend resume state")


def real_setup() -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for real five-backend replay")
    torch.cuda.memory.set_per_process_memory_fraction(MEMORY_FRACTION)


def save_progress(
    output_dir: Path, *, next_batch: int, records: list[dict[str, Any]], backend: Any,
    mock: bool, protocol_hash: str, elapsed_seconds: float, peak_gpu_gib: float,
) -> None:
    slot = next_batch % 2
    records_path = output_dir / f"records.slot{slot}.jsonl"
    state_path = output_dir / (f"state.slot{slot}.json" if mock else f"state.slot{slot}.pt")
    atomic_records(records_path, records)
    state = backend.snapshot()
    if mock:
        atomic_json(state_path, state)
    else:
        state["random"] = common_random_snapshot()
        atomic_torch_save(state_path, state)
    progress = {
        "schema_version": 1, "next_batch": next_batch, "slot": slot,
        "records": records_path.name, "records_sha256": sha256(records_path),
        "records_count": len(records), "state": state_path.name,
        "state_sha256": sha256(state_path), "protocol_sha256": protocol_hash,
        "elapsed_seconds": elapsed_seconds, "peak_gpu_gib": peak_gpu_gib,
    }
    atomic_json(output_dir / "progress.json", progress)


def common_random_snapshot() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_random_snapshot(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def resume_progress(
    output_dir: Path, *, protocol_hash: str, backend: Any, mock: bool, batch_size: int,
) -> tuple[int, list[dict[str, Any]], float, float]:
    progress_path = output_dir / "progress.json"
    if not progress_path.exists():
        return 0, [], 0.0, 0.0
    progress = json.loads(progress_path.read_text())
    if progress.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("resume protocol hash mismatch")
    records_path, state_path = output_dir / progress["records"], output_dir / progress["state"]
    if sha256(records_path) != progress.get("records_sha256"):
        raise RuntimeError("resume record-slot hash mismatch")
    if sha256(state_path) != progress.get("state_sha256"):
        raise RuntimeError("resume state-slot hash mismatch")
    records = read_records(records_path)
    next_batch = int(progress["next_batch"])
    if len(records) != int(progress["records_count"]) or len(records) != next_batch * batch_size:
        raise RuntimeError("resume record count is not an exact completed-batch boundary")
    if mock:
        state = json.loads(state_path.read_text())
    else:
        import torch
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    backend.restore(state)
    if not mock:
        restore_random_snapshot(state["random"])
    return next_batch, records, float(progress.get("elapsed_seconds", 0.0)), \
        float(progress.get("peak_gpu_gib", 0.0))


def validate_record_prefix(
    records: list[dict[str, Any]], rows: list[dict[str, Any]], order: np.ndarray,
    protocol_hash: str,
) -> None:
    if len(records) > len(order) or len(records) % BATCH_SIZE:
        raise RuntimeError("resume records are not a valid completed-batch prefix")
    for position, record in enumerate(records):
        source = rows[int(order[position])]
        required = {
            "key": str(source["key"]), "image_id": str(source.get("image_id", source["sample"])),
            "sample": source["sample"], "target": int(source["target"]),
            "raw_sha256": str(source["raw_sha256"]), "stream_position": position,
            "batch": position // BATCH_SIZE, "batch_position": position % BATCH_SIZE,
            "protocol_sha256": protocol_hash,
        }
        if any(record.get(name) != value for name, value in required.items()):
            raise RuntimeError(f"resume prefix differs from frozen stream at position {position}")


def run(
    path: Path, output_dir: Path, *, mock: bool, common: Any, vendor: Any,
    stop_after_batches: int | None = None,
) -> dict[str, Any] | None:
    manifest = validate_manifest(path, mock=mock, common=common)
    rows = manifest["_rows"]
    order = (
        np.random.default_rng(SEED).permutation(len(rows))
        if manifest["input_stream"].get("shuffle", True) else np.arange(len(rows))
    )
    base = path.resolve().parent
    router_path = resolve(base, manifest["router"]["artifact"]["path"])
    router = common.load_router(router_path)
    features = common.FeatureProvider(manifest)
    images = common.StreamImages()
    kind = manifest["backend"]["kind"]
    row_ids = [str(row.get("image_id", row["sample"])) for row in rows]
    prompt_sha = canonical_sha256(manifest["backend"].get("prompt"))
    ids_sha = hashlib.sha256(
        "\n".join(row_ids).encode()
    ).hexdigest()
    protocol = {
        "schema_version": 1,
        "status": "cpu_mock_no_scientific_evidence" if mock else "frozen_five_backend_routed_replay",
        "manifest_sha256": sha256(path), "runner_sha256": sha256(Path(__file__)),
        "router_sha256": sha256(router_path), "backend": kind,
        "native_reset_policy": RESET_POLICY[kind], "seed": SEED,
        "reset_policy": RESET_POLICY[kind],
        "batch_size": BATCH_SIZE, "memory_fraction": MEMORY_FRACTION,
        "stream_order_sha256": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
        "stream_keys_sha256": hashlib.sha256(
            "\n".join(str(rows[int(index)]["key"]) for index in order).encode()
        ).hexdigest(),
        "routing_inputs": "18-D descriptor plus current-batch statistics only",
        "forbidden_routing_inputs": ["target", "audit_domain", "corruption name", "backend logits"],
        "same_routed_stage_for_all_backends": True,
        "model_sha256": manifest["backend"]["clip_checkpoint"]["sha256"],
        "prompt_sha256": prompt_sha,
        "test_ids_sequence_sha256": ids_sha,
        "backend_native_source_sha256": {
            Path(entry["path"]).name: entry["sha256"]
            for entry in manifest["backend"].get("native_source_artifacts", [])
        },
        "runtime_dependency_sha256": {
            str(resolve(base, entry["path"])): entry["sha256"]
            for entry in manifest["runtime_dependency_closure"]
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if (output_dir / "complete.json").exists():
        raise FileExistsError(f"replay output is already complete: {output_dir}")
    existing_protocol = None
    if protocol_path.exists():
        existing_protocol = json.loads(protocol_path.read_text())
        comparable = dict(existing_protocol)
        for name in ("observed_peak_gpu_gib", "observed_peak_fraction", "memory_gate_pass"):
            comparable.pop(name, None)
        if comparable != protocol:
            raise RuntimeError("resume protocol differs; use a new output directory")
    else:
        if any(output_dir.iterdir()):
            raise RuntimeError("nonempty output directory has no bound protocol")
        atomic_json(protocol_path, protocol)
    protocol_hash = rendered_json_sha256(protocol)
    result_path = output_dir / "result.json"
    if result_path.exists():
        # Recover only the narrow crash window after an atomic final result but
        # before its completion marker. Any inconsistency fails closed.
        recovered = json.loads(result_path.read_text())
        if existing_protocol is None or recovered.get("protocol") != existing_protocol \
                or recovered.get("protocol_sha256") != sha256(protocol_path):
            raise RuntimeError("uncommitted final result/protocol mismatch")
        recovered_records = recovered.get("records")
        if not isinstance(recovered_records, list) or len(recovered_records) != len(rows) \
                or recovered.get("records_sha256") != canonical_sha256(recovered_records):
            raise RuntimeError("uncommitted final result records are invalid")
        finalized_hash = sha256(protocol_path)
        validate_record_prefix(recovered_records, rows, order, finalized_hash)
        recovered_accuracy = 100.0 * sum(bool(row.get("correct")) for row in recovered_records) / len(rows)
        if abs(float(recovered.get("accuracy", -1)) - recovered_accuracy) > 1e-12:
            raise RuntimeError("uncommitted final result accuracy is invalid")
        atomic_json(output_dir / "complete.json", {
            "result_sha256": sha256(result_path),
            "records_sha256": recovered["records_sha256"],
            "protocol_sha256": recovered["protocol_sha256"],
            "recovered_after_atomic_result": True,
        })
        return recovered
    if mock:
        backend, experts = MockFiveBackend(kind), None
    else:
        real_setup()
        import torch
        experts = UnifiedExperts(manifest, torch.device("cuda"), common)
        backend = VendorBackend(manifest, vendor) if kind in {"tent", "clipartt"} \
            else StandardBackend(manifest, common)
        torch.cuda.reset_peak_memory_stats()
    start_batch, records, elapsed_before, prior_peak = resume_progress(
        output_dir, protocol_hash=protocol_hash, backend=backend, mock=mock,
        batch_size=BATCH_SIZE,
    )
    validate_record_prefix(records, rows, order, protocol_hash)
    index_base = Path(manifest["_index_path"]).parent
    started = time.perf_counter()
    batches = (len(order) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch in range(start_batch, batches):
        if stop_after_batches is not None and batch >= stop_after_batches:
            return None
        positions = order[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
        batch_rows = [rows[int(position)] for position in positions]
        arrays = [images.load(index_base, row["image"]) for row in batch_rows]
        hashes = [hashlib.sha256(array.tobytes()).hexdigest() for array in arrays]
        if any(str(row.get("raw_sha256")) != digest for row, digest in zip(batch_rows, hashes)):
            raise RuntimeError("raw image/stream-index hash mismatch")
        descriptors = np.stack([
            features.descriptor(str(row["key"]), image, digest)
            for row, image, digest in zip(batch_rows, arrays, hashes)
        ])
        decisions, utilities = common.route_batch(router, descriptors)
        if mock:
            restored = apply_mock_experts(common, arrays, decisions, manifest["expert_bank"])
        else:
            restored = experts.apply(arrays, decisions)
        predictions = backend.predict(restored)
        for local, (row, prediction, decision, digest) in enumerate(
            zip(batch_rows, predictions, decisions, hashes)
        ):
            expert = "identity" if decision < 0 else manifest["expert_bank"][int(decision)]["name"]
            target = int(row["target"])
            records.append({
                "key": str(row["key"]), "image_id": str(row.get("image_id", row["sample"])),
                "sample": row["sample"], "target": target,
                "prediction": int(prediction), "correct": int(prediction) == target,
                "stream_position": len(records), "batch": batch, "batch_position": local,
                "expert_index": int(decision), "expert": expert, "raw_sha256": digest,
                "decision": "abstain" if decision < 0 else "route",
                "descriptor_sha256": hashlib.sha256(
                    np.asarray(descriptors[local], dtype="<f8").tobytes()
                ).hexdigest(),
                "predicted_utility": None if decision < 0 else float(utilities[local, int(decision)]),
                "protocol_sha256": protocol_hash,
                "audit_domain": row.get("audit_domain"),
            })
        elapsed = elapsed_before + time.perf_counter() - started
        peak = prior_peak
        if not mock:
            import torch
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30,
                       torch.cuda.max_memory_reserved() / 2**30)
        save_progress(
            output_dir, next_batch=batch + 1, records=records, backend=backend,
            mock=mock, protocol_hash=protocol_hash, elapsed_seconds=elapsed,
            peak_gpu_gib=peak,
        )
    timing = {"seconds": elapsed_before + time.perf_counter() - started}
    if not mock:
        import torch
        timing["peak_allocated_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
        timing["peak_reserved_gpu_gib"] = torch.cuda.max_memory_reserved() / 2**30
        timing["peak_gpu_gib"] = max(prior_peak, timing["peak_allocated_gpu_gib"],
                                     timing["peak_reserved_gpu_gib"])
        total = torch.cuda.get_device_properties(0).total_memory
        if timing["peak_gpu_gib"] * 2**30 > MEMORY_FRACTION * total:
            raise RuntimeError("measured peak exceeded the 68% total-memory gate")
        timing["total_gpu_gib"] = total / 2**30
        timing["observed_peak_fraction"] = timing["peak_gpu_gib"] / timing["total_gpu_gib"]
    # The immutable preflight protocol is used during resume. Once the stream
    # is complete, finalize it with the actually observed peak and rebind every
    # output record to the finalized protocol hash.
    protocol["observed_peak_gpu_gib"] = None if mock else timing["peak_gpu_gib"]
    protocol["observed_peak_fraction"] = None if mock else timing["observed_peak_fraction"]
    protocol["memory_gate_pass"] = True
    atomic_json(protocol_path, protocol)
    protocol_hash = sha256(protocol_path)
    for record in records:
        record["protocol_sha256"] = protocol_hash
    result = {
        "protocol": protocol, "protocol_sha256": protocol_hash,
        "accuracy": 100.0 * sum(row["correct"] for row in records) / len(records),
        "records": records, "records_sha256": canonical_sha256(records),
        "expert_counts": dict(Counter(row["expert"] for row in records)),
        "timing": timing,
        "environment": {"mode": "mock" if mock else "cuda", "python": platform.python_version()},
    }
    atomic_json(output_dir / "result.json", result)
    atomic_json(output_dir / "complete.json", {
        "result_sha256": sha256(output_dir / "result.json"),
        "records_sha256": result["records_sha256"], "protocol_sha256": protocol_hash,
    })
    if not mock:
        del backend, experts
        gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--stop-after-batches", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raw = json.loads(args.manifest.read_text())
    kind = raw.get("backend", {}).get("kind")
    if kind not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    common, vendor = bootstrap(kind)
    validated = validate_manifest(args.manifest.resolve(), mock=args.mock, common=common)
    if args.validate_only:
        print(json.dumps({"validated": True, "backend": kind, "rows": len(validated["_rows"]),
                          "mock": args.mock, "reset": RESET_POLICY[kind]}))
        return
    result = run(args.manifest.resolve(), args.output_dir.resolve(), mock=args.mock,
                 common=common, vendor=vendor, stop_after_batches=args.stop_after_batches)
    if result is None:
        print(json.dumps({"status": "intentionally_interrupted_for_resume_test",
                          "backend": kind}))
        return
    print(json.dumps({"status": result["protocol"]["status"], "backend": kind,
                      "records": len(result["records"]), "accuracy": result["accuracy"]}))


if __name__ == "__main__":
    main()
