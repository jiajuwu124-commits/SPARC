#!/usr/bin/env python3
"""Replay a frozen SPARC utility router on a genuinely mixed input stream.

This runner has two deliberately separate modes:

* real replay requires a ``frozen_for_routed_replay`` manifest and CUDA;
* ``--mock`` is CPU-only plumbing validation and is never scientific evidence.

Routing receives only the 18-D masked-risk descriptor and statistics computed
from the current inference batch.  Corruption/domain names and class targets
remain audit metadata and are never passed to the router or an expert.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streammint.utility_router import (  # noqa: E402
    RISK_FEATURE_ORDER,
    UtilityRouter,
    augment_batch_statistics,
    risk_features,
)


SEED = 1109
BATCH_SIZE = 20
MEMORY_FRACTION = 0.68
RISK_BANK = ("identity", "gaussian_1.0", "gaussian_1.5", "median_3")
ALLOWED_EXPERT_KINDS = {
    "fixed",
    "conditional_restorer",
    "snow_specialist",
    "brightness_specialist",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload["records"] if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("input stream index must contain a non-empty record list")
    required = {"key", "sample", "target", "image"}
    for row in records:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"stream row is missing {sorted(required)}")
        image = row["image"]
        if not isinstance(image, dict) or "path" not in image:
            raise ValueError("every stream image requires a path")
        if image.get("format", "npy") == "npy" and "index" not in image:
            raise ValueError("npy stream images require an array index")
    keys = [str(row["key"]) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError("input stream keys must be unique, including across domains")
    return records


def verify_artifact(base: Path, entry: dict[str, Any], label: str) -> Path:
    if not isinstance(entry, dict) or set(("path", "sha256")) - set(entry):
        raise ValueError(f"{label} must contain path and sha256")
    path = resolve(base, str(entry["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != str(entry["sha256"]):
        raise RuntimeError(f"{label} hash mismatch: {path}; expected {entry['sha256']}, got {actual}")
    return path


def validate_manifest(path: Path, *, mock: bool) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported routed-replay manifest schema")
    if int(manifest.get("seed", -1)) != SEED:
        raise ValueError("routed replay requires seed 1109")
    if int(manifest.get("batch_size", -1)) != BATCH_SIZE:
        raise ValueError("routed replay requires batch size 20")
    if float(manifest.get("memory_fraction", -1)) != MEMORY_FRACTION:
        raise ValueError("routed replay requires the 68% GPU-memory cap")
    if not mock and manifest.get("status") != "frozen_for_routed_replay":
        raise RuntimeError("real replay requires status=frozen_for_routed_replay")

    backend = manifest.get("backend", {})
    if backend.get("kind") not in {"clip", "mint"}:
        raise ValueError("backend.kind must be clip or mint")
    if backend.get("kind") == "mint" and backend.get("reset") != "once_before_stream":
        raise ValueError("Mint must reset exactly once before the complete mixed stream")

    experts = manifest.get("expert_bank")
    if not isinstance(experts, list) or not experts:
        raise ValueError("expert_bank must be non-empty")
    names = [str(entry.get("name")) for entry in experts]
    if len(set(names)) != len(names) or "identity" in names:
        raise ValueError("expert names must be unique and must not include identity")
    for entry in experts:
        if entry.get("kind") not in ALLOWED_EXPERT_KINDS:
            raise ValueError(f"unsupported expert kind: {entry.get('kind')}")
        if entry["kind"] == "fixed" and not isinstance(entry.get("operator"), str):
            raise ValueError("fixed expert requires operator")
        if entry["kind"] != "fixed" and "checkpoint" not in entry:
            raise ValueError(f"{entry['kind']} requires checkpoint")
        if entry["kind"] == "brightness_specialist":
            scale = float(entry.get("residual_scale", 1.0))
            if not np.isfinite(scale) or not 0.0 <= scale <= 2.0:
                raise ValueError("brightness specialist residual_scale must be in [0, 2]")
            source_hash = str(entry.get("source_sha256", ""))
            if not len(source_hash) == 64 or any(
                character not in "0123456789abcdef" for character in source_hash
            ):
                raise ValueError("brightness specialist requires lowercase source_sha256")

    source = manifest.get("features", {})
    if source.get("kind") not in {"precomputed_descriptors", "precomputed_risks", "online_risks"}:
        raise ValueError("features.kind must be precomputed_descriptors, precomputed_risks, or online_risks")
    if source.get("kind") == "online_risks" and tuple(source.get("risk_bank", ())) != RISK_BANK:
        raise ValueError(f"online risk bank must be {RISK_BANK}")
    if source.get("kind") != "online_risks" and "artifact" not in source:
        raise ValueError("precomputed features require an artifact")

    base = path.resolve().parent
    verify_artifact(base, manifest["router"]["artifact"], "utility-router artifact")
    index_path = verify_artifact(base, manifest["input_stream"]["index"], "stream index")
    rows = load_json_records(index_path)
    source_entries = manifest["input_stream"].get("source_artifacts", [])
    verified_sources = {verify_artifact(base, item, "input source") for item in source_entries}
    referenced_sources = {resolve(index_path.parent, str(row["image"]["path"])) for row in rows}
    if referenced_sources != verified_sources:
        missing = sorted(str(value) for value in referenced_sources - verified_sources)
        extra = sorted(str(value) for value in verified_sources - referenced_sources)
        raise ValueError(f"source_artifacts must exactly cover stream sources; missing={missing}, extra={extra}")

    if source.get("kind") != "online_risks":
        verify_artifact(base, source["artifact"], "feature artifact")
    for expert in experts:
        if expert["kind"] != "fixed":
            verify_artifact(base, expert["checkpoint"], f"{expert['name']} checkpoint")
        if expert["kind"] == "brightness_specialist":
            source_path = ROOT / "src/streammint/brightness_specialist.py"
            actual_source_hash = sha256_file(source_path)
            if actual_source_hash != expert["source_sha256"]:
                raise RuntimeError(
                    "brightness specialist source hash mismatch: "
                    f"expected {expert['source_sha256']}, got {actual_source_hash}"
                )
    if not mock:
        verify_artifact(base, backend["clip_checkpoint"], "CLIP checkpoint")
        verify_artifact(base, backend["meta"], "ImageNet metadata")

    manifest["_path"] = str(path.resolve())
    manifest["_index_path"] = str(index_path)
    manifest["_rows"] = rows
    return manifest


class StreamImages:
    def __init__(self) -> None:
        self.arrays: dict[Path, np.ndarray] = {}

    def load(self, index_base: Path, specification: dict[str, Any]) -> np.ndarray:
        path = resolve(index_base, str(specification["path"]))
        image_format = specification.get("format", "npy")
        if image_format == "npy":
            if path not in self.arrays:
                self.arrays[path] = np.load(path, mmap_mode="r")
            array = np.asarray(self.arrays[path][int(specification["index"])]).copy()
        elif image_format == "image":
            array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        else:
            raise ValueError(f"unsupported stream image format: {image_format}")
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"stream image must be uint8 HWC RGB, got {array.shape}/{array.dtype}")
        return np.ascontiguousarray(array)


class FeatureProvider:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.config = manifest["features"]
        self.base = Path(manifest["_path"]).parent
        self.kind = self.config["kind"]
        self.by_key: dict[str, tuple[np.ndarray, str | None]] = {}
        if self.kind == "online_risks":
            return
        path = resolve(self.base, self.config["artifact"]["path"])
        with np.load(path, allow_pickle=False) as stored:
            keys = [str(value) for value in stored["keys"].tolist()]
            if "raw_sha256" not in stored:
                raise ValueError("precomputed features require per-row raw_sha256")
            hashes = [value.decode() if isinstance(value, bytes) else str(value)
                      for value in stored["raw_sha256"]]
            if self.kind == "precomputed_descriptors":
                if "feature_order" not in stored:
                    raise ValueError("precomputed descriptors require feature_order")
                feature_order = tuple(str(value) for value in stored["feature_order"].tolist())
                if feature_order != RISK_FEATURE_ORDER:
                    raise ValueError("precomputed descriptor feature_order is not the frozen 18-D order")
            values = stored["features" if self.kind == "precomputed_descriptors" else "risks"]
        if len(keys) != len(values) or len(set(keys)) != len(keys):
            raise ValueError("feature artifact keys are missing, duplicated, or misaligned")
        if len(hashes) != len(keys):
            raise ValueError("feature artifact raw hashes are misaligned")
        for position, key in enumerate(keys):
            self.by_key[key] = (values[position], hashes[position])

    def descriptor(self, key: str, image: np.ndarray, raw_hash: str) -> np.ndarray:
        if self.kind == "online_risks":
            from streammint.risk_route_fast import masked_risks_fast

            risks = masked_risks_fast(Image.fromarray(image, mode="RGB"), RISK_BANK)
            return risk_features(risks[None])[0]
        if key not in self.by_key:
            raise KeyError(f"feature artifact has no row for stream key {key!r}")
        value, expected_hash = self.by_key[key]
        if expected_hash is not None and expected_hash != raw_hash:
            raise RuntimeError(f"feature/raw-image hash mismatch for stream key {key!r}")
        if self.kind == "precomputed_risks":
            return risk_features(np.asarray(value)[None])[0]
        descriptor = np.asarray(value, dtype=np.float64)
        if descriptor.shape != (len(RISK_FEATURE_ORDER),):
            raise ValueError(f"expected 18-D descriptor, got {descriptor.shape}")
        if not np.isfinite(descriptor).all():
            raise ValueError("descriptor contains non-finite values")
        return descriptor


def load_router(path: Path) -> Any:
    """Load the legacy NumPy ridge artifact or the hashed nonlinear joblib."""
    if path.suffix == ".npz":
        return UtilityRouter.load(str(path))
    if path.suffix == ".joblib":
        from streammint.nonlinear_utility_router import NonlinearUtilityRouter

        return NonlinearUtilityRouter.load(path)
    raise ValueError("utility-router artifact must end in .npz or .joblib")


def route_batch(router: Any, descriptors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Route one batch without accepting metadata, labels, images, or logits."""
    descriptors = np.asarray(descriptors, dtype=np.float64)
    if descriptors.ndim != 2 or descriptors.shape[1] != len(RISK_FEATURE_ORDER):
        raise ValueError("route_batch accepts only an [N,18] descriptor matrix")
    batch_ids = np.zeros(len(descriptors), dtype=np.int64)
    features = augment_batch_statistics(descriptors, batch_ids, mode=router.feature_mode)
    return router.select(features)


class MockBackend:
    """A deterministic stateful CPU backend used only to test replay plumbing."""

    def __init__(self, kind: str, classes: int) -> None:
        self.kind = kind
        self.classes = classes
        self.seen = 0

    def __call__(self, images: np.ndarray) -> np.ndarray:
        values = np.rint(images.mean(axis=(1, 2, 3))).astype(np.int64)
        if self.kind == "mint":
            values += self.seen // BATCH_SIZE
            self.seen += len(images)
        return values % self.classes

    def state(self) -> dict[str, Any]:
        return {"seen": self.seen}

    def load_state(self, state: dict[str, Any]) -> None:
        self.seen = int(state["seen"])


def apply_mock_experts(images: list[np.ndarray], decisions: np.ndarray,
                       expert_bank: list[dict[str, Any]]) -> np.ndarray:
    # Mock transformations are intentionally simple but decision-dependent.
    output = np.stack(images).astype(np.float32)
    for row, decision in enumerate(decisions):
        if decision < 0:
            continue
        kind = expert_bank[int(decision)]["kind"]
        delta = {
            "fixed": 1.0,
            "conditional_restorer": 2.0,
            "snow_specialist": 3.0,
            "brightness_specialist": 4.0 * float(
                expert_bank[int(decision)].get("residual_scale", 1.0)
            ),
        }[kind]
        output[row] = np.clip(output[row] + delta, 0.0, 255.0)
    return output


class RealExperts:
    def __init__(self, manifest: dict[str, Any], device: Any) -> None:
        import torch
        from streammint.brightness_specialist import BrightnessClippingSpecialist
        from streammint.conditional_restorer import ConditionalResidualRestorer
        from streammint.snow_specialist import SnowSpecialist

        self.device = device
        self.specifications = manifest["expert_bank"]
        self.models: dict[int, Any] = {}
        base = Path(manifest["_path"]).parent
        for index, entry in enumerate(self.specifications):
            if entry["kind"] == "fixed":
                continue
            checkpoint = torch.load(
                resolve(base, entry["checkpoint"]["path"]), map_location="cpu", weights_only=False
            )
            if entry["kind"] == "conditional_restorer":
                model = ConditionalResidualRestorer(**checkpoint["model"])
                if "condition_index" not in entry:
                    raise ValueError("conditional restorer expert requires a frozen condition_index")
            elif entry["kind"] == "snow_specialist":
                if checkpoint.get("format") != "sparc-snow-specialist-v1":
                    raise RuntimeError("unsupported snow specialist checkpoint")
                model = SnowSpecialist(**checkpoint["model"])
            else:
                if checkpoint.get("format") != "sparc-brightness-clipping-specialist-v1":
                    raise RuntimeError("unsupported brightness specialist checkpoint")
                if checkpoint.get("seed") != SEED:
                    raise RuntimeError("brightness specialist checkpoint seed mismatch")
                if checkpoint.get("uses_imagenet_class_labels") is not False:
                    raise RuntimeError("brightness specialist supervision ledger is missing")
                actual_checkpoint_hash = sha256_file(
                    resolve(base, entry["checkpoint"]["path"])
                )
                if actual_checkpoint_hash != entry["checkpoint"]["sha256"]:
                    raise RuntimeError("brightness specialist checkpoint hash changed after validation")
                source_path = ROOT / "src/streammint/brightness_specialist.py"
                if sha256_file(source_path) != entry["source_sha256"]:
                    raise RuntimeError("brightness specialist source hash changed after validation")
                model = BrightnessClippingSpecialist(**checkpoint["model"])
            model.load_state_dict(checkpoint["state_dict"])
            self.models[index] = model.to(device).eval()

    def apply(self, images: list[np.ndarray], decisions: np.ndarray) -> Any:
        import torch
        from streammint.data import preprocess_image

        cpu = []
        for image, decision in zip(images, decisions):
            if decision >= 0 and self.specifications[int(decision)]["kind"] == "fixed":
                operator = self.specifications[int(decision)]["operator"]
                image = np.asarray(preprocess_image(Image.fromarray(image), operator))
            cpu.append(torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0)
        output = torch.stack(cpu).to(self.device, non_blocking=True)
        for expert_index, model in self.models.items():
            rows = np.flatnonzero(decisions == expert_index)
            if not len(rows):
                continue
            positions = torch.as_tensor(rows, device=self.device)
            subset = output.index_select(0, positions)
            entry = self.specifications[expert_index]
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type, enabled=self.device.type == "cuda"
            ):
                if entry["kind"] == "conditional_restorer":
                    condition = torch.full(
                        (len(rows),), int(entry["condition_index"]), device=self.device,
                        dtype=torch.long,
                    )
                    restored = model(subset, condition)
                    scale = float(entry.get("residual_scale", 1.0))
                    restored = (subset + scale * (restored - subset)).clamp(0.0, 1.0)
                elif entry["kind"] == "brightness_specialist":
                    restored = model(subset)
                    scale = float(entry.get("residual_scale", 1.0))
                    restored = (subset + scale * (restored - subset)).clamp(0.0, 1.0)
                else:
                    restored = model(subset)
            output.index_copy_(0, positions, restored)
        return output


def adapter_snapshot(adapter: Any) -> dict[str, Any]:
    def cpu(value: Any) -> Any:
        return None if value is None else value.detach().cpu()

    return {
        "pre": {name: cpu(getattr(adapter.pre_bank, name)) for name in
                ("class_sum", "class_mass", "total_sum", "total_mass")},
        "post": {name: cpu(getattr(adapter.post_bank, name)) for name in
                 ("class_sum", "class_mass", "total_sum", "total_mass")},
        "gradient_mean": cpu(adapter.gradient_mean),
        "gradient_fast": cpu(adapter.gradient_fast),
        "gradient_batches": adapter.gradient_batches,
        "gradient_scale": cpu(adapter.gradient_scale),
        "last_gradient_cosine": adapter.last_gradient_cosine,
        "last_fast_gate": adapter.last_fast_gate,
        "diagnostics": asdict(adapter.diagnostics),
        "batch_trace": adapter.batch_trace,
    }


def restore_adapter(adapter: Any, state: dict[str, Any]) -> None:
    from streammint.methods import AdapterDiagnostics

    for bank_name in ("pre", "post"):
        bank = getattr(adapter, f"{bank_name}_bank")
        for name, value in state[bank_name].items():
            getattr(bank, name).copy_(value.to(getattr(bank, name).device))
    adapter.gradient_mean = None if state["gradient_mean"] is None else state["gradient_mean"].cuda()
    adapter.gradient_fast = None if state["gradient_fast"] is None else state["gradient_fast"].cuda()
    adapter.gradient_batches = int(state["gradient_batches"])
    adapter.gradient_scale.copy_(state["gradient_scale"].to(adapter.gradient_scale.device))
    adapter.last_gradient_cosine = float(state["last_gradient_cosine"])
    adapter.last_fast_gate = float(state["last_fast_gate"])
    adapter.diagnostics = AdapterDiagnostics(**state["diagnostics"])
    adapter.batch_trace = list(state["batch_trace"])
    adapter._restore_source()
    if adapter.optimizer.state:
        raise RuntimeError("resumed Mint optimizer must be empty at a batch boundary")


def random_snapshot() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_random_snapshot(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def make_real_backend(manifest: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    import clip
    import torch
    from streammint.data import CLIP_MEAN, CLIP_STD
    from streammint.methods import StreamingAdapter, build_text_weights

    base = Path(manifest["_path"]).parent
    config = manifest["backend"]
    checkpoint = resolve(base, config["clip_checkpoint"]["path"])
    meta_path = resolve(base, config["meta"]["path"])
    meta, _ = torch.load(meta_path, weights_only=False)
    classes = [meta[wnid][0] for wnid in sorted(meta)]
    model, _ = clip.load(str(checkpoint), device="cuda")
    model.eval()
    text = build_text_weights(model, classes)
    adapter = None
    if config["kind"] == "mint":
        adapter = StreamingAdapter(
            model, text, method="mint", lr=float(config.get("lr", 0.015)),
            prior=float(config.get("prior", 10000.0)),
        )
    mean = torch.tensor(CLIP_MEAN, device="cuda")[None, :, None, None]
    std = torch.tensor(CLIP_STD, device="cuda")[None, :, None, None]
    return model, text, adapter, (mean, std)


def real_predict(images: Any, backend: tuple[Any, Any, Any, Any]) -> Any:
    import torch
    import torch.nn.functional as F

    model, text, adapter, (mean, std) = backend
    normalized = (images - mean) / std
    if adapter is not None:
        return adapter(normalized).argmax(dim=-1).cpu().numpy()
    with torch.no_grad(), torch.autocast("cuda"):
        features = F.normalize(model.encode_image(normalized).float(), dim=-1)
        return (100.0 * features @ text.T).argmax(dim=-1).cpu().numpy()


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open() as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_progress(output_dir: Path, *, next_batch: int, records: list[dict[str, Any]],
                  state: dict[str, Any], mock: bool) -> None:
    records_path = output_dir / "records.partial.jsonl"
    temporary = records_path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as target:
        for row in records:
            target.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(records_path)
    if mock:
        state_path = output_dir / "state.partial.json"
        atomic_json(state_path, state)
    else:
        import torch

        state_path = output_dir / "state.partial.pt"
        temporary_state = state_path.with_suffix(".pt.tmp")
        torch.save(state, temporary_state)
        temporary_state.replace(state_path)
    atomic_json(output_dir / "progress.json", {
        "next_batch": next_batch,
        "records": len(records),
        "records_sha256": sha256_file(records_path),
        "state_path": state_path.name,
        "state_sha256": sha256_file(state_path),
    })


def resume_progress(output_dir: Path, protocol_hash: str, *, mock: bool,
                    order: np.ndarray, rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None]:
    progress_path = output_dir / "progress.json"
    if not progress_path.exists():
        return 0, [], None
    progress = json.loads(progress_path.read_text())
    records_path = output_dir / "records.partial.jsonl"
    state_path = output_dir / progress["state_path"]
    if sha256_file(records_path) != progress["records_sha256"] or sha256_file(state_path) != progress["state_sha256"]:
        raise RuntimeError("resume artifacts failed hash validation")
    records = read_records(records_path)
    if len(records) != int(progress["records"]):
        raise RuntimeError("resume record count mismatch")
    expected_keys = [str(rows[int(position)]["key"]) for position in order[:len(records)]]
    if [row["key"] for row in records] != expected_keys:
        raise RuntimeError("resume record prefix does not match the frozen stream order")
    if any(row.get("protocol_sha256") != protocol_hash for row in records):
        raise RuntimeError("resume records refer to another protocol")
    if mock:
        state = json.loads(state_path.read_text())
    else:
        import torch

        state = torch.load(state_path, map_location="cpu", weights_only=False)
    return int(progress["next_batch"]), records, state


def run(manifest_path: Path, output_dir: Path, *, mock: bool,
        stop_after_batches: int | None = None) -> dict[str, Any] | None:
    manifest = validate_manifest(manifest_path, mock=mock)
    base = manifest_path.resolve().parent
    rows: list[dict[str, Any]] = manifest["_rows"]
    router_path = resolve(base, manifest["router"]["artifact"]["path"])
    router = load_router(router_path)
    expert_names = tuple(entry["name"] for entry in manifest["expert_bank"])
    if router.expert_names != expert_names:
        raise RuntimeError(f"router experts {router.expert_names} != manifest experts {expert_names}")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows)) if manifest["input_stream"].get("shuffle", True) \
        else np.arange(len(rows))
    order_hash = hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "src/streammint/utility_router.py",
        ROOT / "src/streammint/risk_route_fast.py",
        ROOT / "src/streammint/data.py",
        ROOT / "src/streammint/conditional_restorer.py",
        ROOT / "src/streammint/snow_specialist.py",
        ROOT / "src/streammint/methods.py",
    ]
    if any(
        expert["kind"] == "brightness_specialist"
        for expert in manifest["expert_bank"]
    ):
        source_paths.append(ROOT / "src/streammint/brightness_specialist.py")
    if router_path.suffix == ".joblib":
        source_paths.append(ROOT / "src/streammint/nonlinear_utility_router.py")
    protocol = {
        "schema_version": 1,
        "status": "cpu_mock_no_scientific_evidence" if mock else "frozen_routed_replay",
        "manifest_sha256": sha256_file(manifest_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "source_sha256": {
            str(source.relative_to(ROOT)): sha256_file(source) for source in source_paths
        },
        "router_sha256": sha256_file(router_path),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "memory_fraction": MEMORY_FRACTION,
        "backend": manifest["backend"]["kind"],
        "routing_inputs": (
            "18D descriptor only"
            if router.feature_mode == "none"
            else "18D descriptor + current-batch mean/std/count only"
        ),
        "forbidden_routing_inputs": ["corruption/domain name", "class label", "backend logits", "correctness"],
        "abstention": "identity when UtilityRouter returns -1",
        "stream_order_sha256": order_hash,
        "stream_keys_sha256": sequence_sha256([str(rows[int(i)]["key"]) for i in order]),
        "mint_state": "one continuous adapter over the complete mixed routed stream",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("output protocol changed; use a new output directory")
    atomic_json(protocol_path, protocol)
    protocol_hash = sha256_file(protocol_path)

    feature_provider = FeatureProvider(manifest)
    if feature_provider.kind != "online_risks":
        stream_keys = {str(row["key"]) for row in rows}
        if set(feature_provider.by_key) != stream_keys:
            raise RuntimeError("precomputed feature keys do not exactly match the frozen mixed stream")
    images = StreamImages()
    if mock:
        classes = int(manifest["backend"].get("mock_classes", 10))
        backend: Any = MockBackend(manifest["backend"]["kind"], classes)
        experts: Any = None
    else:
        import torch

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real routed replay; use --mock for CPU plumbing tests")
        torch.cuda.memory.set_per_process_memory_fraction(MEMORY_FRACTION)
        backend = make_real_backend(manifest)
        experts = RealExperts(manifest, torch.device("cuda"))

    start_batch, records, resumed_state = resume_progress(
        output_dir, protocol_hash, mock=mock, order=order, rows=rows
    )
    if resumed_state is not None:
        if mock:
            backend.load_state(resumed_state["backend"])
        else:
            if backend[2] is not None:
                restore_adapter(backend[2], resumed_state["adapter"])
            restore_random_snapshot(resumed_state["random"])

    batches = (len(order) + BATCH_SIZE - 1) // BATCH_SIZE
    index_base = Path(manifest["_index_path"]).parent
    for batch_index in range(start_batch, batches):
        if stop_after_batches is not None and batch_index >= stop_after_batches:
            return None
        positions = order[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
        batch_rows = [rows[int(position)] for position in positions]
        batch_images = [images.load(index_base, row["image"]) for row in batch_rows]
        raw_hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in batch_images]
        for row, raw_hash in zip(batch_rows, raw_hashes):
            expected = row.get("raw_sha256")
            if expected is not None and str(expected) != raw_hash:
                raise RuntimeError(f"raw-image hash mismatch for stream key {row['key']!r}")
        descriptors = np.stack([
            feature_provider.descriptor(str(row["key"]), image, raw_hash)
            for row, image, raw_hash in zip(batch_rows, batch_images, raw_hashes)
        ])
        decisions, predictions = route_batch(router, descriptors)
        # Experts are applied only after the whole batch has been routed.
        if mock:
            restored = apply_mock_experts(batch_images, decisions, manifest["expert_bank"])
            predicted_classes = backend(restored)
        else:
            restored = experts.apply(batch_images, decisions)
            predicted_classes = real_predict(restored, backend)

        for local, (row, raw_hash, decision, prediction) in enumerate(
            zip(batch_rows, raw_hashes, decisions, predicted_classes)
        ):
            expert = "identity" if decision < 0 else expert_names[int(decision)]
            operator = "identity" if decision < 0 else manifest["expert_bank"][int(decision)].get(
                "operator", manifest["expert_bank"][int(decision)]["kind"]
            )
            target = int(row["target"])
            records.append({
                "key": str(row["key"]),
                "sample": row["sample"],
                "target": target,
                "prediction": int(prediction),
                "correct": int(prediction) == target,
                "stream_position": len(records),
                "batch": batch_index,
                "batch_position": local,
                "decision": "abstain" if decision < 0 else "route",
                "expert_index": int(decision),
                "expert": expert,
                "operator": operator,
                "predicted_utility": None if decision < 0 else float(predictions[local, int(decision)]),
                "predicted_utilities": [None if not np.isfinite(value) else float(value)
                                        for value in predictions[local]],
                "descriptor_sha256": hashlib.sha256(
                    np.asarray(descriptors[local], dtype="<f8").tobytes()
                ).hexdigest(),
                "raw_sha256": raw_hash,
                "protocol_sha256": protocol_hash,
                # Audit metadata is copied only after inference has completed.
                "audit_domain": row.get("audit_domain"),
            })
        if mock:
            state = {"backend": backend.state()}
        else:
            state = {
                "adapter": None if backend[2] is None else adapter_snapshot(backend[2]),
                "random": random_snapshot(),
            }
        save_progress(output_dir, next_batch=batch_index + 1, records=records, state=state, mock=mock)

    environment = {
        "mode": "mock" if mock else "cuda",
        "python": platform.python_version(),
    }
    if not mock:
        import torch

        environment.update({
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
        })
    result = {
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "records": records,
        "records_sha256": canonical_sha256(records),
        "accuracy": 100.0 * sum(bool(row["correct"]) for row in records) / len(records),
        "expert_counts": {
            name: sum(row["expert"] == name for row in records)
            for name in ("identity",) + expert_names
        },
        "environment": environment,
    }
    atomic_json(output_dir / "result.json", result)
    atomic_json(output_dir / "complete.json", {
        "result_sha256": sha256_file(output_dir / "result.json"),
        "records_sha256": result["records_sha256"],
        "protocol_sha256": protocol_hash,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mock", action="store_true", help="CPU plumbing test; never scientific evidence")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--stop-after-batches", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.validate_only:
        validated = validate_manifest(args.manifest.resolve(), mock=args.mock)
        print(json.dumps({"validated": True, "status": validated["status"], "mock": args.mock}))
        return
    result = run(
        args.manifest.resolve(), args.output_dir.resolve(), mock=args.mock,
        stop_after_batches=args.stop_after_batches,
    )
    if result is None:
        print(json.dumps({"status": "intentionally_interrupted_for_resume_test"}))
    else:
        print(json.dumps({
            "status": result["protocol"]["status"],
            "accuracy": result["accuracy"],
            "records": len(result["records"]),
            "result": str((args.output_dir / "result.json").resolve()),
        }))


if __name__ == "__main__":
    main()
