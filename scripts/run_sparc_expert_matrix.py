#!/usr/bin/env python3
"""Run a complete, matched stateless-CLIP expert-by-scenario matrix.

The JSON manifest is the only place where experts and scenarios are selected.
This runner intentionally produces development utility evidence, not a final
router decision or confirmation result.  Every expert is paired with identity
on the same shuffled stream and the same cached uint8 corruption bytes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SEED = 1109
MEMORY_FRACTION = 0.68
SUPPORTED_TYPES = {
    "identity",
    "fixed",
    "conditional_restorer",
    "snow_specialist",
    "brightness_specialist",
}
RESULT_FIELDS = (
    "sample",
    "target",
    "batch",
    "stream_position",
    "raw_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def stratified_subset(
    relative_paths: list[str], n: int | None, seed: int, offset: int = 0
) -> list[str]:
    """Mirror streammint.data.stratified_subset without importing GPU code."""
    paths = [value.strip() for value in relative_paths if value.strip()]
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if n is None:
        if offset:
            raise ValueError("offset requires n")
        return sorted(paths)
    if n <= 0 or offset + n > len(paths):
        raise ValueError("invalid n/offset for ID source")
    if n >= len(paths) and offset == 0:
        return sorted(paths)
    by_class: dict[str, list[str]] = {}
    for path in paths:
        by_class.setdefault(path.split("/", 1)[0], []).append(path)
    rng = np.random.default_rng(seed)
    for values in by_class.values():
        rng.shuffle(values)
    classes = sorted(by_class)
    rng.shuffle(classes)
    selected: list[str] = []
    cursor = 0
    while len(selected) < offset + n:
        class_name = classes[cursor % len(classes)]
        row = cursor // len(classes)
        if row < len(by_class[class_name]):
            selected.append(by_class[class_name][row])
        cursor += 1
    return selected[offset:offset + n]


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def scenario_key(scenario: str) -> str:
    name, severity_text = scenario.rsplit(":", 1)
    severity = int(severity_text)
    if severity < 1:
        raise ValueError(f"invalid severity in scenario {scenario!r}")
    return f"{name}_s{severity}"


def result_path(output_dir: Path, expert_name: str, scenario: str, n: int) -> Path:
    key = scenario_key(scenario).replace("/", "-")
    return output_dir / f"ViT-L-14_clip_{expert_name}_{key}_n{n}_seed{SEED}.json"


def load_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.resolve()
    manifest = json.loads(path.read_text())
    base = path.parent
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest.schema_version must be 1")
    if int(manifest.get("seed", -1)) != SEED:
        raise ValueError(f"expert matrix requires seed {SEED}")
    if manifest.get("backend") != "stateless_clip":
        raise ValueError("manifest.backend must be 'stateless_clip'")
    if manifest.get("status") != "development":
        raise ValueError("this runner only accepts status='development'")
    if float(manifest.get("memory_fraction_limit", -1)) != MEMORY_FRACTION:
        raise ValueError(f"memory_fraction_limit must be {MEMORY_FRACTION}")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("manifest.scenarios must be a non-empty list")
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("manifest.scenarios contains duplicates")
    for scenario in scenarios:
        scenario_key(str(scenario))
    experts = manifest.get("experts")
    if not isinstance(experts, list) or not experts:
        raise ValueError("manifest.experts must be a non-empty list")
    names: list[str] = []
    identity_count = 0
    for expert in experts:
        if not isinstance(expert, dict):
            raise ValueError("every expert must be an object")
        name = str(expert.get("name", ""))
        kind = expert.get("type")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"unsafe or empty expert name: {name!r}")
        if kind not in SUPPORTED_TYPES:
            raise ValueError(f"unsupported expert type {kind!r}")
        names.append(name)
        if kind == "identity":
            identity_count += 1
            if set(expert) != {"name", "type"}:
                raise ValueError("identity accepts only name and type")
        elif kind == "fixed":
            if not isinstance(expert.get("operator"), str) or not expert["operator"]:
                raise ValueError(f"fixed expert {name!r} requires operator")
        elif kind == "conditional_restorer":
            if "checkpoint" not in expert:
                raise ValueError(f"conditional expert {name!r} requires checkpoint")
            if not isinstance(expert.get("condition"), str) or not expert["condition"]:
                raise ValueError(f"conditional expert {name!r} requires a fixed condition")
            scale = float(expert.get("residual_scale", 1.0))
            if not np.isfinite(scale) or scale < 0.0:
                raise ValueError(f"invalid residual_scale for {name!r}")
        elif kind == "snow_specialist":
            if "checkpoint" not in expert:
                raise ValueError(f"snow specialist {name!r} requires checkpoint")
        elif kind == "brightness_specialist":
            if "checkpoint" not in expert:
                raise ValueError(f"brightness specialist {name!r} requires checkpoint")
            scale = float(expert.get("residual_scale", 1.0))
            if not np.isfinite(scale) or not 0.0 <= scale <= 2.0:
                raise ValueError(f"invalid residual_scale for {name!r}")
            for field in ("checkpoint_sha256", "source_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", str(expert.get(field, ""))):
                    raise ValueError(
                        f"brightness specialist {name!r} requires a lowercase SHA-256 {field}"
                    )
    if len(set(names)) != len(names):
        raise ValueError("expert names must be unique")
    if identity_count != 1:
        raise ValueError("exactly one identity expert is required")
    required_paths = ("ids", "cache", "meta", "clip_checkpoint")
    for field in required_paths:
        if field not in manifest:
            raise ValueError(f"manifest.{field} is required")
        if not resolve(base, manifest[field]).exists():
            raise FileNotFoundError(f"manifest.{field} does not exist")
    return manifest, base


def checkpoint_hashes(manifest: dict[str, Any], base: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for expert in manifest["experts"]:
        if "checkpoint" in expert:
            path = resolve(base, expert["checkpoint"])
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint does not exist: {path}")
            actual = sha256(path)
            if expert["type"] == "brightness_specialist" and actual != expert["checkpoint_sha256"]:
                raise RuntimeError(f"brightness specialist checkpoint hash mismatch: {path}")
            hashes[str(path)] = actual
    return hashes


def validate_checkpoint_metadata(
    manifest: dict[str, Any], base: Path
) -> dict[str, dict[str, Any]]:
    """Validate lightweight checkpoint metadata on CPU before any GPU work."""
    metadata: dict[str, dict[str, Any]] = {}
    for expert in manifest["experts"]:
        if "checkpoint" not in expert:
            continue
        path = resolve(base, expert["checkpoint"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint.get("model"), dict) or "state_dict" not in checkpoint:
            raise RuntimeError(f"invalid checkpoint schema: {path}")
        if int(checkpoint.get("seed", -1)) != SEED:
            raise RuntimeError(f"checkpoint was not trained with seed {SEED}: {path}")
        if expert["type"] == "conditional_restorer":
            scenarios = list(checkpoint.get("scenarios", []))
            if expert["condition"] not in scenarios:
                raise RuntimeError(
                    f"condition {expert['condition']!r} is absent from checkpoint {path}"
                )
            metadata[expert["name"]] = {
                "format": "ConditionalResidualRestorer",
                "condition": expert["condition"],
                "checkpoint_conditions": scenarios,
            }
        elif expert["type"] == "snow_specialist":
            if checkpoint.get("format") != "sparc-snow-specialist-v1":
                raise RuntimeError(f"unsupported snow specialist checkpoint: {path}")
            metadata[expert["name"]] = {"format": checkpoint["format"]}
        else:
            if checkpoint.get("format") != "sparc-brightness-clipping-specialist-v1":
                raise RuntimeError(f"unsupported brightness specialist checkpoint: {path}")
            if checkpoint.get("uses_imagenet_class_labels") is not False:
                raise RuntimeError(
                    f"brightness specialist supervision ledger is missing: {path}"
                )
            source = ROOT / "src/streammint/brightness_specialist.py"
            if sha256(source) != expert["source_sha256"]:
                raise RuntimeError("brightness specialist source hash mismatch")
            metadata[expert["name"]] = {
                "format": checkpoint["format"],
                "residual_scale": float(expert.get("residual_scale", 1.0)),
                "source_sha256": expert["source_sha256"],
            }
        del checkpoint
    return metadata


def validate_cache(
    manifest: dict[str, Any], base: Path, ids: list[str]
) -> tuple[dict[str, Any], dict[str, str], Path]:
    cache = resolve(base, manifest["cache"])
    protocol_path = cache / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    expected_ids = sequence_sha256(ids)
    if int(protocol.get("n", -1)) != len(ids):
        raise RuntimeError("cache length does not match manifest IDs")
    if protocol.get("ids_sha256") != expected_ids:
        raise RuntimeError("cache ID sequence does not match manifest IDs")
    if int(protocol.get("subset_seed", -1)) != SEED:
        raise RuntimeError(f"cache subset_seed must be {SEED}")
    if int(protocol.get("corruption_seed", -1)) != SEED:
        raise RuntimeError(f"cache corruption_seed must be {SEED}")
    file_hashes: dict[str, str] = {}
    for scenario in manifest["scenarios"]:
        key = scenario_key(scenario)
        entry = protocol.get("files", {}).get(key)
        if not isinstance(entry, dict):
            raise RuntimeError(f"cache is missing scenario {scenario!r}")
        path = cache / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"cache array does not exist: {path}")
        actual = sha256(path)
        if actual != entry.get("sha256"):
            raise RuntimeError(f"cache array hash mismatch: {path}")
        array = np.load(path, mmap_mode="r")
        if array.shape != (len(ids), 224, 224, 3) or array.dtype != np.uint8:
            raise RuntimeError(f"invalid cache array schema: {path}")
        file_hashes[scenario] = actual
    return protocol, file_hashes, cache


def build_protocol(
    manifest_path: Path, manifest: dict[str, Any], base: Path
) -> tuple[dict[str, Any], list[str], Path]:
    ids_path = resolve(base, manifest["ids"])
    source_ids = [line for line in ids_path.read_text().splitlines() if line]
    if not source_ids or len(set(source_ids)) != len(source_ids):
        raise ValueError("IDs must be non-empty and unique")
    requested_n = manifest.get("n")
    ids = stratified_subset(
        source_ids,
        None if requested_n is None else int(requested_n),
        SEED,
        int(manifest.get("offset", 0)),
    )
    cache_protocol, cache_file_hashes, cache = validate_cache(manifest, base, ids)
    checkpoints = checkpoint_hashes(manifest, base)
    checkpoint_metadata = validate_checkpoint_metadata(manifest, base)
    source_paths = {
        "scripts/run_sparc_expert_matrix.py": Path(__file__).resolve(),
        "src/streammint/data.py": ROOT / "src/streammint/data.py",
        "src/streammint/methods.py": ROOT / "src/streammint/methods.py",
    }
    if any(expert["type"] == "conditional_restorer" for expert in manifest["experts"]):
        source_paths["src/streammint/conditional_restorer.py"] = (
            ROOT / "src/streammint/conditional_restorer.py"
        )
    if any(expert["type"] == "snow_specialist" for expert in manifest["experts"]):
        source_paths["src/streammint/snow_specialist.py"] = ROOT / "src/streammint/snow_specialist.py"
    if any(expert["type"] == "brightness_specialist" for expert in manifest["experts"]):
        source_paths["src/streammint/brightness_specialist.py"] = (
            ROOT / "src/streammint/brightness_specialist.py"
        )
    normalized_experts = []
    for expert in manifest["experts"]:
        normalized = dict(expert)
        if "checkpoint" in normalized:
            normalized["checkpoint"] = str(resolve(base, normalized["checkpoint"]))
        normalized_experts.append(normalized)
    protocol = {
        "schema_version": 1,
        "status": "development_expert_utility_matrix_not_confirmation_evidence",
        "backend": "stateless_clip",
        "seed": SEED,
        "subset_seed": SEED,
        "corruption_seed": SEED,
        "ids_source": str(ids_path),
        "ids_source_sha256": sha256(ids_path),
        "ids_sha256": sequence_sha256(ids),
        "n": len(ids),
        "offset": int(manifest.get("offset", 0)),
        "cache": str(cache),
        "cache_protocol_sha256": sha256(cache / "protocol.json"),
        "cache_source_sha256": cache_protocol.get("source_sha256", {}),
        "cache_file_sha256": cache_file_hashes,
        "scenarios": manifest["scenarios"],
        "experts": normalized_experts,
        "complete_cross_product": True,
        "batch_size": int(manifest.get("batch_size", 20)),
        "workers": int(manifest.get("workers", 8)),
        "memory_fraction_limit": MEMORY_FRACTION,
        "clip_checkpoint": str(resolve(base, manifest["clip_checkpoint"])),
        "clip_checkpoint_sha256": sha256(resolve(base, manifest["clip_checkpoint"])),
        "meta": str(resolve(base, manifest["meta"])),
        "meta_sha256": sha256(resolve(base, manifest["meta"])),
        "checkpoint_sha256": checkpoints,
        "checkpoint_metadata": checkpoint_metadata,
        "manifest_source": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path.resolve()),
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "record_schema": list(RESULT_FIELDS),
        "utility_target": "expert correctness minus paired identity correctness",
        "evaluation_labels": "ImageNet labels are used only to score predictions",
    }
    if protocol["batch_size"] < 1 or protocol["workers"] < 0:
        raise ValueError("batch_size must be positive and workers non-negative")
    return protocol, ids, cache


def run_protocol(protocol: dict[str, Any], expert: dict[str, Any], scenario: str) -> dict[str, Any]:
    return {
        "protocol_sha256": canonical_sha256(protocol),
        "backend": "stateless_clip",
        "expert": expert["name"],
        "expert_spec_sha256": canonical_sha256(expert),
        "scenario": scenario,
        "seed": SEED,
        "cache_file_sha256": protocol["cache_file_sha256"][scenario],
    }


def validate_records(
    payload: dict[str, Any], expected_protocol: dict[str, Any], n: int,
    identity_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if payload.get("protocol") != expected_protocol:
        raise RuntimeError("result protocol mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != n:
        raise RuntimeError("result record count mismatch")
    seen: set[int] = set()
    for position, row in enumerate(records):
        for field in RESULT_FIELDS:
            if field not in row:
                raise RuntimeError(f"result is missing record field {field!r}")
        sample = int(row["sample"])
        if sample in seen or not (0 <= sample < n):
            raise RuntimeError("duplicate or invalid sample index")
        seen.add(sample)
        if int(row["stream_position"]) != position:
            raise RuntimeError("non-contiguous stream_position")
        raw_hash = str(row["raw_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", raw_hash):
            raise RuntimeError("invalid raw_sha256")
        if identity_records is not None:
            identity = identity_records[position]
            for field in RESULT_FIELDS:
                if row[field] != identity[field]:
                    raise RuntimeError(f"expert/identity mismatch for {field}")
    expected_accuracy = 100.0 * sum(bool(row.get("correct")) for row in records) / n
    if abs(float(payload.get("accuracy", float("nan"))) - expected_accuracy) > 1e-12:
        raise RuntimeError("stored accuracy does not match records")
    return records


class CachedScenarioDataset(Dataset):
    def __init__(
        self, cache: Path, cache_protocol: dict[str, Any], scenario: str,
        ids: list[str], classes: dict[str, int], operator: str | None,
    ):
        entry = cache_protocol["files"][scenario_key(scenario)]
        self.images = np.load(cache / entry["path"], mmap_mode="r")
        self.ids = ids
        self.classes = classes
        self.operator = operator

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        array = np.asarray(self.images[index]).copy()
        raw_hash = hashlib.sha256(array.tobytes()).hexdigest()
        if self.operator is not None:
            sys.path.insert(0, str(ROOT / "src"))
            from streammint.data import preprocess_image
            image = preprocess_image(Image.fromarray(array, mode="RGB"), self.operator)
            array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        target = self.classes[self.ids[index].split("/", 1)[0]]
        return tensor, target, index, raw_hash


def load_expert(expert: dict[str, Any], base: Path, device: str):
    kind = expert["type"]
    if kind in {"identity", "fixed"}:
        return None, None
    checkpoint_path = resolve(base, expert["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sys.path.insert(0, str(ROOT / "src"))
    if kind == "conditional_restorer":
        from streammint.conditional_restorer import ConditionalResidualRestorer
        model = ConditionalResidualRestorer(**checkpoint["model"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        metadata = {"scenarios": list(checkpoint["scenarios"])}
    elif kind == "snow_specialist":
        from streammint.snow_specialist import SnowSpecialist
        if checkpoint.get("format") != "sparc-snow-specialist-v1":
            raise RuntimeError("unsupported snow specialist checkpoint format")
        model = SnowSpecialist(**checkpoint["model"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        metadata = {}
    elif kind == "brightness_specialist":
        from streammint.brightness_specialist import BrightnessClippingSpecialist
        if checkpoint.get("format") != "sparc-brightness-clipping-specialist-v1":
            raise RuntimeError("unsupported brightness specialist checkpoint format")
        if sha256(checkpoint_path) != expert["checkpoint_sha256"]:
            raise RuntimeError("brightness specialist checkpoint hash changed after validation")
        source = ROOT / "src/streammint/brightness_specialist.py"
        if sha256(source) != expert["source_sha256"]:
            raise RuntimeError("brightness specialist source hash changed after validation")
        model = BrightnessClippingSpecialist(**checkpoint["model"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        metadata = {}
    else:  # pragma: no cover - guarded by manifest validation
        raise AssertionError(kind)
    model.eval()
    return model, metadata


def apply_expert(
    images: torch.Tensor, expert: dict[str, Any], model, metadata,
    scenario: str,
) -> torch.Tensor:
    kind = expert["type"]
    if kind in {"identity", "fixed"}:
        return images
    with torch.no_grad(), torch.autocast(
        device_type=images.device.type, enabled=images.is_cuda
    ):
        if kind in {"snow_specialist", "brightness_specialist"}:
            restored = model(images)
            if kind == "brightness_specialist":
                scale = float(expert.get("residual_scale", 1.0))
                restored = (images + scale * (restored - images)).clamp(0.0, 1.0)
            return restored
        condition_name = expert["condition"]
        if condition_name not in metadata["scenarios"]:
            raise RuntimeError(
                f"conditional expert {expert['name']!r} has no condition for {scenario!r}"
            )
        condition = metadata["scenarios"].index(condition_name)
        condition_ids = torch.full(
            (len(images),), condition, device=images.device, dtype=torch.long
        )
        restored = model(images, condition_ids)
        scale = float(expert.get("residual_scale", 1.0))
        return (images + scale * (restored - images)).clamp(0.0, 1.0)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validate-only", action="store_true",
        help="validate manifest/cache/checkpoint hashes without importing CLIP or using GPU",
    )
    args = parser.parse_args()

    manifest, base = load_manifest(args.manifest)
    protocol, ids, cache = build_protocol(args.manifest, manifest, base)
    output_dir = args.output_dir.resolve()
    if args.validate_only:
        print(json.dumps({
            "valid": True,
            "protocol_sha256": canonical_sha256(protocol),
            "jobs": len(protocol["scenarios"]) * len(protocol["experts"]),
            "n": len(ids),
        }, indent=2))
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; refusing silent CPU fallback")
    torch.cuda.memory.set_per_process_memory_fraction(MEMORY_FRACTION)

    sys.path.insert(0, str(ROOT / "src"))
    import clip
    from streammint.data import CLIP_MEAN, CLIP_STD
    from streammint.methods import build_text_weights

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if previous != protocol:
            raise RuntimeError("protocol/source/checkpoint hash changed; use a new output directory")
    else:
        write_json_atomic(protocol_path, protocol)

    meta, _ = torch.load(resolve(base, manifest["meta"]), weights_only=False)
    classes = {wnid: index for index, wnid in enumerate(sorted(meta))}
    class_names = [meta[wnid][0] for wnid in sorted(meta)]
    clip_model, _ = clip.load(protocol["clip_checkpoint"], device="cuda")
    clip_model.eval()
    text = build_text_weights(clip_model, class_names)
    mean = torch.tensor(CLIP_MEAN, device="cuda")[None, :, None, None]
    std = torch.tensor(CLIP_STD, device="cuda")[None, :, None, None]
    cache_protocol = json.loads((cache / "protocol.json").read_text())

    identity = next(expert for expert in manifest["experts"] if expert["type"] == "identity")
    ordered_experts = [identity] + [
        expert for expert in manifest["experts"] if expert["type"] != "identity"
    ]
    identity_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for expert in ordered_experts:
        model, metadata = load_expert(expert, base, "cuda")
        for scenario in manifest["scenarios"]:
            output = result_path(output_dir, expert["name"], scenario, len(ids))
            expected = run_protocol(protocol, {
                **expert,
                **({"checkpoint": str(resolve(base, expert["checkpoint"]))}
                   if "checkpoint" in expert else {}),
            }, scenario)
            identity_records = identity_by_scenario.get(scenario)
            if output.exists():
                payload = json.loads(output.read_text())
                records = validate_records(payload, expected, len(ids), identity_records)
                if expert["type"] == "identity":
                    identity_by_scenario[scenario] = records
                print(f"verified skip {output.name}", flush=True)
                continue

            operator = expert.get("operator") if expert["type"] == "fixed" else None
            dataset = CachedScenarioDataset(
                cache, cache_protocol, scenario, ids, classes, operator
            )
            generator = torch.Generator().manual_seed(SEED)
            loader = DataLoader(
                dataset,
                batch_size=protocol["batch_size"],
                shuffle=True,
                num_workers=protocol["workers"],
                generator=generator,
                pin_memory=True,
            )
            records: list[dict[str, Any]] = []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for batch_index, (images, labels, samples, raw_hashes) in enumerate(loader):
                images = images.cuda(non_blocking=True)
                images = apply_expert(images, expert, model, metadata, scenario)
                normalized = (images - mean) / std
                with torch.no_grad(), torch.autocast("cuda"):
                    features = F.normalize(clip_model.encode_image(normalized).float(), dim=-1)
                    predictions = (100.0 * features @ text.T).argmax(-1).cpu().tolist()
                for row, prediction in enumerate(predictions):
                    target = int(labels[row])
                    records.append({
                        "sample": int(samples[row]),
                        "target": target,
                        "prediction": prediction,
                        "correct": prediction == target,
                        "scenario": scenario,
                        "batch": batch_index,
                        "stream_position": len(records),
                        "raw_sha256": str(raw_hashes[row]),
                    })
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            payload = {
                "protocol": expected,
                "accuracy": 100.0 * sum(row["correct"] for row in records) / len(records),
                "records": records,
                "timing": {
                    "seconds": elapsed,
                    "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "memory_fraction_limit": MEMORY_FRACTION,
                },
                "environment": {
                    "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "python": platform.python_version(),
                },
            }
            validate_records(payload, expected, len(ids), identity_records)
            write_json_atomic(output, payload)
            if expert["type"] == "identity":
                identity_by_scenario[scenario] = records
            print(json.dumps({
                "expert": expert["name"], "scenario": scenario,
                "accuracy": payload["accuracy"], "seconds": round(elapsed, 2),
                "peak_gpu_gib": round(payload["timing"]["peak_gpu_gib"], 3),
            }), flush=True)
            del dataset, loader
        del model, metadata
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
