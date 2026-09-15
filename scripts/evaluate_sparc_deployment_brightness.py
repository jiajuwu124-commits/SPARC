#!/usr/bin/env python3
"""Matched CLIP/Mint evaluation for the deployment brightness restorer."""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streammint.data import CLIP_MEAN, CLIP_STD, stratified_subset  # noqa: E402
from streammint.deployment_brightness import DeploymentBrightnessRestorer  # noqa: E402
from streammint.methods import StreamingAdapter, build_text_weights  # noqa: E402


SEED = 1109
B_IDS = ROOT / "configs/sparc_final_confirmation1k_ids.txt"
B_CACHE = ROOT / "cache/sparc_diagnostic1k_b_corruptions_s5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def comparison_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return row["sample"], row["target"], row["batch"], row["stream_position"], row["raw_sha256"]


class BrightnessDataset(Dataset):
    def __init__(self, path: Path, ids: list[str], class_to_index: dict[str, int]):
        self.images = np.load(path, mmap_mode="r")
        self.ids = ids
        self.class_to_index = class_to_index

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        array = np.asarray(self.images[index]).copy()
        return (
            torch.from_numpy(array).permute(2, 0, 1).float() / 255,
            self.class_to_index[self.ids[index].split("/", 1)[0]], index,
            hashlib.sha256(array.tobytes()).hexdigest(),
        )


def validate(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str], Path, DeploymentBrightnessRestorer]:
    if args.seed != SEED or args.batch_size != 20:
        raise ValueError("evaluation requires seed 1109 and batch size 20")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "sparc-deployment-supervised-brightness-v1":
        raise RuntimeError("unsupported deployment brightness checkpoint")
    if checkpoint.get("seed") != SEED:
        raise RuntimeError("checkpoint seed mismatch")
    if checkpoint.get("uses_imagenet_class_labels_during_training") is not True:
        raise RuntimeError("checkpoint supervision ledger is missing")
    if checkpoint.get("inference_requires_class_labels") is not False:
        raise RuntimeError("checkpoint does not guarantee label-free inference")
    if checkpoint.get("inference_requires_corruption_identity") is not False:
        raise RuntimeError("checkpoint does not guarantee degradation-ID-free inference")
    training_protocol = json.loads(args.training_protocol.read_text())
    if training_protocol.get("status") != "redesign_complete_not_final_tested":
        raise RuntimeError("training protocol is not frozen redesign evidence")
    if training_protocol.get("checkpoint_sha256") != sha256(args.checkpoint):
        raise RuntimeError("checkpoint/training-protocol hash mismatch")
    if args.split == "redesign_validation_b":
        if args.ids.resolve() != B_IDS.resolve() or args.cache.resolve() != B_CACHE.resolve():
            raise RuntimeError("validation_b must use the frozen diagnostic-B ID/cache pair")
        final_manifest = None
    else:
        if args.final_freeze_manifest is None:
            raise RuntimeError("untouched final requires an explicit post-redesign freeze manifest")
        final_manifest = json.loads(args.final_freeze_manifest.read_text())
        if final_manifest.get("status") != "authorized_untouched_final":
            raise RuntimeError("final split has not been explicitly authorized/frozen")
        if final_manifest.get("checkpoint_sha256") != sha256(args.checkpoint):
            raise RuntimeError("final freeze manifest/checkpoint mismatch")
        if final_manifest.get("ids_sha256") != sha256(args.ids):
            raise RuntimeError("final freeze manifest/ID-file mismatch")
    all_ids = [line for line in args.ids.read_text().splitlines() if line]
    ids = stratified_subset(all_ids, args.n, SEED) if args.n else all_ids
    protocol_path = args.cache / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if final_manifest is not None and final_manifest.get("cache_protocol_sha256") != sha256(protocol_path):
        raise RuntimeError("final freeze manifest/cache-protocol mismatch")
    if protocol.get("n") != len(ids) or protocol.get("ids_sha256") != sequence_sha256(ids):
        raise RuntimeError("evaluation cache/ID sequence mismatch")
    if protocol.get("subset_seed") != SEED or protocol.get("corruption_seed") != SEED:
        raise RuntimeError("evaluation cache seed mismatch")
    entry = protocol.get("files", {}).get("brightness_s5", {})
    array_path = args.cache / str(entry.get("path", ""))
    if sha256(array_path) != entry.get("sha256"):
        raise RuntimeError("evaluation brightness array hash mismatch")
    if final_manifest is not None and final_manifest.get("brightness_array_sha256") != sha256(array_path):
        raise RuntimeError("final freeze manifest/brightness-array mismatch")
    images = np.load(array_path, mmap_mode="r")
    if images.shape != (len(ids), 224, 224, 3) or images.dtype != np.uint8:
        raise RuntimeError("evaluation brightness array schema mismatch")
    model = DeploymentBrightnessRestorer(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if list(inspect.signature(model.forward).parameters) != ["images", "return_aux"]:
        raise RuntimeError("restorer inference API is not image-only")
    with torch.no_grad():
        output = model(torch.rand(1, 3, 64, 64))
    if output.shape != (1, 3, 64, 64) or not torch.isfinite(output).all():
        raise RuntimeError("CPU inference smoke failed")
    ledger = {
        "schema_version": 1, "status": args.split, "scenario": "brightness:5",
        "seed": SEED, "batch_size": 20, "n": len(ids),
        "ids_source": str(args.ids.resolve()), "ids_source_sha256": sha256(args.ids),
        "ids_sequence_sha256": sequence_sha256(ids),
        "cache_protocol": str(protocol_path.resolve()),
        "cache_protocol_sha256": sha256(protocol_path),
        "brightness_array": str(array_path.resolve()), "brightness_array_sha256": sha256(array_path),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "training_protocol": str(args.training_protocol.resolve()),
        "training_protocol_sha256": sha256(args.training_protocol),
        "clip_checkpoint_sha256": sha256(args.clip_checkpoint),
        "meta_sha256": sha256(args.meta),
        "evaluation_labels_used_only_after_image_only_restoration": True,
        "restorer_inputs": ["corrupted_rgb_image"],
        "final_freeze_manifest_sha256": (
            sha256(args.final_freeze_manifest) if args.final_freeze_manifest else None
        ),
    }
    return ledger, ids, array_path, model


def set_determinism() -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("redesign_validation_b", "untouched_final"), required=True)
    parser.add_argument("--final-freeze-manifest", type=Path)
    parser.add_argument("--backends", default="clip,mint")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--meta", type=Path, default=ROOT.parent / "datasets/imagenet/meta.bin")
    parser.add_argument("--clip-checkpoint", type=Path, default=ROOT.parent / ".cache/clip/ViT-L-14.pt")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    ledger, ids, array_path, restorer = validate(args)
    if args.validate_only:
        print(json.dumps({**ledger, "cpu_inference_smoke_pass": True}, indent=2))
        return
    set_determinism()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required after CPU validate-only")
    torch.cuda.memory.set_per_process_memory_fraction(0.68)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    meta, _ = torch.load(args.meta, map_location="cpu", weights_only=False)
    wnids = sorted(meta)
    class_to_index = {wnid: index for index, wnid in enumerate(wnids)}
    clip_model, _ = clip.load(str(args.clip_checkpoint), device="cuda")
    clip_model.eval()
    text = build_text_weights(clip_model, [meta[wnid][0] for wnid in wnids])
    restorer = restorer.cuda().eval()
    mean = torch.tensor(CLIP_MEAN, device="cuda")[None, :, None, None]
    std = torch.tensor(CLIP_STD, device="cuda")[None, :, None, None]
    dataset = BrightnessDataset(array_path, ids, class_to_index)
    backends = [value for value in args.backends.split(",") if value]
    if not backends or any(value not in {"clip", "mint"} for value in backends):
        raise ValueError("backends must be a non-empty subset of clip,mint")
    saved: dict[str, dict[str, Path]] = {}
    protocol = {
        **ledger, "backends": backends, "methods": ["identity", "deployment_brightness"],
        "memory_fraction_limit": 0.68,
        "source_sha256": {
            "scripts/evaluate_sparc_deployment_brightness.py": sha256(Path(__file__)),
            "src/streammint/deployment_brightness.py": sha256(ROOT / "src/streammint/deployment_brightness.py"),
        },
    }
    atomic_json(args.output_dir / "protocol.json", protocol)
    for backend in backends:
        saved[backend] = {}
        for method in ("identity", "deployment_brightness"):
            adapter = (
                StreamingAdapter(clip_model, text, method="mint", lr=0.015, prior=10000.0)
                if backend == "mint" else None
            )
            loader = DataLoader(
                dataset, batch_size=20, shuffle=True, num_workers=args.workers,
                generator=torch.Generator().manual_seed(SEED), pin_memory=True,
            )
            records = []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for batch, (images, labels, samples, hashes) in enumerate(loader):
                images = images.cuda(non_blocking=True)
                # Labels are deliberately not moved until restoration is complete.
                if method == "deployment_brightness":
                    with torch.no_grad(), torch.autocast("cuda"):
                        images = restorer(images)
                normalized = (images - mean) / std
                if adapter is None:
                    with torch.no_grad(), torch.autocast("cuda"):
                        features = F.normalize(clip_model.encode_image(normalized).float(), dim=-1)
                        logits = 100.0 * features @ text.T
                else:
                    logits = adapter(normalized)
                predictions = logits.argmax(-1).cpu().tolist()
                for local, prediction in enumerate(predictions):
                    target = int(labels[local])
                    records.append({
                        "sample": int(samples[local]), "target": target, "prediction": prediction,
                        "correct": prediction == target, "batch": batch,
                        "stream_position": len(records), "raw_sha256": hashes[local],
                        "scenario": "brightness:5",
                    })
            torch.cuda.synchronize()
            result = {
                "protocol_sha256": sha256(args.output_dir / "protocol.json"),
                "backend": backend, "method": method,
                "accuracy": 100.0 * sum(row["correct"] for row in records) / len(records),
                "comparison_key_sha256": hashlib.sha256(json.dumps(
                    [comparison_key(row) for row in records], separators=(",", ":")
                ).encode()).hexdigest(),
                "records": records,
                "timing": {"seconds": time.perf_counter() - started,
                           "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30},
                "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                                "python": platform.python_version()},
            }
            path = args.output_dir / f"ViT-L-14_{backend}_{method}_brightnesss5_n{len(ids)}_seed1109.json"
            atomic_json(path, result)
            saved[backend][method] = path
            print(json.dumps({"backend": backend, "method": method,
                              "accuracy": result["accuracy"]}), flush=True)
            if adapter is not None:
                adapter.reset()
            del adapter, loader
            gc.collect()
            torch.cuda.empty_cache()
    audit = {"schema_version": 1, "passed": True, "backends": {}}
    for backend, paths in saved.items():
        identity = json.loads(paths["identity"].read_text())
        reference = [comparison_key(row) for row in identity["records"]]
        audit["backends"][backend] = {}
        for method, path in paths.items():
            payload = json.loads(path.read_text())
            passed = [comparison_key(row) for row in payload["records"]] == reference
            audit["passed"] = audit["passed"] and passed
            audit["backends"][backend][method] = {"passed": passed, "sha256": sha256(path)}
    atomic_json(args.output_dir / "matched_pair_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("identity/restorer streams are not paired")


if __name__ == "__main__":
    main()
