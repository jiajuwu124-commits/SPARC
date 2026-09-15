#!/usr/bin/env python3
"""Train a deployment-supervised brightness restorer on redesign data only.

Development and diagnostic A are pooled for training. Diagnostic B is used
only for validation/model selection. These three splits are redesign evidence;
no untouched/final input is accepted. ImageNet labels supervise frozen ViT-L
cross-entropy during training, but the saved restorer's forward API is image
only.
"""
from __future__ import annotations

import argparse
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streammint.data import CLIP_MEAN, CLIP_STD, stratified_subset  # noqa: E402
from streammint.deployment_brightness import (  # noqa: E402
    DeploymentBrightnessRestorer,
    deployment_brightness_loss,
)
from streammint.methods import build_text_weights  # noqa: E402


SEED = 1109
SPLITS = {
    "development": {
        "role": "redesign_training",
        "corrupt": ROOT / "cache/sparc_dev1k_corruptions_s5",
        "clean": ROOT / "cache/sparc_dev1k_clean",
    },
    "diagnostic_a": {
        "role": "redesign_training",
        "corrupt": ROOT / "cache/sparc_diagnostic1k_a_corruptions_s5",
        "clean": ROOT / "cache/sparc_diagnostic1k_a_clean",
    },
    "diagnostic_b": {
        "role": "redesign_validation_model_selection",
        "corrupt": ROOT / "cache/sparc_diagnostic1k_b_corruptions_s5",
        "clean": ROOT / "cache/sparc_diagnostic1k_b_clean",
    },
}
DEFAULT_INIT = (
    ROOT / "results/sparc_utility_router_development_20260915/"
    "brightness_clipping_specialist_w32_e30/best.pt"
)
DEFAULT_CLIP = ROOT.parent / ".cache/clip/ViT-L-14.pt"
DEFAULT_META = ROOT.parent / "datasets/imagenet/meta.bin"


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


def atomic_torch_save(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def verify_array(cache: Path, protocol: dict[str, Any], key: str) -> Path:
    entry = protocol.get("files", {}).get(key, {})
    path = cache / str(entry.get("path", ""))
    if not path.is_file() or sha256(path) != entry.get("sha256"):
        raise RuntimeError(f"missing or hash-mismatched cache array: {path}")
    array = np.load(path, mmap_mode="r")
    if array.shape != (1000, 224, 224, 3) or array.dtype != np.uint8:
        raise RuntimeError(f"invalid paired-cache schema: {path}")
    return path


def validate_inputs(init_checkpoint: Path, clip_checkpoint: Path, meta_path: Path) -> dict[str, Any]:
    split_records: dict[str, Any] = {}
    id_sets: dict[str, set[str]] = {}
    for name, specification in SPLITS.items():
        corrupt_cache = Path(specification["corrupt"])
        clean_cache = Path(specification["clean"])
        corrupt_protocol_path = corrupt_cache / "protocol.json"
        clean_protocol_path = clean_cache / "protocol.json"
        corrupt_protocol = json.loads(corrupt_protocol_path.read_text())
        clean_protocol = json.loads(clean_protocol_path.read_text())
        for protocol in (corrupt_protocol, clean_protocol):
            if protocol.get("schema_version") != 1 or protocol.get("n") != 1000:
                raise RuntimeError(f"{name} cache must contain exactly 1,000 rows")
            if protocol.get("subset_seed") != SEED or protocol.get("corruption_seed") != SEED:
                raise RuntimeError(f"{name} cache seed mismatch")
        if corrupt_protocol["ids_sha256"] != clean_protocol["ids_sha256"]:
            raise RuntimeError(f"{name} corrupt/clean ID mismatch")
        ids_path = Path(corrupt_protocol["ids_source"]).resolve()
        if ids_path != Path(clean_protocol["ids_source"]).resolve():
            raise RuntimeError(f"{name} corrupt/clean ID-source mismatch")
        all_ids = [line for line in ids_path.read_text().splitlines() if line]
        ids = stratified_subset(all_ids, 1000, SEED) if len(all_ids) != 1000 else all_ids
        if len(ids) != 1000 or sequence_sha256(ids) != corrupt_protocol["ids_sha256"]:
            raise RuntimeError(f"{name} ID sequence mismatch")
        corrupt = verify_array(corrupt_cache, corrupt_protocol, "brightness_s5")
        clean = verify_array(clean_cache, clean_protocol, "clean_s5")
        id_sets[name] = set(ids)
        split_records[name] = {
            "role": specification["role"], "n": len(ids),
            "ids_source": str(ids_path), "ids_source_sha256": sha256(ids_path),
            "ids_sequence_sha256": sequence_sha256(ids),
            "corrupt_protocol": str(corrupt_protocol_path.resolve()),
            "corrupt_protocol_sha256": sha256(corrupt_protocol_path),
            "corrupt_array": str(corrupt.resolve()), "corrupt_array_sha256": sha256(corrupt),
            "clean_protocol": str(clean_protocol_path.resolve()),
            "clean_protocol_sha256": sha256(clean_protocol_path),
            "clean_array": str(clean.resolve()), "clean_array_sha256": sha256(clean),
        }
    overlaps = {
        "development__diagnostic_a": len(id_sets["development"] & id_sets["diagnostic_a"]),
        "development__diagnostic_b": len(id_sets["development"] & id_sets["diagnostic_b"]),
        "diagnostic_a__diagnostic_b": len(id_sets["diagnostic_a"] & id_sets["diagnostic_b"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"redesign split overlap: {overlaps}")
    if not init_checkpoint.is_file() or not clip_checkpoint.is_file() or not meta_path.is_file():
        raise FileNotFoundError("initialization checkpoint, CLIP checkpoint, or metadata is missing")
    initialization = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    if initialization.get("format") != "sparc-brightness-clipping-specialist-v1":
        raise RuntimeError("unsupported initialization checkpoint")
    model = DeploymentBrightnessRestorer(**initialization["model"])
    model.load_state_dict(initialization["state_dict"])
    if list(inspect.signature(model.forward).parameters) != ["images", "return_aux"]:
        raise RuntimeError("restorer inference API is not image-only")
    with torch.no_grad():
        source = torch.rand(2, 3, 64, 64)
        target = torch.rand_like(source)
        output = model(source, return_aux=True)
        semantic = torch.tensor(1.0)
        loss, terms = deployment_brightness_loss(output, target, semantic)
    if not torch.isfinite(loss) or not {"reconstruction", "semantic_cross_entropy",
                                        "learned_residual_regularizer"}.issubset(terms):
        raise RuntimeError("deployment loss CPU smoke test failed")
    return {
        "schema_version": 1, "status": "cpu_validate_only_pass",
        "seed": SEED, "splits": split_records, "split_overlaps": overlaps,
        "training_rows": 2000, "validation_rows": 1000,
        "init_checkpoint": str(init_checkpoint.resolve()),
        "init_checkpoint_sha256": sha256(init_checkpoint),
        "clip_checkpoint": str(clip_checkpoint.resolve()),
        "clip_checkpoint_sha256": sha256(clip_checkpoint),
        "meta": str(meta_path.resolve()), "meta_sha256": sha256(meta_path),
        "uses_imagenet_class_labels_during_training": True,
        "validation_b_used_for_model_selection": True,
        "inference_requires_labels_or_corruption_id": False,
        "untouched_or_final_inputs": [],
        "cpu_model_and_loss_smoke_pass": True,
    }


class PairedDataset(Dataset):
    def __init__(self, corrupt: Path, clean: Path, labels: list[int], split: str):
        self.corrupt = np.load(corrupt, mmap_mode="r")
        self.clean = np.load(clean, mmap_mode="r")
        self.labels = labels
        self.split = split

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        corrupt = torch.from_numpy(np.asarray(self.corrupt[index]).copy()).permute(2, 0, 1).float() / 255
        clean = torch.from_numpy(np.asarray(self.clean[index]).copy()).permute(2, 0, 1).float() / 255
        return corrupt, clean, self.labels[index], index, self.split


def labels_for(record: dict[str, Any], class_to_index: dict[str, int]) -> list[int]:
    all_ids = [line for line in Path(record["ids_source"]).read_text().splitlines() if line]
    ids = stratified_subset(all_ids, 1000, SEED) if len(all_ids) != 1000 else all_ids
    return [class_to_index[value.split("/", 1)[0]] for value in ids]


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--clip-checkpoint", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--semantic-weight", type=float, default=0.05)
    parser.add_argument("--residual-weight", type=float, default=0.02)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("invalid training schedule")
    if args.semantic_weight < 0 or args.residual_weight < 0:
        raise ValueError("loss weights must be non-negative")
    inputs = validate_inputs(args.init_checkpoint, args.clip_checkpoint, args.meta)
    inputs["requested_hyperparameters"] = {
        "epochs": args.epochs, "batch_size": args.batch_size, "workers": args.workers,
        "learning_rate": args.lr, "semantic_weight": args.semantic_weight,
        "residual_weight": args.residual_weight, "memory_fraction_limit": 0.68,
    }
    inputs["source_sha256"] = {
        "scripts/train_sparc_deployment_brightness.py": sha256(Path(__file__)),
        "src/streammint/deployment_brightness.py": sha256(ROOT / "src/streammint/deployment_brightness.py"),
        "src/streammint/brightness_specialist.py": sha256(ROOT / "src/streammint/brightness_specialist.py"),
    }
    if args.validation_report is not None:
        args.validation_report.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.validation_report, inputs)
    if args.validate_only:
        print(json.dumps(inputs, indent=2))
        return
    set_determinism()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required after CPU validate-only")
    torch.cuda.memory.set_per_process_memory_fraction(0.68)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    meta, _ = torch.load(args.meta, map_location="cpu", weights_only=False)
    wnids = sorted(meta)
    class_to_index = {wnid: index for index, wnid in enumerate(wnids)}
    records = inputs["splits"]
    datasets = {
        name: PairedDataset(
            Path(record["corrupt_array"]), Path(record["clean_array"]),
            labels_for(record, class_to_index), name,
        )
        for name, record in records.items()
    }
    training = ConcatDataset([datasets["development"], datasets["diagnostic_a"]])
    validation = datasets["diagnostic_b"]
    loaders = {
        "train": DataLoader(
            training, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
            generator=torch.Generator().manual_seed(SEED), pin_memory=True,
            persistent_workers=args.workers > 0,
        ),
        "validation_b": DataLoader(
            validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0,
        ),
    }
    initialization = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    restorer = DeploymentBrightnessRestorer(**initialization["model"]).to(device)
    restorer.load_state_dict(initialization["state_dict"])
    semantic_model, _ = clip.load(str(args.clip_checkpoint), device=device)
    semantic_model.eval().requires_grad_(False)
    text_weights = build_text_weights(semantic_model, [meta[wnid][0] for wnid in wnids])
    mean = torch.tensor(CLIP_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(CLIP_STD, device=device)[None, :, None, None]
    optimizer = torch.optim.AdamW(restorer.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    def step(batch, training_phase: bool):
        corrupt, clean, labels, _indices, _splits = batch
        corrupt = corrupt.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training_phase:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            output = restorer(corrupt, return_aux=True)
            semantic_features = F.normalize(
                semantic_model.encode_image((output.restored - mean) / std).float(), dim=-1
            )
            semantic_ce = F.cross_entropy(100.0 * semantic_features @ text_weights.T, labels)
            loss, terms = deployment_brightness_loss(
                output, clean, semantic_ce, semantic_weight=args.semantic_weight,
                residual_weight=args.residual_weight,
            )
        if training_phase:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        return {name: float(value.detach()) for name, value in terms.items()}, len(corrupt)

    history, best = [], float("inf")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        row: dict[str, Any] = {"epoch": epoch}
        for phase in ("train", "validation_b"):
            restorer.train(phase == "train")
            sums: dict[str, float] = {}
            count = 0
            context = torch.enable_grad() if phase == "train" else torch.no_grad()
            with context:
                for batch in loaders[phase]:
                    terms, size = step(batch, phase == "train")
                    count += size
                    for name, value in terms.items():
                        sums[name] = sums.get(name, 0.0) + size * value
            row[phase] = {name: value / count for name, value in sums.items()}
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["validation_b"]["total"] < best:
            best = row["validation_b"]["total"]
            atomic_torch_save(args.output_dir / "best.pt", {
                "format": "sparc-deployment-supervised-brightness-v1",
                "state_dict": restorer.state_dict(), "model": initialization["model"],
                "seed": SEED, "epoch": epoch, "validation_b_total": best,
                "uses_imagenet_class_labels_during_training": True,
                "training_split_roles": {name: record["role"] for name, record in records.items()},
                "inference_requires_class_labels": False,
                "inference_requires_corruption_identity": False,
            })
    protocol = {
        **inputs, "status": "redesign_complete_not_final_tested",
        "method": "deployment-supervised clipping-aware brightness restorer",
        "objective": ["paired_reconstruction", "frozen_vitl_class_cross_entropy",
                      "learned_residual_regularization"],
        "epochs": args.epochs, "batch_size": args.batch_size, "workers": args.workers,
        "learning_rate": args.lr, "semantic_weight": args.semantic_weight,
        "residual_weight": args.residual_weight, "memory_fraction_limit": 0.68,
        "checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "best_validation_b_total": best, "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "python": platform.python_version()},
        "source_sha256": {
            "scripts/train_sparc_deployment_brightness.py": sha256(Path(__file__)),
            "src/streammint/deployment_brightness.py": sha256(ROOT / "src/streammint/deployment_brightness.py"),
            "src/streammint/brightness_specialist.py": sha256(ROOT / "src/streammint/brightness_specialist.py"),
        },
    }
    atomic_json(args.output_dir / "protocol.json", protocol)
    print(json.dumps({"checkpoint": str(args.output_dir / "best.pt"),
                      "checkpoint_sha256": protocol["checkpoint_sha256"],
                      "best_validation_b_total": best}, indent=2))


if __name__ == "__main__":
    main()
