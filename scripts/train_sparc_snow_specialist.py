#!/usr/bin/env python3
"""Train SPARC's development-only occlusion-aware snow specialist.

This script never reads ImageNet class labels.  It verifies and consumes the
frozen 1K corrupted/clean image caches and the corresponding clean CLIP feature
cache, using a deterministic 800/200 image-ID split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streammint.data import CLIP_MEAN, CLIP_STD, stratified_subset  # noqa: E402
from streammint.snow_specialist import SnowSpecialist, snow_specialist_loss  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def development_split(ids: list[str], seed: int) -> tuple[list[int], list[int]]:
    """Return the project's frozen hash-ranked 800/200 development split."""
    if len(ids) != 1000:
        raise ValueError("snow specialist expects exactly 1,000 development IDs")
    ranked = sorted(
        range(len(ids)),
        key=lambda index: hashlib.sha256(
            f"sparc-restorer-split-{seed}:{ids[index]}".encode()
        ).hexdigest(),
    )
    return ranked[:800], ranked[800:]


def verify_npy(cache_dir: Path, entry: dict) -> Path:
    path = cache_dir / entry["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != entry["sha256"]:
        raise RuntimeError(f"cache file hash mismatch: {path}")
    array = np.load(path, mmap_mode="r")
    if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
        raise RuntimeError(f"cache file shape/dtype mismatch: {path}")
    return path


class SnowPairDataset(Dataset):
    def __init__(self, snow: Path, clean: Path, indices: list[int]):
        self.snow = np.load(snow, mmap_mode="r")
        self.clean = np.load(clean, mmap_mode="r")
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        source = torch.from_numpy(np.asarray(self.snow[index]).copy())
        target = torch.from_numpy(np.asarray(self.clean[index]).copy())
        return (
            source.permute(2, 0, 1).float() / 255.0,
            target.permute(2, 0, 1).float() / 255.0,
            index,
        )


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupt-cache", type=Path, required=True)
    parser.add_argument("--clean-cache", type=Path, required=True)
    parser.add_argument("--semantic-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1109)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--semantic-backbone", default="ViT-B/16")
    parser.add_argument("--clip-cache", type=Path, default=Path.home() / ".cache/clip")
    args = parser.parse_args()

    if args.seed != 1109:
        raise ValueError("the frozen development protocol requires seed 1109")
    if args.semantic_weight < 0:
        raise ValueError("semantic weight must be non-negative")
    set_determinism(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for training")
    torch.cuda.memory.set_per_process_memory_fraction(0.68)

    corrupt_protocol_path = args.corrupt_cache / "protocol.json"
    clean_protocol_path = args.clean_cache / "protocol.json"
    semantic_protocol_path = args.semantic_cache / "protocol.json"
    corrupt_protocol = json.loads(corrupt_protocol_path.read_text())
    clean_protocol = json.loads(clean_protocol_path.read_text())
    semantic_protocol = json.loads(semantic_protocol_path.read_text())

    if corrupt_protocol["n"] != 1000 or clean_protocol["n"] != 1000:
        raise RuntimeError("expected frozen 1K development image caches")
    if corrupt_protocol["ids_sha256"] != clean_protocol["ids_sha256"]:
        raise RuntimeError("corrupt and clean caches use different image IDs")
    if corrupt_protocol["subset_seed"] != 1109 or corrupt_protocol["corruption_seed"] != 1109:
        raise RuntimeError("corruption cache does not follow the seed-1109 protocol")
    if clean_protocol["subset_seed"] != 1109:
        raise RuntimeError("clean cache does not follow the seed-1109 subset protocol")
    if semantic_protocol["backbone"] != args.semantic_backbone:
        raise RuntimeError("semantic backbone mismatch")
    if semantic_protocol["clean_cache_protocol_sha256"] != sha256(clean_protocol_path):
        raise RuntimeError("semantic feature cache and clean cache do not match")

    snow_path = verify_npy(args.corrupt_cache, corrupt_protocol["files"]["snow_s5"])
    clean_path = verify_npy(args.clean_cache, clean_protocol["files"]["clean_s5"])
    feature_path = args.semantic_cache / "clean_features.npy"
    if sha256(feature_path) != semantic_protocol["feature_sha256"]:
        raise RuntimeError("semantic feature file hash mismatch")
    clean_features = np.load(feature_path, mmap_mode="r")
    if list(clean_features.shape) != semantic_protocol["shape"]:
        raise RuntimeError("semantic feature shape mismatch")

    all_ids = [line for line in Path(corrupt_protocol["ids_source"]).read_text().splitlines() if line]
    ids = stratified_subset(all_ids, corrupt_protocol["n"], corrupt_protocol["subset_seed"])
    if sequence_sha256(ids) != corrupt_protocol["ids_sha256"]:
        raise RuntimeError("selected image IDs do not match the corruption cache")
    train_indices, validation_indices = development_split(ids, args.seed)
    train_set = SnowPairDataset(snow_path, clean_path, train_indices)
    validation_set = SnowPairDataset(snow_path, clean_path, validation_indices)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda")
    model = SnowSpecialist(width=args.width).to(device)
    clip_checkpoint = args.clip_cache / f"{args.semantic_backbone.replace('/', '-')}.pt"
    if sha256(clip_checkpoint) != semantic_protocol["clip_checkpoint_sha256"]:
        raise RuntimeError("CLIP checkpoint does not match the semantic cache")
    semantic_model, _ = clip.load(str(clip_checkpoint), device=device)
    semantic_model.eval().requires_grad_(False)
    clean_feature_tensor = torch.from_numpy(np.asarray(clean_features).copy()).to(device)
    clip_mean = torch.tensor(CLIP_MEAN, device=device)[None, :, None, None]
    clip_std = torch.tensor(CLIP_STD, device=device)[None, :, None, None]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    def step(batch, training: bool):
        source, target, index = batch
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        index = index.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            output = model(source, return_aux=True)
            restored_features = F.normalize(
                semantic_model.encode_image((output.restored - clip_mean) / clip_std).float(),
                dim=-1,
            )
            semantic = (
                1.0 - (restored_features * clean_feature_tensor[index]).sum(dim=-1)
            ).mean()
            loss, terms = snow_specialist_loss(
                output,
                source,
                target,
                semantic_loss=semantic,
                semantic_weight=args.semantic_weight,
            )
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        return {name: float(value.detach()) for name, value in terms.items()}, len(source)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    history: list[dict] = []
    best = float("inf")
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sums: dict[str, float] = {}
        train_n = 0
        for batch in train_loader:
            terms, count = step(batch, True)
            train_n += count
            for name, value in terms.items():
                train_sums[name] = train_sums.get(name, 0.0) + count * value

        model.eval()
        validation_sums: dict[str, float] = {}
        validation_n = 0
        with torch.no_grad():
            for batch in validation_loader:
                terms, count = step(batch, False)
                validation_n += count
                for name, value in terms.items():
                    validation_sums[name] = validation_sums.get(name, 0.0) + count * value
        row = {
            "epoch": epoch,
            "train": {name: value / train_n for name, value in train_sums.items()},
            "validation": {
                name: value / validation_n for name, value in validation_sums.items()
            },
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        selection_value = row["validation"]["total"]
        if selection_value < best:
            best = selection_value
            torch.save(
                {
                    "format": "sparc-snow-specialist-v1",
                    "state_dict": model.state_dict(),
                    "model": {"width": args.width},
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_total": best,
                    "uses_imagenet_class_labels": False,
                },
                args.output_dir / "best.pt",
            )

    protocol = {
        "schema_version": 1,
        "status": "development_only",
        "method": "occlusion-aware residual-gated snow specialist",
        "seed": args.seed,
        "uses_imagenet_class_labels": False,
        "uses_corruption_identity": "snow-only deployment specialist",
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "train_indices_sha256": hashlib.sha256(
            np.asarray(train_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(
            np.asarray(validation_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "train_ids_sha256": sequence_sha256([ids[index] for index in train_indices]),
        "validation_ids_sha256": sequence_sha256(
            [ids[index] for index in validation_indices]
        ),
        "corrupt_cache_protocol_sha256": sha256(corrupt_protocol_path),
        "clean_cache_protocol_sha256": sha256(clean_protocol_path),
        "semantic_cache_protocol_sha256": sha256(semantic_protocol_path),
        "snow_cache_sha256": sha256(snow_path),
        "clean_cache_sha256": sha256(clean_path),
        "clean_feature_sha256": sha256(feature_path),
        "clip_checkpoint_sha256": sha256(clip_checkpoint),
        "checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "width": args.width,
        "learning_rate": args.lr,
        "semantic_weight": args.semantic_weight,
        "semantic_backbone": args.semantic_backbone,
        "memory_fraction_limit": 0.68,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "source_sha256": {
            "scripts/train_sparc_snow_specialist.py": sha256(Path(__file__)),
            "src/streammint/snow_specialist.py": sha256(
                ROOT / "src/streammint/snow_specialist.py"
            ),
        },
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            {
                "best_validation_total": best,
                "checkpoint_sha256": protocol["checkpoint_sha256"],
                "peak_gpu_gib": protocol["peak_gpu_gib"],
                "elapsed_seconds": protocol["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
