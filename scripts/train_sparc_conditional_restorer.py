#!/usr/bin/env python3
"""Train SPARC's compact conditional restorer on development-only paired images."""
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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import clip


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streammint.conditional_restorer import ConditionalResidualRestorer  # noqa: E402
from streammint.data import stratified_subset  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PairedCacheDataset(Dataset):
    def __init__(self, corrupt_cache: Path, clean_cache: Path, scenarios, indices, crop, train):
        corrupt_protocol = json.loads((corrupt_cache / "protocol.json").read_text())
        clean_protocol = json.loads((clean_cache / "protocol.json").read_text())
        if corrupt_protocol["ids_sha256"] != clean_protocol["ids_sha256"]:
            raise RuntimeError("clean and corruption caches use different image IDs")
        self.corrupt = [
            np.load(corrupt_cache / corrupt_protocol["files"][f"{name}_s5"]["path"], mmap_mode="r")
            for name in scenarios
        ]
        self.clean = np.load(
            clean_cache / clean_protocol["files"]["clean_s5"]["path"], mmap_mode="r"
        )
        self.indices = list(indices)
        self.crop = int(crop)
        self.train = bool(train)

    def __len__(self):
        return len(self.indices) * len(self.corrupt)

    def __getitem__(self, item):
        scenario = item // len(self.indices)
        index = self.indices[item % len(self.indices)]
        source = torch.from_numpy(np.asarray(self.corrupt[scenario][index]).copy()).permute(2, 0, 1).float() / 255
        target = torch.from_numpy(np.asarray(self.clean[index]).copy()).permute(2, 0, 1).float() / 255
        if self.crop < source.shape[-1]:
            if self.train:
                top = torch.randint(0, source.shape[-2] - self.crop + 1, ()).item()
                left = torch.randint(0, source.shape[-1] - self.crop + 1, ()).item()
                if torch.rand(()) < 0.5:
                    source, target = source.flip(-1), target.flip(-1)
            else:
                top = (source.shape[-2] - self.crop) // 2
                left = (source.shape[-1] - self.crop) // 2
            source = source[:, top:top + self.crop, left:left + self.crop]
            target = target[:, top:top + self.crop, left:left + self.crop]
        return source, target, scenario, index


def restoration_loss(output, target):
    charbonnier = torch.sqrt((output - target).square() + 1e-6).mean()
    grad_x = (output[..., 1:] - output[..., :-1]) - (target[..., 1:] - target[..., :-1])
    grad_y = (output[..., 1:, :] - output[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    gradient = grad_x.abs().mean() + grad_y.abs().mean()
    coarse = F.l1_loss(F.avg_pool2d(output, 4), F.avg_pool2d(target, 4))
    return charbonnier + 0.15 * gradient + 0.20 * coarse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupt-cache", type=Path, required=True)
    parser.add_argument("--clean-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", default="defocus_blur,snow,frost,fog")
    parser.add_argument("--seed", type=int, default=1109)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--semantic-weight", type=float, default=0.0)
    parser.add_argument("--semantic-cache", type=Path)
    parser.add_argument("--semantic-backbone", default="ViT-B/16")
    parser.add_argument("--clip-cache", type=Path, default=Path.home() / ".cache/clip")
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    torch.cuda.memory.set_per_process_memory_fraction(0.68)

    scenarios = [value for value in args.scenarios.split(",") if value]
    corrupt_protocol = json.loads((args.corrupt_cache / "protocol.json").read_text())
    all_ids = Path(corrupt_protocol["ids_source"]).read_text().splitlines()
    ids = stratified_subset(
        all_ids, corrupt_protocol["n"], corrupt_protocol["subset_seed"]
    )
    if corrupt_protocol["n"] != 1000:
        raise RuntimeError("expected the frozen 1K development cache")
    ranked = sorted(
        range(len(ids)),
        key=lambda index: hashlib.sha256(f"sparc-restorer-split-{args.seed}:{ids[index]}".encode()).hexdigest(),
    )
    train_indices, validation_indices = ranked[:800], ranked[800:]
    train = PairedCacheDataset(
        args.corrupt_cache, args.clean_cache, scenarios, train_indices, args.crop, True
    )
    validation = PairedCacheDataset(
        args.corrupt_cache, args.clean_cache, scenarios, validation_indices, args.crop, False
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        generator=generator, pin_memory=True, persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )

    model = ConditionalResidualRestorer(len(scenarios), width=args.width).cuda()
    semantic_model = None
    clean_features = None
    if args.semantic_weight > 0:
        if args.crop != 224:
            raise ValueError("semantic training requires --crop 224 for aligned cached features")
        if args.semantic_cache is None:
            raise ValueError("--semantic-cache is required when --semantic-weight > 0")
        semantic_protocol = json.loads((args.semantic_cache / "protocol.json").read_text())
        if semantic_protocol["backbone"] != args.semantic_backbone:
            raise RuntimeError("semantic cache backbone mismatch")
        if semantic_protocol["clean_cache_protocol_sha256"] != sha(args.clean_cache / "protocol.json"):
            raise RuntimeError("semantic features and clean cache do not match")
        clean_features = torch.from_numpy(
            np.load(args.semantic_cache / "clean_features.npy")
        ).cuda()
        clip_path = args.clip_cache / f"{args.semantic_backbone.replace('/', '-')}.pt"
        semantic_model, _ = clip.load(str(clip_path), device="cuda")
        semantic_model.eval().requires_grad_(False)
        clip_mean = torch.tensor(
            (0.48145466, 0.4578275, 0.40821073), device="cuda"
        )[None, :, None, None]
        clip_std = torch.tensor(
            (0.26862954, 0.26130258, 0.27577711), device="cuda"
        )[None, :, None, None]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history, best = [], float("inf")
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for source, target, scenario, index in train_loader:
            source = source.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            scenario = scenario.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                output = model(source, scenario)
                pixel_loss = restoration_loss(output, target)
                if semantic_model is None:
                    semantic_loss = output.new_zeros(())
                else:
                    output_features = F.normalize(
                        semantic_model.encode_image((output - clip_mean) / clip_std).float(), dim=-1
                    )
                    semantic_loss = (1.0 - (output_features * clean_features[index]).sum(-1)).mean()
                loss = pixel_loss + args.semantic_weight * semantic_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach()) * len(source)
        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for source, target, scenario, index in validation_loader:
                source = source.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                scenario = scenario.cuda(non_blocking=True)
                index = index.cuda(non_blocking=True)
                with torch.autocast("cuda"):
                    output = model(source, scenario)
                    pixel_loss = restoration_loss(output, target)
                    if semantic_model is None:
                        semantic_loss = output.new_zeros(())
                    else:
                        output_features = F.normalize(
                            semantic_model.encode_image((output - clip_mean) / clip_std).float(), dim=-1
                        )
                        semantic_loss = (1.0 - (output_features * clean_features[index]).sum(-1)).mean()
                    loss = pixel_loss + args.semantic_weight * semantic_loss
                validation_total += float(loss) * len(source)
        row = {
            "epoch": epoch,
            "train_loss": train_total / len(train),
            "validation_loss": validation_total / len(validation),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["validation_loss"] < best:
            best = row["validation_loss"]
            torch.save({
                "state_dict": model.state_dict(),
                "model": {"scenario_count": len(scenarios), "width": args.width},
                "scenarios": scenarios,
                "seed": args.seed,
                "epoch": epoch,
                "validation_loss": best,
            }, args.output_dir / "best.pt")

    protocol = {
        "status": "development_only",
        "seed": args.seed,
        "scenarios": scenarios,
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "train_indices_sha256": hashlib.sha256(np.asarray(train_indices, dtype=np.int64).tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(np.asarray(validation_indices, dtype=np.int64).tobytes()).hexdigest(),
        "corrupt_cache_protocol_sha256": sha(args.corrupt_cache / "protocol.json"),
        "clean_cache_protocol_sha256": sha(args.clean_cache / "protocol.json"),
        "checkpoint_sha256": sha(args.output_dir / "best.pt"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "crop": args.crop,
        "width": args.width,
        "lr": args.lr,
        "semantic_weight": args.semantic_weight,
        "semantic_backbone": args.semantic_backbone if args.semantic_weight > 0 else None,
        "semantic_cache_protocol_sha256": (
            sha(args.semantic_cache / "protocol.json") if args.semantic_weight > 0 else None
        ),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "history": history,
        "source_sha256": {
            "scripts/train_sparc_conditional_restorer.py": sha(Path(__file__)),
            "src/streammint/conditional_restorer.py": sha(ROOT / "src/streammint/conditional_restorer.py"),
        },
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({
        "best_validation_loss": best,
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "peak_gpu_gib": protocol["peak_gpu_gib"],
        "elapsed_seconds": protocol["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
