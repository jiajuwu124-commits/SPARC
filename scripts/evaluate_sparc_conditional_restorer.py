#!/usr/bin/env python3
"""Evaluate the development-only conditional restorer with oracle scenario selection."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streammint.conditional_restorer import ConditionalResidualRestorer  # noqa: E402
from streammint.data import CLIP_MEAN, CLIP_STD, stratified_subset  # noqa: E402
from streammint.methods import StreamingAdapter, build_text_weights  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CachedDataset(Dataset):
    def __init__(self, cache_dir, scenario, ids, classes):
        protocol = json.loads((cache_dir / "protocol.json").read_text())
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if protocol["ids_sha256"] != digest or protocol["n"] != len(ids):
            raise RuntimeError("cache and selected IDs do not match")
        key = scenario.replace(":", "_s")
        self.images = np.load(cache_dir / protocol["files"][key]["path"], mmap_mode="r")
        self.ids = ids
        self.classes = classes

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        array = np.asarray(self.images[index]).copy()
        image_hash = hashlib.sha256(array.tobytes()).hexdigest()
        image = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        target = self.classes[self.ids[index].split("/", 1)[0]]
        return image, target, index, image_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status", choices=("development", "diagnostic"), required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--backends", default="clip,mint")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=1109)
    parser.add_argument("--seed", type=int, default=1109)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--restorer-scales", default="1.0",
        help="comma-separated residual scales; development screening only unless frozen",
    )
    parser.add_argument("--meta", type=Path, default=ROOT / "data/imagenet/meta.bin")
    parser.add_argument("--clip-cache", type=Path, default=Path.home() / ".cache/clip")
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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

    all_ids = [line for line in args.ids.read_text().splitlines() if line]
    ids = stratified_subset(all_ids, args.n, args.subset_seed) if args.n else all_ids
    scenarios = [value for value in args.scenarios.split(",") if value]
    backends = [value for value in args.backends.split(",") if value]
    restorer_scales = [float(value) for value in args.restorer_scales.split(",") if value]
    if any(value not in {"clip", "mint"} for value in backends):
        raise ValueError("backend must be clip or mint")
    meta, _ = torch.load(args.meta, weights_only=False)
    classes = {wnid: index for index, wnid in enumerate(sorted(meta))}

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    restorer = ConditionalResidualRestorer(**checkpoint["model"]).cuda()
    restorer.load_state_dict(checkpoint["state_dict"])
    restorer.eval()
    trained_scenarios = checkpoint["scenarios"]
    if any(scenario.split(":")[0] not in trained_scenarios for scenario in scenarios):
        raise RuntimeError("requested scenario was not used to train this specialist")

    clip_path = args.clip_cache / "ViT-L-14.pt"
    model, _ = clip.load(str(clip_path), device="cuda")
    model.eval()
    text = build_text_weights(model, [meta[wnid][0] for wnid in sorted(meta)])
    mean = torch.tensor(CLIP_MEAN, device="cuda")[None, :, None, None]
    std = torch.tensor(CLIP_STD, device="cuda")[None, :, None, None]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": args.status,
        "routing": "oracle scenario condition for specialist diagnosis only; not a deployable routing result",
        "seed": args.seed,
        "ids_source": str(args.ids.resolve()),
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "n": len(ids),
        "scenarios": scenarios,
        "backends": backends,
        "batch_size": args.batch_size,
        "checkpoint_sha256": sha(args.checkpoint),
        "restorer_scales": restorer_scales,
        "cache_protocol_sha256": sha(args.cache / "protocol.json"),
        "runner_sha256": sha(Path(__file__)),
        "model_source_sha256": sha(ROOT / "src/streammint/conditional_restorer.py"),
    }
    protocol_path = args.output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    for backend in backends:
        for scenario in scenarios:
            condition = trained_scenarios.index(scenario.split(":")[0])
            methods = [("identity", None)] + [
                (
                    "conditional_restorer" if scale == 1.0 else
                    f"conditional_restorer_scale{str(scale).replace('.', 'p')}",
                    scale,
                )
                for scale in restorer_scales
            ]
            for method, residual_scale in methods:
                key = scenario.replace(":", "s")
                output_path = args.output_dir / f"ViT-L-14_{backend}_{method}_{key}_n{len(ids)}_seed{args.seed}.json"
                torch.manual_seed(args.seed)
                adapter = (
                    StreamingAdapter(model, text, method="mint", lr=0.015, prior=10000.0)
                    if backend == "mint" else None
                )
                loader = DataLoader(
                    CachedDataset(args.cache, scenario, ids, classes),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    generator=torch.Generator().manual_seed(args.seed), pin_memory=True,
                )
                records = []
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                for batch_index, (images, labels, samples, raw_hashes) in enumerate(loader):
                    images = images.cuda(non_blocking=True)
                    if residual_scale is not None:
                        scenario_ids = torch.full(
                            (len(images),), condition, device="cuda", dtype=torch.long
                        )
                        with torch.no_grad(), torch.autocast("cuda"):
                            restored = restorer(images, scenario_ids)
                            images = (images + residual_scale * (restored - images)).clamp(0.0, 1.0)
                    normalized = (images - mean) / std
                    if backend == "mint":
                        logits = adapter(normalized)
                    else:
                        with torch.no_grad(), torch.autocast("cuda"):
                            features = F.normalize(model.encode_image(normalized).float(), dim=-1)
                            logits = 100.0 * features @ text.T
                    predictions = logits.argmax(-1).cpu().tolist()
                    for row, prediction in enumerate(predictions):
                        target = int(labels[row])
                        records.append({
                            "sample": int(samples[row]), "target": target,
                            "prediction": prediction, "correct": prediction == target,
                            "scenario": scenario, "batch": batch_index,
                            "stream_position": len(records), "raw_sha256": raw_hashes[row],
                        })
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                payload = {
                    "protocol": {
                        "protocol_sha256": sha(protocol_path), "backend": backend,
                        "method": method, "scenario": scenario, "seed": args.seed,
                    },
                    "accuracy": 100 * sum(row["correct"] for row in records) / len(records),
                    "records": records,
                    "timing": {"seconds": elapsed, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30},
                    "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": platform.python_version()},
                }
                output_path.write_text(json.dumps(payload, indent=2) + "\n")
                print(json.dumps({
                    "backend": backend, "method": method, "scenario": scenario,
                    "accuracy": payload["accuracy"], "seconds": round(elapsed, 2),
                    "peak_gpu_gib": round(payload["timing"]["peak_gpu_gib"], 3),
                }), flush=True)
                if adapter is not None:
                    adapter.reset()
                del adapter, loader
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
