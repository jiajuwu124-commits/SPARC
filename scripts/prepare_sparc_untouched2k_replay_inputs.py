#!/usr/bin/env python3
"""Prepare label-free replay inputs for the already frozen SPARC untouched 2K.

This program reads only the frozen ID ledger and corruption arrays. It neither
loads a recognition model nor reads any prediction/result artifact. Per-
corruption descriptors and stream indexes are written atomically and can be
reused only after their source/hash contracts pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streammint.risk_route_fast import masked_risks_fast  # noqa: E402
from streammint.utility_router import RISK_FEATURE_ORDER, risk_features  # noqa: E402


IDS = ROOT / "configs/sparc_untouched_confirmation2k_ids.txt"
FREEZE = ROOT / "results/sparc_untouched_confirmation_20260915/manifest.json"
CACHE = ROOT / "cache/sparc_untouched_confirmation2k_corruptions_s5"
OUTPUT = ROOT / "cache/sparc_untouched_confirmation2k_descriptors_s5"
INDEX_OUTPUT = ROOT / "results/sparc_untouched_replay_inputs_20260915"
RISK_SOURCE = ROOT / "src/streammint/risk_route_fast.py"
FEATURE_SOURCE = ROOT / "src/streammint/utility_router.py"
SEED, N = 1109, 2000
SCENARIOS = (
    "gaussian_noise:5", "shot_noise:5", "impulse_noise:5",
    "defocus_blur:5", "glass_blur:5", "motion_blur:5", "zoom_blur:5",
    "snow:5", "frost:5", "fog:5", "brightness:5",
    "contrast:5", "elastic_transform:5", "pixelate:5", "jpeg_compression:5",
)
RISK_BANK = ("identity", "gaussian_1.0", "gaussian_1.5", "median_3")
_IMAGES: np.ndarray | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **values)
    temporary.replace(path)


def key(scenario: str) -> str:
    name, severity = scenario.rsplit(":", 1)
    return f"{name}_s{severity}"


def init_worker(path: str) -> None:
    global _IMAGES
    _IMAGES = np.load(path, mmap_mode="r")


def extract(index: int) -> tuple[int, np.ndarray, str]:
    if _IMAGES is None:
        raise RuntimeError("worker image array was not initialized")
    image = np.asarray(_IMAGES[index]).copy()
    return index, masked_risks_fast(Image.fromarray(image, mode="RGB"), RISK_BANK), \
        hashlib.sha256(image.tobytes()).hexdigest()


def descriptor_valid(
    path: Path, scenario: str, ids: list[str], source_protocol_sha: str, source_array_sha: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(("keys", "image_ids", "samples", "raw_sha256", "risks", "features",
                    "feature_order", "scenario", "source_protocol_sha256",
                    "source_array_sha256", "ids_sha256")) - set(data.files):
                return False
            risks, features = data["risks"], data["features"]
            return (
                risks.shape == (N, 4, 2) and risks.dtype == np.float32
                and features.shape == (N, 18) and features.dtype == np.float64
                and str(data["scenario"].item()) == scenario
                and str(data["source_protocol_sha256"].item()) == source_protocol_sha
                and str(data["source_array_sha256"].item()) == source_array_sha
                and str(data["ids_sha256"].item()) == sequence_sha256(ids)
                and [str(value) for value in data["image_ids"].tolist()] == ids
                and np.array_equal(data["samples"], np.arange(N, dtype=np.int64))
                and tuple(str(value) for value in data["feature_order"].tolist()) == tuple(RISK_FEATURE_ORDER)
                and np.array_equal(features, risk_features(risks))
            )
    except (OSError, ValueError, KeyError):
        return False


def build_descriptor(
    array_path: Path, output: Path, scenario: str, ids: list[str],
    source_protocol_sha: str, source_array_sha: str, workers: int,
) -> None:
    risks = np.empty((N, 4, 2), dtype=np.float32)
    raw_hashes = np.empty(N, dtype="S64")
    if workers == 1:
        init_worker(str(array_path))
        iterator = map(extract, range(N))
        for index, value, raw_hash in iterator:
            risks[index], raw_hashes[index] = value, raw_hash
    else:
        context = mp.get_context("fork")
        with context.Pool(workers, initializer=init_worker, initargs=(str(array_path),),
                          maxtasksperchild=250) as pool:
            for index, value, raw_hash in pool.imap(extract, range(N), chunksize=4):
                risks[index], raw_hashes[index] = value, raw_hash
    atomic_npz(
        output, keys=np.asarray([f"{scenario}#{i}" for i in range(N)]),
        image_ids=np.asarray(ids), samples=np.arange(N, dtype=np.int64),
        raw_sha256=raw_hashes, risks=risks, features=risk_features(risks),
        feature_order=np.asarray(RISK_FEATURE_ORDER), scenario=np.asarray(scenario),
        source_protocol_sha256=np.asarray(source_protocol_sha),
        source_array_sha256=np.asarray(source_array_sha),
        ids_sha256=np.asarray(sequence_sha256(ids)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1,4]")
    ids = [line.strip() for line in IDS.read_text().splitlines() if line.strip()]
    freeze, protocol = json.loads(FREEZE.read_text()), json.loads((CACHE / "protocol.json").read_text())
    ids_sha = sequence_sha256(ids)
    if len(ids) != N or len(set(ids)) != N or freeze.get("ids_sha256") != ids_sha:
        raise RuntimeError("frozen ID contract mismatch")
    expected_protocol = {"n": N, "ids_sha256": ids_sha, "subset_seed": SEED,
                         "corruption_seed": SEED, "scenarios": list(SCENARIOS)}
    if any(protocol.get(name) != value for name, value in expected_protocol.items()):
        raise RuntimeError("untouched cache protocol mismatch")
    source_protocol_sha = sha256(CACHE / "protocol.json")
    classes = sorted({value.split("/", 1)[0] for value in ids})
    if len(classes) != 1000 or any(sum(value.startswith(name + "/") for value in ids) != 2
                                   for name in classes):
        raise RuntimeError("untouched IDs are not exactly two per ImageNet class")
    targets = {name: index for index, name in enumerate(classes)}
    artifacts = {}
    for scenario in SCENARIOS:
        entry = protocol["files"][key(scenario)]
        array_path = (CACHE / entry["path"]).resolve()
        if not array_path.is_file() or array_path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"missing/size-mismatched input array: {scenario}")
        source_array_sha = sha256(array_path)
        if source_array_sha != entry["sha256"]:
            raise RuntimeError(f"input array SHA mismatch: {scenario}")
        descriptor = OUTPUT / "by_scenario" / f"{key(scenario)}.npz"
        valid = descriptor_valid(descriptor, scenario, ids, source_protocol_sha, source_array_sha)
        if args.validate_only and not valid:
            raise RuntimeError(f"descriptor is missing or invalid: {scenario}")
        if not valid:
            build_descriptor(array_path, descriptor, scenario, ids, source_protocol_sha,
                             source_array_sha, args.workers)
        if not descriptor_valid(descriptor, scenario, ids, source_protocol_sha, source_array_sha):
            raise RuntimeError(f"descriptor post-write audit failed: {scenario}")
        with np.load(descriptor, allow_pickle=False) as data:
            raw_hashes = [value.decode() if isinstance(value, bytes) else str(value)
                          for value in data["raw_sha256"]]
        records = [{
            "key": f"{scenario}#{sample}", "image_id": image_id, "sample": sample,
            "target": targets[image_id.split("/", 1)[0]], "audit_domain": scenario,
            "image": {"path": str(array_path), "format": "npy", "index": sample},
            "raw_sha256": raw_hashes[sample],
        } for sample, image_id in enumerate(ids)]
        index_payload = {
            "schema_version": 1, "status": "frozen_untouched_input_only_no_outcomes",
            "split": "test", "scenario": scenario, "rows": N, "seed": SEED,
            "test_ids_sequence_sha256": ids_sha, "records": records,
            "provenance": {
                "freeze_manifest": str(FREEZE), "freeze_manifest_sha256": sha256(FREEZE),
                "source_cache_protocol_sha256": source_protocol_sha,
                "source_array_sha256": source_array_sha, "descriptor_sha256": sha256(descriptor),
                "recognition_model_loaded": False, "prediction_outcomes_read": False,
            },
        }
        index_path = INDEX_OUTPUT / "per_corruption" / f"{key(scenario)}_stream.json"
        if index_path.exists() and json.loads(index_path.read_text()) != index_payload:
            if args.validate_only:
                raise RuntimeError(f"stream index changed: {scenario}")
            atomic_json(index_path, index_payload)
        elif not index_path.exists():
            if args.validate_only:
                raise RuntimeError(f"stream index missing: {scenario}")
            atomic_json(index_path, index_payload)
        artifacts[scenario] = {
            "source_array": {"path": str(array_path), "sha256": source_array_sha},
            "descriptor": {"path": str(descriptor), "sha256": sha256(descriptor)},
            "stream_index": {"path": str(index_path), "sha256": sha256(index_path)},
        }
        print(json.dumps({"scenario": scenario, "status": "reused" if valid else "built"}), flush=True)
    report = {
        "schema_version": 1, "status": "prepared_untouched_input_only_no_outcomes",
        "seed": SEED, "n_per_scenario": N, "scenarios": list(SCENARIOS),
        "ids": {"path": str(IDS), "sha256": sha256(IDS), "sequence_sha256": ids_sha},
        "freeze_manifest": {"path": str(FREEZE), "sha256": sha256(FREEZE)},
        "source_cache_protocol_sha256": source_protocol_sha, "artifacts": artifacts,
        "descriptor_uses_labels": False, "recognition_model_loaded": False,
        "prediction_outcomes_read": False, "gpu_used": False,
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "source_artifacts": {
            "masked_risk_extractor": {"path": str(RISK_SOURCE), "sha256": sha256(RISK_SOURCE)},
            "descriptor_transform": {"path": str(FEATURE_SOURCE), "sha256": sha256(FEATURE_SOURCE)},
        },
    }
    report_path = INDEX_OUTPUT / "readiness.json"
    if report_path.exists() and json.loads(report_path.read_text()) != report:
        if args.validate_only:
            raise RuntimeError("input readiness report changed")
        atomic_json(report_path, report)
    elif not report_path.exists():
        if args.validate_only:
            raise RuntimeError("input readiness report missing")
        atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "scenarios": len(artifacts),
                      "report": str(report_path), "sha256": sha256(report_path)}, indent=2))


if __name__ == "__main__":
    main()
