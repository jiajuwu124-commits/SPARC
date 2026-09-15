#!/usr/bin/env python3
"""Build a label-free 15K masked-risk descriptor cache from frozen uint8 inputs.

Only the development corruption cache is read.  No ImageNet metadata, category
label, model prediction, result directory, diagnostic split, or confirmation
split is an input to this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from streammint.risk_route_fast import masked_risks_fast  # noqa: E402
from streammint.utility_router import RISK_FEATURE_ORDER, risk_features  # noqa: E402


SEED = 1109
RISK_BANK = ("identity", "gaussian_1.0", "gaussian_1.5", "median_3")
SCENARIOS = (
    "gaussian_noise:5",
    "shot_noise:5",
    "impulse_noise:5",
    "defocus_blur:5",
    "glass_blur:5",
    "motion_blur:5",
    "zoom_blur:5",
    "snow:5",
    "frost:5",
    "fog:5",
    "brightness:5",
    "contrast:5",
    "elastic_transform:5",
    "pixelate:5",
    "jpeg_compression:5",
)
EXPECTED_FEATURE_ORDER = tuple(
    [f"log10_risk_v{view}_{loss}" for view in range(4) for loss in ("mse", "mae")]
    + [f"normalized_risk_v{view}_{loss}" for view in range(4) for loss in ("mse", "mae")]
    + ["identity_gain_mse", "identity_gain_mae"]
)
_WORKER_IMAGES: np.ndarray | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def scenario_key(scenario: str) -> str:
    name, severity = scenario.rsplit(":", 1)
    return f"{name}_s{int(severity)}"


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def validate_input_cache(cache: Path) -> tuple[dict[str, Any], dict[str, str]]:
    protocol_path = cache / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != 1:
        raise RuntimeError("unsupported corruption-cache protocol")
    if int(protocol.get("n", -1)) != 1000:
        raise RuntimeError("descriptor source must contain exactly 1,000 images per scenario")
    if int(protocol.get("subset_seed", -1)) != SEED:
        raise RuntimeError("descriptor source subset seed must be 1109")
    if int(protocol.get("corruption_seed", -1)) != SEED:
        raise RuntimeError("descriptor source corruption seed must be 1109")
    if tuple(protocol.get("scenarios", ())) != SCENARIOS:
        raise RuntimeError("descriptor source does not use the frozen 15-scenario order")
    if not re.fullmatch(r"[0-9a-f]{64}", str(protocol.get("ids_sha256", ""))):
        raise RuntimeError("descriptor source lacks a valid ID-sequence hash")
    hashes = {}
    for scenario in SCENARIOS:
        entry = protocol.get("files", {}).get(scenario_key(scenario))
        if not isinstance(entry, dict):
            raise RuntimeError(f"source cache is missing {scenario}")
        path = cache / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != entry.get("sha256"):
            raise RuntimeError(f"source cache file hash mismatch: {path}")
        array = np.load(path, mmap_mode="r")
        if array.shape != (1000, 224, 224, 3) or array.dtype != np.uint8:
            raise RuntimeError(f"source cache array schema mismatch: {path}")
        hashes[scenario] = actual
    return protocol, hashes


def source_protocol(cache: Path, cache_protocol: dict[str, Any], hashes: dict[str, str], workers: int) -> dict[str, Any]:
    if tuple(RISK_FEATURE_ORDER) != EXPECTED_FEATURE_ORDER or len(RISK_FEATURE_ORDER) != 18:
        raise RuntimeError("runtime descriptor definition differs from the frozen 18-D order")
    sources = {
        "scripts/build_sparc_dev15_descriptor_cache.py": Path(__file__).resolve(),
        "src/streammint/risk_route_fast.py": ROOT / "src/streammint/risk_route_fast.py",
        "src/streammint/utility_router.py": ROOT / "src/streammint/utility_router.py",
        "src/streammint/data.py": ROOT / "src/streammint/data.py",
        "src/streammint/__init__.py": ROOT / "src/streammint/__init__.py",
    }
    return {
        "schema_version": 1,
        "status": "development_only_label_free_descriptor_cache",
        "seed": SEED,
        "subset_seed": SEED,
        "corruption_seed": SEED,
        "n_per_scenario": 1000,
        "scenario_count": len(SCENARIOS),
        "total_rows": 1000 * len(SCENARIOS),
        "scenarios": list(SCENARIOS),
        "ids_source": cache_protocol.get("ids_source"),
        "ids_sha256": cache_protocol["ids_sha256"],
        "input_cache": str(cache.resolve()),
        "input_cache_protocol_sha256": sha256(cache / "protocol.json"),
        "input_array_sha256": hashes,
        "input_cache_source_sha256": cache_protocol.get("source_sha256", {}),
        "risk_bank": list(RISK_BANK),
        "risk_shape": [4, 2],
        "descriptor_shape": [18],
        "feature_order": list(RISK_FEATURE_ORDER),
        "uses_class_labels": False,
        "reads_imagenet_metadata": False,
        "reads_model_predictions": False,
        "reads_results_or_untouched_data": False,
        "workers": workers,
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
    }


def _init_worker(array_path: str) -> None:
    global _WORKER_IMAGES
    _WORKER_IMAGES = np.load(array_path, mmap_mode="r")


def _extract_index(index: int) -> tuple[int, np.ndarray, str]:
    if _WORKER_IMAGES is None:  # pragma: no cover - process initializer guards this
        raise RuntimeError("descriptor worker was not initialized")
    image = np.asarray(_WORKER_IMAGES[index]).copy()
    raw_hash = hashlib.sha256(image.tobytes()).hexdigest()
    risks = masked_risks_fast(Image.fromarray(image, mode="RGB"), RISK_BANK)
    return index, risks, raw_hash


def validate_arrays(
    risks: np.ndarray,
    features: np.ndarray,
    samples: np.ndarray,
    raw_hashes: np.ndarray,
    n: int,
) -> None:
    if risks.shape != (n, 4, 2) or risks.dtype != np.float32:
        raise RuntimeError(f"invalid risk array {risks.shape}/{risks.dtype}")
    if features.shape != (n, 18) or features.dtype != np.float64:
        raise RuntimeError(f"invalid descriptor array {features.shape}/{features.dtype}")
    if not np.isfinite(risks).all() or np.any(risks < 0):
        raise RuntimeError("risk array contains invalid values")
    if not np.isfinite(features).all():
        raise RuntimeError("descriptor array contains invalid values")
    np.testing.assert_array_equal(samples, np.arange(n, dtype=np.int64))
    if raw_hashes.shape != (n,):
        raise RuntimeError("raw-image hash vector has invalid shape")
    for value in raw_hashes:
        text = value.decode() if isinstance(value, bytes) else str(value)
        if not re.fullmatch(r"[0-9a-f]{64}", text):
            raise RuntimeError("raw-image hash vector contains an invalid value")
    recomputed = risk_features(risks)
    if not np.array_equal(features, recomputed):
        raise RuntimeError("stored 18-D descriptors do not match frozen risk_features")


def validate_shard(
    path: Path,
    *,
    scenario: str,
    source_protocol_sha256: str,
    source_array_sha256: str,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as stored:
        required = {
            "risks",
            "features",
            "samples",
            "raw_sha256",
            "scenario",
            "source_protocol_sha256",
            "source_array_sha256",
            "feature_order",
        }
        if required - set(stored.files):
            raise RuntimeError(f"descriptor shard is incomplete: {path}")
        if str(stored["scenario"].item()) != scenario:
            raise RuntimeError(f"descriptor shard scenario mismatch: {path}")
        if str(stored["source_protocol_sha256"].item()) != source_protocol_sha256:
            raise RuntimeError(f"descriptor shard protocol mismatch: {path}")
        if str(stored["source_array_sha256"].item()) != source_array_sha256:
            raise RuntimeError(f"descriptor shard source-array mismatch: {path}")
        if tuple(str(value) for value in stored["feature_order"].tolist()) != RISK_FEATURE_ORDER:
            raise RuntimeError(f"descriptor shard feature order mismatch: {path}")
        risks = stored["risks"]
        features = stored["features"]
        samples = stored["samples"]
        raw_hashes = stored["raw_sha256"]
        validate_arrays(risks, features, samples, raw_hashes, 1000)
        return {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "rows": len(samples),
            "raw_hash_sequence_sha256": hashlib.sha256(
                np.asarray(raw_hashes, dtype="S64").tobytes()
            ).hexdigest(),
        }


def build_shard(
    array_path: Path,
    output_path: Path,
    scenario: str,
    source_protocol_sha256: str,
    source_array_sha256: str,
    workers: int,
) -> dict[str, Any]:
    context = mp.get_context("fork")
    risks = np.empty((1000, 4, 2), dtype=np.float32)
    raw_hashes = np.empty(1000, dtype="S64")
    with context.Pool(
        workers,
        initializer=_init_worker,
        initargs=(str(array_path),),
        maxtasksperchild=250,
    ) as pool:
        for index, values, raw_hash in pool.imap(_extract_index, range(1000), chunksize=4):
            risks[index] = values
            raw_hashes[index] = raw_hash
    features = risk_features(risks)
    samples = np.arange(1000, dtype=np.int64)
    validate_arrays(risks, features, samples, raw_hashes, 1000)
    atomic_npz(
        output_path,
        risks=risks,
        features=features,
        samples=samples,
        raw_sha256=raw_hashes,
        scenario=np.asarray(scenario),
        source_protocol_sha256=np.asarray(source_protocol_sha256),
        source_array_sha256=np.asarray(source_array_sha256),
        feature_order=np.asarray(RISK_FEATURE_ORDER),
    )
    return validate_shard(
        output_path,
        scenario=scenario,
        source_protocol_sha256=source_protocol_sha256,
        source_array_sha256=source_array_sha256,
    )


def assemble(
    output_dir: Path,
    source_protocol_sha256: str,
    shard_artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    risks, features, samples, hashes, scenarios, keys = [], [], [], [], [], []
    for scenario in SCENARIOS:
        path = Path(shard_artifacts[scenario]["path"])
        with np.load(path, allow_pickle=False) as stored:
            risks.append(stored["risks"])
            features.append(stored["features"])
            samples.append(stored["samples"])
            hashes.append(stored["raw_sha256"])
        scenarios.extend([scenario] * 1000)
        keys.extend(f"{scenario}#{index}" for index in range(1000))
    risk_array = np.concatenate(risks, axis=0)
    feature_array = np.concatenate(features, axis=0)
    sample_array = np.concatenate(samples, axis=0)
    raw_hash_array = np.concatenate(hashes, axis=0)
    if risk_array.shape != (15000, 4, 2):
        raise RuntimeError("assembled risk matrix has an invalid shape")
    if feature_array.shape != (15000, 18) or not np.isfinite(feature_array).all():
        raise RuntimeError("assembled descriptor matrix has an invalid shape/value")
    if not np.array_equal(feature_array, risk_features(risk_array)):
        raise RuntimeError("assembled descriptors do not match frozen risk_features")
    if len(set(keys)) != 15000:
        raise RuntimeError("assembled descriptor keys are not unique")
    artifact = output_dir / "descriptors_dev15.npz"
    atomic_npz(
        artifact,
        keys=np.asarray(keys),
        scenarios=np.asarray(scenarios),
        samples=sample_array,
        raw_sha256=raw_hash_array,
        risks=risk_array,
        features=feature_array,
        feature_order=np.asarray(RISK_FEATURE_ORDER),
        source_protocol_sha256=np.asarray(source_protocol_sha256),
    )
    return {
        "path": str(artifact.resolve()),
        "sha256": sha256(artifact),
        "rows": 15000,
        "risk_shape": list(risk_array.shape),
        "feature_shape": list(feature_array.shape),
        "feature_dtype": str(feature_array.dtype),
        "finite": bool(np.isfinite(feature_array).all()),
        "key_sequence_sha256": hashlib.sha256(
            "\n".join(keys).encode()
        ).hexdigest(),
        "raw_hash_sequence_sha256": hashlib.sha256(
            np.asarray(raw_hash_array, dtype="S64").tobytes()
        ).hexdigest(),
        "feature_bytes_sha256": hashlib.sha256(
            np.asarray(feature_array, dtype="<f8").tobytes()
        ).hexdigest(),
    }


def validate_final(output_dir: Path, source_protocol_sha256: str) -> dict[str, Any]:
    audit_path = output_dir / "audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("source_protocol_sha256") != source_protocol_sha256:
        raise RuntimeError("final descriptor audit refers to another source protocol")
    artifact_entry = audit.get("artifact", {})
    artifact = Path(artifact_entry.get("path", ""))
    if not artifact.is_file() or sha256(artifact) != artifact_entry.get("sha256"):
        raise RuntimeError("final descriptor artifact hash mismatch")
    with np.load(artifact, allow_pickle=False) as stored:
        if str(stored["source_protocol_sha256"].item()) != source_protocol_sha256:
            raise RuntimeError("final descriptor source protocol mismatch")
        risks = stored["risks"]
        features = stored["features"]
        if risks.shape != (15000, 4, 2) or features.shape != (15000, 18):
            raise RuntimeError("final descriptor artifact shape mismatch")
        if tuple(str(value) for value in stored["feature_order"].tolist()) != RISK_FEATURE_ORDER:
            raise RuntimeError("final descriptor feature order mismatch")
        if not np.isfinite(features).all() or not np.array_equal(features, risk_features(risks)):
            raise RuntimeError("final descriptor values failed finite/recompute audit")
        keys = [str(value) for value in stored["keys"].tolist()]
        expected_keys = [
            f"{scenario}#{index}" for scenario in SCENARIOS for index in range(1000)
        ]
        if keys != expected_keys:
            raise RuntimeError("final descriptor row order mismatch")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-cache",
        type=Path,
        default=ROOT / "cache/sparc_dev1k_corruptions_s5_dev15",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "cache/sparc_dev1k_masked_risk_descriptors_dev15",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        raise ValueError("workers must be in [1, 32]")
    cache = args.input_cache.resolve()
    output_dir = args.output_dir.resolve()
    cache_protocol, input_hashes = validate_input_cache(cache)
    protocol = source_protocol(cache, cache_protocol, input_hashes, args.workers)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError("descriptor source protocol changed; use a new output directory")
    else:
        if args.validate_only:
            raise FileNotFoundError("descriptor protocol does not exist")
        atomic_json(protocol_path, protocol)
    protocol_hash = sha256(protocol_path)
    if (output_dir / "audit.json").exists():
        audit = validate_final(output_dir, protocol_hash)
        print(json.dumps({"status": "verified_reuse", **audit["artifact"]}, indent=2))
        return
    if args.validate_only:
        raise FileNotFoundError("completed descriptor audit does not exist")

    shards = output_dir / "shards"
    shards.mkdir(exist_ok=True)
    artifacts = {}
    started = time.perf_counter()
    for number, scenario in enumerate(SCENARIOS, start=1):
        key = scenario_key(scenario)
        source_entry = cache_protocol["files"][key]
        array_path = cache / source_entry["path"]
        shard_path = shards / f"{key}.npz"
        if shard_path.exists():
            artifact = validate_shard(
                shard_path,
                scenario=scenario,
                source_protocol_sha256=protocol_hash,
                source_array_sha256=input_hashes[scenario],
            )
            action = "verified_skip"
        else:
            artifact = build_shard(
                array_path,
                shard_path,
                scenario,
                protocol_hash,
                input_hashes[scenario],
                args.workers,
            )
            action = "built"
        artifacts[scenario] = artifact
        print(
            json.dumps(
                {
                    "scenario": scenario,
                    "progress": f"{number}/{len(SCENARIOS)}",
                    "action": action,
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                }
            ),
            flush=True,
        )
    final = assemble(output_dir, protocol_hash, artifacts)
    audit = {
        "schema_version": 1,
        "status": "complete_development_only_descriptor_cache",
        "source_protocol_sha256": protocol_hash,
        "artifact": final,
        "shards": artifacts,
        "checks": {
            "scenario_order_exact": True,
            "sample_order_exact_0_to_999_per_scenario": True,
            "raw_sha256_recorded_per_row": True,
            "input_arrays_sha256_verified": True,
            "risk_shape_exact_15000x4x2": True,
            "descriptor_shape_exact_15000x18": True,
            "descriptor_finite": True,
            "descriptor_recomputed_from_risks_exact": True,
            "feature_order_exact": True,
            "class_labels_or_metadata_read": False,
            "results_or_untouched_data_read": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
        },
    }
    atomic_json(output_dir / "audit.json", audit)
    validate_final(output_dir, protocol_hash)
    print(json.dumps({"status": "complete", **final}, indent=2))


if __name__ == "__main__":
    main()
