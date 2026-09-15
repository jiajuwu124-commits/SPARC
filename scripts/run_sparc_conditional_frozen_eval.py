#!/usr/bin/env python3
"""Run or validate a diagnostic evaluation with one development-frozen scale.

The wrapper deliberately exposes no residual-scale argument.  It reads the
single scale, scenario, checkpoint hash, seed, and backend set from a frozen
development manifest, invokes the existing evaluator, and then independently
checks exact identity/plugin pairing and all stored accuracies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEED = 1109
BATCH_SIZE = 20
PAIR_FIELDS = ("sample", "target", "batch", "stream_position", "raw_sha256")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def method_for_scale(scale: float) -> str:
    return (
        "conditional_restorer"
        if scale == 1.0
        else f"conditional_restorer_scale{str(scale).replace('.', 'p')}"
    )


def paired_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in PAIR_FIELDS)


def paired_sha256(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        [paired_key(record) for record in records], separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_frozen_manifest(path: Path, checkpoint: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported frozen-scale manifest schema")
    if manifest.get("status") != "frozen_after_development":
        raise RuntimeError("diagnostic evaluation requires a development-frozen scale")
    if manifest.get("method") != "ConditionalResidualRestorer":
        raise RuntimeError("frozen manifest is for another method")
    if int(manifest.get("seed", -1)) != SEED:
        raise RuntimeError("frozen scale must use seed 1109")
    if manifest.get("uses_diagnostic_or_confirmation_results") is not False:
        raise RuntimeError("scale manifest does not exclude diagnostic/confirmation selection")
    scenario = manifest.get("scenario")
    if not isinstance(scenario, str) or not re.fullmatch(r"[^:]+:[1-9][0-9]*", scenario):
        raise ValueError("frozen manifest has an invalid scenario")
    backends = manifest.get("backends")
    if (
        not isinstance(backends, list)
        or len(backends) < 2
        or len(set(backends)) != len(backends)
        or any(value not in {"clip", "mint"} for value in backends)
    ):
        raise ValueError("frozen manifest requires at least two unique supported backends")
    scale = float(manifest.get("selected_residual_scale", math.nan))
    if not math.isfinite(scale) or not 0.0 <= scale <= 2.0:
        raise ValueError("frozen residual scale is outside [0, 2]")
    threshold = float(manifest.get("minimum_gain_pp_strictly_greater_than", math.nan))
    minimum_gain = float(manifest.get("selected_minimum_gain_pp", math.nan))
    if not math.isfinite(threshold) or not math.isfinite(minimum_gain):
        raise RuntimeError("frozen manifest lacks its gain threshold evidence")
    if threshold < 1.5 or minimum_gain <= threshold:
        raise RuntimeError("frozen scale does not satisfy the strict >1.5 pp requirement")
    checkpoint_hash = sha256(checkpoint)
    if manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("checkpoint does not match the frozen scale manifest")

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("seed", -1)) != SEED:
        raise RuntimeError("conditional checkpoint was not trained with seed 1109")
    if not isinstance(payload.get("model"), dict) or "state_dict" not in payload:
        raise RuntimeError("conditional checkpoint schema is invalid")
    condition = scenario.rsplit(":", 1)[0]
    if condition not in payload.get("scenarios", []):
        raise RuntimeError("conditional checkpoint lacks the frozen scenario condition")
    return manifest


def validate_records(
    path: Path,
    *,
    protocol_sha256: str,
    backend: str,
    method: str,
    scenario: str,
    seed: int,
    n: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected_protocol = {
        "protocol_sha256": protocol_sha256,
        "backend": backend,
        "method": method,
        "scenario": scenario,
        "seed": seed,
    }
    if payload.get("protocol") != expected_protocol:
        raise RuntimeError(f"result protocol mismatch: {path}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != n:
        raise RuntimeError(f"result record count mismatch: {path}")
    samples = set()
    for position, record in enumerate(records):
        if any(field not in record for field in PAIR_FIELDS):
            raise RuntimeError(f"result record schema mismatch: {path}")
        sample = int(record["sample"])
        if sample in samples or not 0 <= sample < n:
            raise RuntimeError(f"duplicate or invalid sample: {path}")
        samples.add(sample)
        if int(record["stream_position"]) != position:
            raise RuntimeError(f"non-contiguous stream position: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record["raw_sha256"])):
            raise RuntimeError(f"invalid raw image hash: {path}")
    accuracy = 100.0 * sum(bool(record.get("correct")) for record in records) / n
    if abs(float(payload.get("accuracy", math.nan)) - accuracy) > 1e-12:
        raise RuntimeError(f"stored accuracy does not match records: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "accuracy": accuracy,
        "pair_key_sha256": paired_sha256(records),
        "records": records,
    }


def validate_output(
    output_dir: Path,
    frozen_manifest_path: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    frozen = validate_frozen_manifest(frozen_manifest_path, checkpoint)
    protocol_path = output_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    scenario = frozen["scenario"]
    backends = frozen["backends"]
    scale = float(frozen["selected_residual_scale"])
    if protocol.get("status") != "diagnostic":
        raise RuntimeError("output is not diagnostic evidence")
    if int(protocol.get("seed", -1)) != SEED:
        raise RuntimeError("diagnostic output does not use seed 1109")
    if protocol.get("checkpoint_sha256") != frozen["checkpoint_sha256"]:
        raise RuntimeError("diagnostic output checkpoint differs from frozen manifest")
    if protocol.get("scenarios") != [scenario]:
        raise RuntimeError("diagnostic output scenario differs from frozen manifest")
    if protocol.get("backends") != backends:
        raise RuntimeError("diagnostic output backends differ from frozen manifest")
    if int(protocol.get("batch_size", -1)) != BATCH_SIZE:
        raise RuntimeError("diagnostic evaluation must use batch size 20")
    scales = [float(value) for value in protocol.get("restorer_scales", [])]
    if scales != [scale]:
        raise RuntimeError("diagnostic output did not use exactly the frozen scale")
    n = int(protocol.get("n", -1))
    if n <= 0:
        raise RuntimeError("diagnostic protocol has invalid sample count")

    protocol_hash = sha256(protocol_path)
    method = method_for_scale(scale)
    indexed = {}
    for path in sorted(output_dir.glob("*.json")):
        if path.name in {
            "protocol.json",
            "matched_pair_audit.json",
            "frozen_scale_validation.json",
        }:
            continue
        payload = json.loads(path.read_text())
        details = payload.get("protocol")
        if not isinstance(details, dict) or details.get("scenario") != scenario:
            continue
        key = (details.get("backend"), details.get("method"))
        if key in indexed:
            raise RuntimeError(f"duplicate diagnostic result: {key}")
        indexed[key] = path
    expected_keys = {
        (backend, candidate)
        for backend in backends
        for candidate in ("identity", method)
    }
    if set(indexed) != expected_keys:
        raise RuntimeError(
            f"diagnostic result set mismatch; expected={sorted(expected_keys)}, "
            f"found={sorted(indexed)}"
        )

    artifacts = {}
    shared_identity_key = None
    for backend in backends:
        identity = validate_records(
            indexed[(backend, "identity")],
            protocol_sha256=protocol_hash,
            backend=backend,
            method="identity",
            scenario=scenario,
            seed=SEED,
            n=n,
        )
        plugin = validate_records(
            indexed[(backend, method)],
            protocol_sha256=protocol_hash,
            backend=backend,
            method=method,
            scenario=scenario,
            seed=SEED,
            n=n,
        )
        identity_keys = [paired_key(record) for record in identity.pop("records")]
        plugin_keys = [paired_key(record) for record in plugin.pop("records")]
        if identity_keys != plugin_keys:
            raise RuntimeError(f"{backend} identity/plugin records are not exactly paired")
        if shared_identity_key is None:
            shared_identity_key = identity["pair_key_sha256"]
        elif identity["pair_key_sha256"] != shared_identity_key:
            raise RuntimeError("backend identity streams are not exactly paired")
        artifacts[backend] = {
            "identity": identity,
            "plugin": plugin,
            "gain_pp": plugin["accuracy"] - identity["accuracy"],
        }
    native_audit_path = output_dir / "matched_pair_audit.json"
    native_audit = json.loads(native_audit_path.read_text())
    if native_audit.get("passed") is not True:
        raise RuntimeError("evaluator matched-pair audit did not pass")
    return {
        "schema_version": 1,
        "status": "validated_diagnostic_frozen_scale",
        "seed": SEED,
        "scenario": scenario,
        "backends": backends,
        "residual_scale": scale,
        "checkpoint_sha256": sha256(checkpoint),
        "frozen_scale_manifest_sha256": sha256(frozen_manifest_path),
        "diagnostic_protocol_sha256": protocol_hash,
        "evaluator_matched_pair_audit_sha256": sha256(native_audit_path),
        "pair_key_sha256": shared_identity_key,
        "artifacts": artifacts,
        "validator_source_sha256": sha256(Path(__file__)),
    }


def write_audit(path: Path, audit: dict[str, Any]) -> None:
    if path.exists() and json.loads(path.read_text()) != audit:
        raise RuntimeError("existing frozen-scale validation differs")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-scale-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--meta", type=Path, default=ROOT / "data/imagenet/meta.bin")
    parser.add_argument("--clip-cache", type=Path, default=Path.home() / ".cache/clip")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an existing output directory without launching the evaluator",
    )
    args = parser.parse_args()

    frozen_path = args.frozen_scale_manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    frozen = validate_frozen_manifest(frozen_path, checkpoint)
    output_dir = args.output_dir.resolve()
    if not args.validate_only:
        if args.ids is None or args.cache is None:
            raise ValueError("--ids and --cache are required unless --validate-only is used")
        if output_dir.exists():
            raise FileExistsError("diagnostic output directory already exists")
        command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_sparc_conditional_restorer.py"),
            "--ids",
            str(args.ids.resolve()),
            "--cache",
            str(args.cache.resolve()),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--status",
            "diagnostic",
            "--scenarios",
            frozen["scenario"],
            "--backends",
            ",".join(frozen["backends"]),
            "--n",
            str(args.n),
            "--subset-seed",
            str(SEED),
            "--seed",
            str(SEED),
            "--batch-size",
            str(BATCH_SIZE),
            "--workers",
            str(args.workers),
            "--restorer-scales",
            str(frozen["selected_residual_scale"]),
            "--meta",
            str(args.meta.resolve()),
            "--clip-cache",
            str(args.clip_cache.resolve()),
        ]
        subprocess.run(command, check=True)
    audit = validate_output(output_dir, frozen_path, checkpoint)
    write_audit(output_dir / "frozen_scale_validation.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
