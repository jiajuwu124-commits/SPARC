#!/usr/bin/env python3
"""Freeze one ConditionalResidualRestorer scale across declared backends.

The tool consumes only development result records.  It chooses the scale that
maximizes the minimum paired accuracy gain across the declared backends; exact
ties use the smaller scale.  A manifest is emitted only when that worst-backend
gain is strictly greater than the predeclared threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SEED = 1109
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


def records_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [paired_key(record) for record in records], separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_result(
    path: Path,
    *,
    expected_protocol_sha256: str,
    backend: str,
    method: str,
    scenario: str,
    n: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    details = payload.get("protocol", {})
    expected = {
        "protocol_sha256": expected_protocol_sha256,
        "backend": backend,
        "method": method,
        "scenario": scenario,
        "seed": SEED,
    }
    if details != expected:
        raise RuntimeError(f"result protocol mismatch: {path}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != n:
        raise RuntimeError(f"result record count mismatch: {path}")
    seen = set()
    for position, record in enumerate(records):
        if not isinstance(record, dict) or any(field not in record for field in PAIR_FIELDS):
            raise RuntimeError(f"result record schema mismatch: {path}")
        if int(record["stream_position"]) != position:
            raise RuntimeError(f"non-contiguous stream position: {path}")
        sample = int(record["sample"])
        if sample in seen or not 0 <= sample < n:
            raise RuntimeError(f"duplicate or invalid sample: {path}")
        seen.add(sample)
        if not re.fullmatch(r"[0-9a-f]{64}", str(record["raw_sha256"])):
            raise RuntimeError(f"invalid raw hash: {path}")
    accuracy = 100.0 * sum(bool(record.get("correct")) for record in records) / n
    stored_accuracy = float(payload.get("accuracy", math.nan))
    if not math.isfinite(stored_accuracy) or abs(stored_accuracy - accuracy) > 1e-12:
        raise RuntimeError(f"stored accuracy does not match records: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "accuracy": stored_accuracy,
        "pair_key_sha256": records_sha256(records),
        "records": records,
    }


def collect_results(
    development_dir: Path,
    checkpoint: Path,
    scenario: str,
    backends: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    protocol_path = development_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    checkpoint_hash = sha256(checkpoint)
    if protocol.get("status") != "development":
        raise RuntimeError("scale freezing requires status=development evidence")
    if int(protocol.get("seed", -1)) != SEED:
        raise RuntimeError("development evidence must use seed 1109")
    if protocol.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("development evidence and checkpoint hashes differ")
    scenarios = protocol.get("scenarios")
    if not isinstance(scenarios, list) or scenario not in scenarios:
        raise RuntimeError("declared scenario is absent from development evidence")
    if list(protocol.get("backends", [])) != backends:
        raise RuntimeError("declared backend order does not match development protocol")
    scales = [float(value) for value in protocol.get("restorer_scales", [])]
    if not scales or len(set(scales)) != len(scales):
        raise RuntimeError("development protocol requires unique residual scales")
    if any(not math.isfinite(scale) or not 0.0 <= scale <= 2.0 for scale in scales):
        raise RuntimeError("development residual scale is outside [0, 2]")

    expected_methods = {"identity"} | {method_for_scale(scale) for scale in scales}
    indexed: dict[tuple[str, str], Path] = {}
    protocol_hash = sha256(protocol_path)
    for path in sorted(development_dir.glob("*.json")):
        if path.name in {"protocol.json", "matched_pair_audit.json"}:
            continue
        payload = json.loads(path.read_text())
        details = payload.get("protocol")
        if not isinstance(details, dict):
            continue
        backend = details.get("backend")
        method = details.get("method")
        if backend not in backends or method not in expected_methods:
            continue
        if details.get("scenario") != scenario:
            continue
        key = (str(backend), str(method))
        if key in indexed:
            raise RuntimeError(f"duplicate result for {key}")
        indexed[key] = path

    evidence: dict[str, dict[str, dict[str, Any]]] = {}
    cross_backend_identity_key = None
    for backend in backends:
        missing = [
            method for method in expected_methods if (backend, method) not in indexed
        ]
        if missing:
            raise RuntimeError(f"missing {backend} results: {sorted(missing)}")
        identity = validate_result(
            indexed[(backend, "identity")],
            expected_protocol_sha256=protocol_hash,
            backend=backend,
            method="identity",
            scenario=scenario,
            n=int(protocol["n"]),
        )
        identity_key = identity["pair_key_sha256"]
        if cross_backend_identity_key is None:
            cross_backend_identity_key = identity_key
        elif identity_key != cross_backend_identity_key:
            raise RuntimeError("backend identity streams are not exactly paired")
        evidence[backend] = {"identity": identity}
        identity_records = [paired_key(record) for record in identity.pop("records")]
        for scale in scales:
            method = method_for_scale(scale)
            candidate = validate_result(
                indexed[(backend, method)],
                expected_protocol_sha256=protocol_hash,
                backend=backend,
                method=method,
                scenario=scenario,
                n=int(protocol["n"]),
            )
            candidate_records = [paired_key(record) for record in candidate.pop("records")]
            if candidate_records != identity_records:
                raise RuntimeError(
                    f"{backend}/{method} is not paired with its identity stream"
                )
            evidence[backend][method] = candidate
    protocol_summary = {
        "path": str(protocol_path.resolve()),
        "sha256": protocol_hash,
        "checkpoint_sha256": checkpoint_hash,
        "n": int(protocol["n"]),
        "scales": scales,
        "pair_key_sha256": cross_backend_identity_key,
    }
    return protocol_summary, evidence


def freeze_manifest(
    development_dir: Path,
    checkpoint: Path,
    scenario: str,
    backends: list[str],
    minimum_gain: float,
) -> dict[str, Any]:
    if len(backends) < 2 or len(set(backends)) != len(backends):
        raise ValueError("declare at least two unique backends")
    if any(backend not in {"clip", "mint"} for backend in backends):
        raise ValueError("supported backends are clip and mint")
    protocol, evidence = collect_results(
        development_dir, checkpoint, scenario, backends
    )
    candidates = []
    for scale in protocol["scales"]:
        method = method_for_scale(scale)
        gains = {
            backend: evidence[backend][method]["accuracy"]
            - evidence[backend]["identity"]["accuracy"]
            for backend in backends
        }
        candidates.append(
            {
                "residual_scale": scale,
                "method": method,
                "accuracy_by_backend": {
                    backend: evidence[backend][method]["accuracy"]
                    for backend in backends
                },
                "gain_pp_by_backend": gains,
                "minimum_gain_pp": min(gains.values()),
                "mean_gain_pp": sum(gains.values()) / len(gains),
                "result_artifacts": {
                    backend: {
                        key: value
                        for key, value in evidence[backend][method].items()
                        if key != "accuracy"
                    }
                    for backend in backends
                },
            }
        )
    selected = sorted(
        candidates,
        key=lambda row: (-row["minimum_gain_pp"], row["residual_scale"]),
    )[0]
    if selected["minimum_gain_pp"] <= minimum_gain:
        raise RuntimeError(
            f"best minimum gain {selected['minimum_gain_pp']:.6f} is not strictly "
            f"greater than {minimum_gain:.6f} pp"
        )
    return {
        "schema_version": 1,
        "status": "frozen_after_development",
        "method": "ConditionalResidualRestorer",
        "scenario": scenario,
        "seed": SEED,
        "backends": backends,
        "selection_rule": "maximize minimum paired gain across declared backends; exact ties use smaller scale",
        "minimum_gain_pp_strictly_greater_than": minimum_gain,
        "selected_residual_scale": selected["residual_scale"],
        "selected_minimum_gain_pp": selected["minimum_gain_pp"],
        "selected_gain_pp_by_backend": selected["gain_pp_by_backend"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "development_protocol": {
            "path": protocol["path"],
            "sha256": protocol["sha256"],
            "n": protocol["n"],
            "pair_key_sha256": protocol["pair_key_sha256"],
        },
        "identity_artifacts": {
            backend: evidence[backend]["identity"] for backend in backends
        },
        "candidates": candidates,
        "uses_diagnostic_or_confirmation_results": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--backends", default="clip,mint")
    parser.add_argument("--minimum-gain", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    backends = [value for value in args.backends.split(",") if value]
    manifest = freeze_manifest(
        args.development_dir.resolve(),
        args.checkpoint.resolve(),
        args.scenario,
        backends,
        args.minimum_gain,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
