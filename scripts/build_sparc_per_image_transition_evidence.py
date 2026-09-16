#!/usr/bin/env python3
"""Build compact per-image SPARC correctness-transition evidence.

The output contains no images or model features beyond the already released
18-D router descriptors.  It converts archived identity/expert prediction
records into an int8 matrix in {-1, 0, 1}, verifies raw-input hashes and the
frozen batch-utility targets, and records the complete source lineage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def index_result_files(config: dict) -> dict[tuple[str, str], Path]:
    primary = Path(config["matrix_dir"])
    supplemental = {
        name: Path(path) for name, path in config["supplemental_experts"].items()
    }
    wanted = {"identity", *config["experts"]}
    index: dict[tuple[str, str], Path] = {}
    for root in (primary, *supplemental.values()):
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            protocol = payload.get("protocol")
            records = payload.get("records")
            if not isinstance(protocol, dict) or not isinstance(records, list):
                continue
            expert = str(protocol.get("expert", ""))
            scenario = str(protocol.get("scenario", ""))
            if expert not in wanted or scenario not in config["scenarios"]:
                continue
            if expert == "identity" and root != primary:
                continue
            if expert in supplemental and root != supplemental[expert]:
                continue
            key = (expert, scenario)
            if key in index:
                raise RuntimeError(f"duplicate expert/scenario result: {key}")
            index[key] = path
    expected = {
        (expert, scenario)
        for expert in wanted
        for scenario in config["scenarios"]
    }
    missing = sorted(expected - set(index))
    if missing:
        raise RuntimeError(f"missing {len(missing)} expert/scenario files: {missing[:5]}")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    config_path = args.source_config.resolve()
    source_manifest_path = args.source_manifest.resolve()
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    config = json.loads(config_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    expected_hashes = source_manifest["source_sha256"]

    with np.load(dataset, allow_pickle=False) as stored:
        utilities = stored["utilities"].astype(np.float64)
        groups = stored["groups"].astype(str)
        batch_ids = stored["batch_ids"].astype(str)
        samples = stored["samples"].astype(np.int64)
        scenarios = stored["scenarios"].astype(str)
        raw_sha256 = stored["raw_sha256"].astype(str)
        expert_names = tuple(str(value) for value in stored["expert_names"])
        seed = int(stored["seed"].item())

    row_index = {
        (str(scenario), int(sample)): row
        for row, (scenario, sample) in enumerate(zip(scenarios, samples))
    }
    if len(row_index) != len(samples):
        raise RuntimeError("scenario/sample keys are not unique")
    file_index = index_result_files(config)
    identity = np.zeros(len(samples), dtype=bool)
    expert_correct = np.zeros((len(samples), len(expert_names)), dtype=bool)
    seen_identity = np.zeros(len(samples), dtype=bool)
    seen_experts = np.zeros_like(expert_correct, dtype=bool)
    lineage: dict[str, dict] = {}

    for scenario in config["scenarios"]:
        identity_path = file_index[("identity", scenario)]
        for slot, expert in enumerate(("identity", *expert_names)):
            path = identity_path if expert == "identity" else file_index[(expert, scenario)]
            actual_hash = sha256(path)
            if expected_hashes.get(str(path)) != actual_hash:
                raise RuntimeError(f"source hash mismatch or missing manifest entry: {path}")
            payload = json.loads(path.read_text())
            lineage[str(path)] = {
                "sha256": actual_hash,
                "records": len(payload["records"]),
            }
            for record in payload["records"]:
                key = (scenario, int(record["sample"]))
                if key not in row_index:
                    raise RuntimeError(f"unexpected source record: {key}")
                row = row_index[key]
                if str(record["raw_sha256"]) != raw_sha256[row]:
                    raise RuntimeError(f"raw-input mismatch at {key} for {expert}")
                correct = bool(record["correct"])
                if expert == "identity":
                    identity[row] = correct
                    seen_identity[row] = True
                else:
                    expert_correct[row, slot - 1] = correct
                    seen_experts[row, slot - 1] = True
    if not seen_identity.all() or not seen_experts.all():
        raise RuntimeError("incomplete per-image correctness reconstruction")

    deltas = expert_correct.astype(np.int8) - identity[:, None].astype(np.int8)
    maximum_error = 0.0
    for batch_id in np.unique(batch_ids):
        rows = np.flatnonzero(batch_ids == batch_id)
        if len(rows) != 20:
            raise RuntimeError(f"batch {batch_id} has {len(rows)} rows")
        if not np.allclose(utilities[rows], utilities[rows[0]], atol=0, rtol=0):
            raise RuntimeError(f"batch target varies within {batch_id}")
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(deltas[rows].mean(axis=0) - utilities[rows[0]]))),
        )
    if maximum_error > 1e-12:
        raise RuntimeError(f"transition evidence does not reconstruct utilities: {maximum_error}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".partial-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            deltas=deltas,
            identity_correct=identity,
            expert_correct=expert_correct,
            samples=samples,
            scenarios=scenarios,
            groups=groups,
            batch_ids=batch_ids,
            raw_sha256=raw_sha256,
            expert_names=np.asarray(expert_names),
            seed=np.asarray(seed, dtype=np.int64),
            dataset_sha256=np.asarray(sha256(dataset)),
        )
    temporary.replace(output)
    manifest = {
        "schema_version": 1,
        "status": "passed_compact_per_image_transition_evidence_build",
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "source_config_sha256": sha256(config_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "seed": seed,
        "rows": len(samples),
        "experts": len(expert_names),
        "source_prediction_files": len(lineage),
        "max_batch_target_reconstruction_error": maximum_error,
        "output": str(output),
        "output_sha256": sha256(output),
        "source_lineage": lineage,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in (
        "status", "rows", "experts", "source_prediction_files",
        "max_batch_target_reconstruction_error", "output_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
