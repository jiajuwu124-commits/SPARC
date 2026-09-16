#!/usr/bin/env python3
"""Audit the frozen deployment-Brightness training/test ID lineage.

This gate reads only the training protocol and ID lists.  It never opens an
image, checkpoint, descriptor, prediction, or untouched result.  Development,
diagnostic-A, and diagnostic-B selections are reconstructed with the exact
seed-1109 rule used by the trainer, then checked against the untouched 2K IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SEED = 1109
N_PER_SPLIT = 1000
EXPECTED_TEST_ROWS = 2000
SPLIT_ROLES = {
    "development": "redesign_training",
    "diagnostic_a": "redesign_training",
    "diagnostic_b": "redesign_validation_model_selection",
}
PAIR_NAMES = (
    ("development", "diagnostic_a"),
    ("development", "diagnostic_b"),
    ("diagnostic_a", "diagnostic_b"),
)


class LineageError(RuntimeError):
    """A frozen lineage invariant was not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sequence_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def resolve_from(base: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def read_unique_ids(path: Path, label: str) -> list[str]:
    if not path.is_file():
        raise LineageError(f"missing {label}: {path}")
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not values:
        raise LineageError(f"{label} is empty: {path}")
    if len(values) != len(set(values)):
        raise LineageError(f"{label} contains duplicate IDs: {path}")
    return values


def stratified_subset_exact(
    relative_paths: Sequence[str], n: int, seed: int, offset: int = 0,
) -> list[str]:
    """Byte-for-byte logic equivalent to the trainer's seed-1109 selector."""
    paths = [value.strip() for value in relative_paths if value.strip()]
    if n <= 0 or offset < 0 or offset + n > len(paths):
        raise LineageError("invalid stratified selection size/offset")
    if n >= len(paths) and offset == 0:
        return sorted(paths)
    by_class: dict[str, list[str]] = {}
    for value in paths:
        by_class.setdefault(value.split("/", 1)[0], []).append(value)
    if not by_class:
        raise LineageError("cannot stratify an empty ID source")
    rng = np.random.default_rng(seed)
    for values in by_class.values():
        rng.shuffle(values)
    classes = sorted(by_class)
    rng.shuffle(classes)
    selected: list[str] = []
    cursor = 0
    while len(selected) < offset + n:
        category = classes[cursor % len(classes)]
        row = cursor // len(classes)
        if row < len(by_class[category]):
            selected.append(by_class[category][row])
        cursor += 1
    return selected[offset : offset + n]


def reconstruct_exact_1k(source_ids: Sequence[str]) -> list[str]:
    # This conditional is intentional: the trainer preserves an already-frozen
    # 1K file's order, but stratifies a larger source list.
    if len(source_ids) == N_PER_SPLIT:
        return list(source_ids)
    return stratified_subset_exact(source_ids, N_PER_SPLIT, SEED)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LineageError(message)


def audit_lineage(training_protocol_path: Path, test_ids_path: Path) -> dict[str, Any]:
    training_protocol_path = training_protocol_path.resolve()
    test_ids_path = test_ids_path.resolve()
    require(training_protocol_path.is_file(), f"missing training protocol: {training_protocol_path}")
    try:
        protocol = json.loads(training_protocol_path.read_text())
    except json.JSONDecodeError as error:
        raise LineageError(f"invalid training protocol JSON: {error}") from error
    require(isinstance(protocol, dict), "training protocol root must be an object")
    require(protocol.get("status") == "redesign_complete_not_final_tested",
            "training protocol is not a completed pre-final redesign artifact")
    require(protocol.get("seed") == SEED, "training protocol seed must be 1109")
    require(protocol.get("training_rows") == 2000, "training_rows must equal 2000")
    require(protocol.get("validation_rows") == 1000, "validation_rows must equal 1000")
    require("untouched_or_final_inputs" in protocol,
            "training protocol must explicitly declare untouched_or_final_inputs")
    require(protocol.get("untouched_or_final_inputs") == [],
            "training protocol used or declared an untouched/final input")

    splits = protocol.get("splits")
    require(isinstance(splits, dict) and set(splits) == set(SPLIT_ROLES),
            "training protocol must contain exactly development, diagnostic_a, diagnostic_b")
    reconstructed: dict[str, list[str]] = {}
    split_reports: dict[str, Any] = {}
    for name, expected_role in SPLIT_ROLES.items():
        record = splits[name]
        require(isinstance(record, dict), f"split {name} must be an object")
        require(record.get("role") == expected_role, f"split {name} role mismatch")
        require(record.get("n") == N_PER_SPLIT, f"split {name} n must equal 1000")
        require(isinstance(record.get("ids_source"), str), f"split {name} lacks ids_source")
        source = resolve_from(training_protocol_path.parent, record["ids_source"])
        source_ids = read_unique_ids(source, f"{name} IDs")
        source_hash = sha256_file(source)
        require(source_hash == record.get("ids_source_sha256"),
                f"split {name} ID-source SHA-256 mismatch")
        selected = reconstruct_exact_1k(source_ids)
        require(len(selected) == N_PER_SPLIT and len(set(selected)) == N_PER_SPLIT,
                f"split {name} did not reconstruct 1,000 unique IDs")
        selected_hash = sequence_sha256(selected)
        require(selected_hash == record.get("ids_sequence_sha256"),
                f"split {name} reconstructed sequence SHA-256 mismatch")
        reconstructed[name] = selected
        split_reports[name] = {
            "role": expected_role,
            "n": len(selected),
            "ids_source": str(source),
            "ids_source_rows": len(source_ids),
            "ids_source_sha256": source_hash,
            "reconstructed_ids_sequence_sha256": selected_hash,
            "selection": (
                "preserve_frozen_1k_file_order" if len(source_ids) == N_PER_SPLIT
                else "seed1109_stratified_first_1k"
            ),
        }

    overlaps = {
        f"{left}__{right}": len(set(reconstructed[left]) & set(reconstructed[right]))
        for left, right in PAIR_NAMES
    }
    require(all(value == 0 for value in overlaps.values()),
            f"redesign splits overlap: {overlaps}")
    require(protocol.get("split_overlaps") == overlaps,
            "protocol split_overlaps does not equal independently reconstructed overlaps")
    development_union = set().union(*(set(values) for values in reconstructed.values()))
    require(len(development_union) == 3000,
            "development/A/B union must contain exactly 3,000 IDs")

    test_ids = read_unique_ids(test_ids_path, "untouched test IDs")
    require(len(test_ids) == EXPECTED_TEST_ROWS,
            f"untouched test ID file must contain exactly {EXPECTED_TEST_ROWS} IDs")
    test_overlap = sorted(development_union & set(test_ids))
    require(not test_overlap,
            f"training/selection IDs overlap untouched test IDs ({len(test_overlap)} rows)")

    lineage_contract = {
        "seed": SEED,
        "training_rows": 2000,
        "validation_rows": 1000,
        "development_selection_union_count": len(development_union),
        "development_selection_union_sha256": sequence_sha256(sorted(development_union)),
        "untouched_test_count": len(test_ids),
        "untouched_test_sequence_sha256": sequence_sha256(test_ids),
        "development_test_overlap": 0,
        "split_overlaps": overlaps,
        "split_sequence_sha256": {
            name: split_reports[name]["reconstructed_ids_sequence_sha256"]
            for name in SPLIT_ROLES
        },
    }
    return {
        "schema_version": 1,
        "status": "passed_training_test_lineage_audit",
        "ready": True,
        "seed": SEED,
        "auditor": str(Path(__file__).resolve()),
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "training_protocol": str(training_protocol_path),
        "training_protocol_sha256": sha256_file(training_protocol_path),
        "untouched_test_ids": str(test_ids_path),
        "untouched_test_ids_sha256": sha256_file(test_ids_path),
        "splits": split_reports,
        "lineage_contract": lineage_contract,
        "lineage_contract_sha256": canonical_sha256(lineage_contract),
        "checks": {
            "exact_seed1109_sequences_reconstructed": True,
            "development_internal_overlap": 0,
            "development_test_overlap": 0,
            "training_rows_exact": True,
            "validation_rows_exact": True,
            "untouched_or_final_inputs_empty": True,
        },
        "gpu_used": False,
        "untouched_prediction_results_read": False,
        "read_scope": ["training_protocol", "development_id_sources", "untouched_test_id_list"],
    }


def failure_payload(error: Exception, training_protocol: Path, test_ids: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "blocked_training_test_lineage_audit",
        "ready": False,
        "error": str(error),
        "auditor": str(Path(__file__).resolve()),
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "training_protocol": str(training_protocol.resolve()),
        "untouched_test_ids": str(test_ids.resolve()),
        "gpu_used": False,
        "untouched_prediction_results_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output in {args.training_protocol.resolve(), args.test_ids.resolve()}:
        raise SystemExit("refusing to overwrite an audit input")
    try:
        report = audit_lineage(args.training_protocol, args.test_ids)
    except (LineageError, OSError, KeyError, TypeError, ValueError) as error:
        report = failure_payload(error, args.training_protocol, args.test_ids)
        atomic_json(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1)
    atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
