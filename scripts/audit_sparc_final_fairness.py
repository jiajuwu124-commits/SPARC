#!/usr/bin/env python3
"""Independent post-run fairness and reproducibility audit for SPARC Full-15.

The auditor is outcome-preserving: it checks and reports every pre-specified
cell, never removes non-positive gains, and writes no experiment output.  It is
implemented separately from the frozen executor so a second code path verifies
pairing, hashes, seeds, reset semantics, memory limits, routing identity, and
development/test separation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_sparc_full15_table1 as table1  # noqa: E402

PAIR_FIELDS = (
    "key", "image_id", "sample", "target", "audit_domain", "raw_sha256",
    "descriptor_sha256", "stream_position", "batch", "batch_position",
)
PROTOCOL_PAIR_FIELDS = (
    "backend", "batch_size", "seed", "runner_sha256", "model_sha256",
    "prompt_sha256", "reset_policy", "native_reset_policy", "memory_fraction",
    "stream_keys_sha256", "stream_order_sha256", "test_ids_sequence_sha256",
)


class AuditError(RuntimeError):
    """Raised on any fairness or integrity violation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(rendered).hexdigest()


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def audit_result(output: Path, manifest_path: Path, *, expected_rows: int,
                 expected_backend: str, expected_scenario: str,
                 expected_runner: str) -> dict[str, Any]:
    manifest = load(manifest_path)
    result_path, protocol_path, complete_path = (
        output / "result.json", output / "protocol.json", output / "complete.json"
    )
    require(all(path.is_file() for path in (result_path, protocol_path, complete_path)),
            f"incomplete output: {output}")
    result, protocol, complete = load(result_path), load(protocol_path), load(complete_path)
    require(complete.get("result_sha256") == sha256(result_path),
            f"result completion hash mismatch: {output}")
    require(complete.get("protocol_sha256") == sha256(protocol_path),
            f"protocol completion hash mismatch: {output}")
    require(result.get("protocol_sha256") == sha256(protocol_path),
            f"result/protocol hash mismatch: {output}")
    require(result.get("protocol") == protocol, f"embedded protocol mismatch: {output}")
    records = result.get("records")
    require(isinstance(records, list) and len(records) == expected_rows,
            f"expected {expected_rows} records: {output}")
    records_hash = canonical_sha256(records)
    require(result.get("records_sha256") == records_hash
            and complete.get("records_sha256") == records_hash,
            f"canonical records hash mismatch: {output}")
    reconstructed = 100.0 * sum(bool(row.get("correct")) for row in records) / expected_rows
    require(math.isclose(float(result.get("accuracy")), reconstructed, abs_tol=1e-12),
            f"accuracy is not reconstructible: {output}")
    require(protocol.get("status") == "frozen_five_backend_routed_replay",
            f"unexpected protocol status: {output}")
    require(protocol.get("backend") == expected_backend and protocol.get("seed") == 1109,
            f"backend/seed mismatch: {output}")
    require(protocol.get("runner_sha256") == expected_runner,
            f"runner hash mismatch: {output}")
    require(protocol.get("batch_size") == 20,
            f"batch mismatch: {output}")
    require(float(protocol.get("memory_fraction")) <= 0.70
            and bool(protocol.get("memory_gate_pass")),
            f"memory gate violation: {output}")
    require(all(str(row.get("audit_domain")) == expected_scenario for row in records),
            f"scenario mismatch inside records: {output}")
    require(all(int(row.get("stream_position", -1)) == index
                for index, row in enumerate(records)),
            f"non-canonical stream positions: {output}")
    router = manifest.get("router", {}).get("artifact", {})
    require(protocol.get("router_sha256") == router.get("sha256"),
            f"manifest/protocol router mismatch: {output}")
    require(protocol.get("manifest_sha256") == sha256(manifest_path),
            f"manifest hash mismatch: {output}")
    return {
        "accuracy": reconstructed,
        "records": records,
        "protocol": protocol,
        "result_sha256": sha256(result_path),
        "protocol_sha256": sha256(protocol_path),
        "records_sha256": records_hash,
    }


def audit_pair(parent: dict[str, Any], plugin: dict[str, Any], *, label: str) -> dict[str, Any]:
    for field in PROTOCOL_PAIR_FIELDS:
        require(parent["protocol"].get(field) == plugin["protocol"].get(field),
                f"parent/plugin protocol mismatch in {field}: {label}")
    left, right = parent["records"], plugin["records"]
    require([[row.get(field) for field in PAIR_FIELDS] for row in left]
            == [[row.get(field) for field in PAIR_FIELDS] for row in right],
            f"row-level pairing mismatch: {label}")
    counts = Counter(str(row.get("expert")) for row in right)
    return {
        "parent_accuracy": parent["accuracy"],
        "plugin_accuracy": plugin["accuracy"],
        "gain_pp": plugin["accuracy"] - parent["accuracy"],
        "expert_counts": dict(sorted(counts.items())),
        "parent_result_sha256": parent["result_sha256"],
        "plugin_result_sha256": plugin["result_sha256"],
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(matrix_path: Path, lock_path: Path, summary_path: Path, output: Path) -> dict[str, Any]:
    matrix_path, lock_path, summary_path = map(Path.resolve, (matrix_path, lock_path, summary_path))
    matrix, lock, summary = map(load, (matrix_path, lock_path, summary_path))
    matrix_hash = sha256(matrix_path)
    require(matrix.get("status") == "frozen_before_evaluation", "matrix is not frozen")
    require(lock.get("status") == "frozen_before_any_evaluation", "lock status mismatch")
    require(lock.get("matrix_sha256") == matrix_hash, "lock/matrix hash mismatch")
    require(summary.get("manifest_sha256") == matrix_hash, "summary/matrix hash mismatch")
    require(summary.get("runner_sha256") == lock.get("runner_sha256"),
            "summary/lock runner mismatch")
    try:
        summary_cells = table1.validate_summary(summary)
    except table1.TableInputError as error:
        raise AuditError(str(error)) from error
    conditions = matrix.get("conditions")
    require(isinstance(conditions, list) and len(conditions) == 75,
            "matrix must contain 75 conditions")
    expected_keys = {(backend, scenario) for backend in table1.BACKENDS
                     for scenario in table1.SCENARIOS}
    keyed = {(str(row.get("backend")), str(row.get("scenario"))): row for row in conditions}
    require(set(keyed) == expected_keys and len(keyed) == len(conditions),
            "matrix condition inventory mismatch")

    routing_reference: dict[str, list[tuple[Any, ...]]] = {}
    pair_reports = []
    total_records = 0
    for backend in table1.BACKENDS:
        for scenario in table1.SCENARIOS:
            condition = keyed[backend, scenario]
            parent = audit_result(Path(condition["parent"]["output_dir"]),
                                  Path(condition["parent"]["manifest"]),
                                  expected_rows=2000, expected_backend=backend,
                                  expected_scenario=scenario,
                                  expected_runner=str(lock["runner_sha256"]))
            plugin = audit_result(Path(condition["plugin"]["output_dir"]),
                                  Path(condition["plugin"]["manifest"]),
                                  expected_rows=2000, expected_backend=backend,
                                  expected_scenario=scenario,
                                  expected_runner=str(lock["runner_sha256"]))
            pair = audit_pair(parent, plugin, label=f"{backend}/{scenario}")
            reported = summary_cells[backend, scenario]
            for field in ("parent_accuracy", "plugin_accuracy", "gain_pp"):
                require(math.isclose(float(pair[field]), float(reported[field]), abs_tol=1e-12),
                        f"summary does not reconstruct for {backend}/{scenario}/{field}")
            require(pair["expert_counts"] == reported["expert_counts"],
                    f"summary expert counts mismatch: {backend}/{scenario}")
            route = [(row.get("key"), row.get("image_id"), row.get("descriptor_sha256"),
                      row.get("expert"), row.get("expert_index"), row.get("decision"))
                     for row in plugin["records"]]
            if scenario in routing_reference:
                require(route == routing_reference[scenario],
                        f"backend-dependent routing sequence: {backend}/{scenario}")
            else:
                routing_reference[scenario] = route
            pair_reports.append({"backend": backend, "scenario": scenario, **pair})
            total_records += len(parent["records"]) + len(plugin["records"])

    first_manifest = load(Path(keyed[table1.BACKENDS[0], table1.SCENARIOS[0]]["plugin"]["manifest"]))
    lineage_entry = first_manifest.get("freeze_provenance", {}).get("training_test_lineage_report", {})
    lineage_path = Path(str(lineage_entry.get("path", "")))
    require(lineage_path.is_file() and sha256(lineage_path) == lineage_entry.get("sha256"),
            "lineage report is missing or hash-mismatched")
    lineage = load(lineage_path)
    require(lineage.get("status") == "passed_training_test_lineage_audit"
            and lineage.get("lineage_contract", {}).get("development_test_overlap") == 0
            and lineage.get("seed") == 1109,
            "training/test lineage gate failed")

    report = {
        "schema_version": 1,
        "status": "passed_independent_full15_fairness_audit",
        "matrix": str(matrix_path), "matrix_sha256": matrix_hash,
        "lock": str(lock_path), "lock_sha256": sha256(lock_path),
        "summary": str(summary_path), "summary_sha256": sha256(summary_path),
        "seed": 1109, "conditions": 75, "result_files": 150,
        "records_checked": total_records,
        "pairing_fields": list(PAIR_FIELDS),
        "protocol_pair_fields": list(PROTOCOL_PAIR_FIELDS),
        "backend_independent_routing_sequences": True,
        "development_test_overlap": 0,
        "all_negative_or_zero_cells_retained": True,
        "acceptance": summary["acceptance"],
        "pairs": pair_reports,
    }
    atomic_json(output.resolve(), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path,
                        default=ROOT / "configs/sparc_untouched2k_five_backend_matrix.frozen.json")
    parser.add_argument("--lock", type=Path,
                        default=ROOT / "configs/sparc_untouched2k_five_backend_matrix.lock.json")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "results/sparc_untouched2k_five_backend_summary_v1/summary.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/sparc_untouched2k_five_backend_summary_v1/independent_fairness_audit.json")
    args = parser.parse_args()
    try:
        report = run(args.matrix, args.lock, args.summary, args.output)
    except AuditError as error:
        print(json.dumps({"status": "failed_independent_fairness_audit", "error": str(error)}, indent=2))
        raise SystemExit(2)
    print(json.dumps({key: report[key] for key in (
        "status", "conditions", "result_files", "records_checked",
        "backend_independent_routing_sequences", "development_test_overlap",
    )}, indent=2))


if __name__ == "__main__":
    main()
