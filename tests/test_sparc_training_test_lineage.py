from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_sparc_training_test_lineage.py"
SPEC = importlib.util.spec_from_file_location("sparc_training_test_lineage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_ids(path: Path, values: list[str]) -> Path:
    path.write_text("\n".join(values) + "\n")
    return path


def make_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    sources = {
        "development": write_ids(
            tmp_path / "development.txt",
            [f"n{index % 1000:08d}/dev_{index:04d}.JPEG" for index in range(4000)],
        ),
        "diagnostic_a": write_ids(
            tmp_path / "diagnostic_a.txt",
            [f"n{index:08d}/a.JPEG" for index in range(1000, 2000)],
        ),
        "diagnostic_b": write_ids(
            tmp_path / "diagnostic_b.txt",
            [f"n{index:08d}/b.JPEG" for index in range(2000, 3000)],
        ),
    }
    roles = MODULE.SPLIT_ROLES
    splits = {}
    for name, path in sources.items():
        source_ids = path.read_text().splitlines()
        selected = MODULE.reconstruct_exact_1k(source_ids)
        splits[name] = {
            "role": roles[name], "n": 1000, "ids_source": str(path),
            "ids_source_sha256": file_hash(path),
            "ids_sequence_sha256": MODULE.sequence_sha256(selected),
        }
    protocol = {
        "status": "redesign_complete_not_final_tested", "seed": 1109,
        "training_rows": 2000, "validation_rows": 1000,
        "untouched_or_final_inputs": [], "splits": splits,
        "split_overlaps": {
            "development__diagnostic_a": 0,
            "development__diagnostic_b": 0,
            "diagnostic_a__diagnostic_b": 0,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    test_ids = write_ids(
        tmp_path / "untouched.txt",
        [f"n{index:08d}/test.JPEG" for index in range(5000, 7000)],
    )
    return protocol_path, test_ids, protocol


def test_pass_reconstructs_three_splits_and_zero_test_overlap(tmp_path: Path) -> None:
    protocol, test_ids, _ = make_fixture(tmp_path)
    report = MODULE.audit_lineage(protocol, test_ids)
    assert report["ready"] is True
    assert report["lineage_contract"]["development_selection_union_count"] == 3000
    assert report["lineage_contract"]["development_test_overlap"] == 0
    assert report["splits"]["development"]["selection"] == "seed1109_stratified_first_1k"
    assert report["splits"]["diagnostic_a"]["selection"] == "preserve_frozen_1k_file_order"
    assert report["untouched_prediction_results_read"] is False


def test_sequence_hash_is_independently_reconstructed(tmp_path: Path) -> None:
    protocol_path, test_ids, protocol = make_fixture(tmp_path)
    protocol["splits"]["development"]["ids_sequence_sha256"] = "0" * 64
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(MODULE.LineageError, match="reconstructed sequence"):
        MODULE.audit_lineage(protocol_path, test_ids)


def test_internal_overlap_is_rejected(tmp_path: Path) -> None:
    protocol_path, test_ids, protocol = make_fixture(tmp_path)
    shared = Path(protocol["splits"]["diagnostic_a"]["ids_source"])
    protocol["splits"]["diagnostic_b"]["ids_source"] = str(shared)
    protocol["splits"]["diagnostic_b"]["ids_source_sha256"] = file_hash(shared)
    protocol["splits"]["diagnostic_b"]["ids_sequence_sha256"] = (
        protocol["splits"]["diagnostic_a"]["ids_sequence_sha256"]
    )
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(MODULE.LineageError, match="redesign splits overlap"):
        MODULE.audit_lineage(protocol_path, test_ids)


def test_untouched_overlap_is_rejected(tmp_path: Path) -> None:
    protocol_path, test_ids, protocol = make_fixture(tmp_path)
    selected = Path(protocol["splits"]["diagnostic_b"]["ids_source"]).read_text().splitlines()
    untouched = test_ids.read_text().splitlines()
    write_ids(test_ids, [selected[0], *untouched[1:]])
    with pytest.raises(MODULE.LineageError, match="overlap untouched"):
        MODULE.audit_lineage(protocol_path, test_ids)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("training_rows", 1999, "training_rows"),
        ("validation_rows", 999, "validation_rows"),
        ("untouched_or_final_inputs", ["forbidden"], "untouched/final"),
    ],
)
def test_scope_and_size_contracts_are_hard_gates(
    tmp_path: Path, field: str, value, message: str,
) -> None:
    protocol_path, test_ids, protocol = make_fixture(tmp_path)
    protocol[field] = value
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(MODULE.LineageError, match=message):
        MODULE.audit_lineage(protocol_path, test_ids)


def test_cli_writes_atomic_hash_bound_report_and_failure(tmp_path: Path) -> None:
    protocol, test_ids, _ = make_fixture(tmp_path)
    output = tmp_path / "report.json"
    command = [sys.executable, str(SCRIPT), "--training-protocol", str(protocol),
               "--test-ids", str(test_ids), "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "passed_training_test_lineage_audit"
    assert report["auditor_sha256"] == file_hash(SCRIPT)
    assert not list(tmp_path.glob("report.json.partial-*"))

    bad = json.loads(protocol.read_text())
    bad["untouched_or_final_inputs"] = ["test"]
    protocol.write_text(json.dumps(bad))
    failed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert failed.returncode == 1
    failure = json.loads(output.read_text())
    assert failure["status"] == "blocked_training_test_lineage_audit"
    assert failure["ready"] is False
