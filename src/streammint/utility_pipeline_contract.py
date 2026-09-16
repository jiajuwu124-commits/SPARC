"""CPU-only contracts joining the expert matrix, risk cache, and utility router."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _decode_hashes(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _source_checks(source_hashes: dict[str, str], errors: list[str], prefix: str) -> int:
    checked = 0
    for relative, expected in source_hashes.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            errors.append(f"{prefix} source missing: {relative}")
            continue
        observed = sha256(path)
        if observed != expected:
            errors.append(f"{prefix} source hash mismatch: {relative}")
            continue
        checked += 1
    return checked


def validate_label_sources(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return a score-free readiness report and a legacy builder manifest if ready."""
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    base = manifest_path.parent
    errors: list[str] = []
    if manifest.get("schema_version") != 2:
        errors.append("label manifest schema_version must be 2")
    if manifest.get("status") != "development_utility_label_sources":
        errors.append("label manifest status must be development_utility_label_sources")
    if manifest.get("seed") != 1109:
        errors.append("label protocol requires seed 1109")
    if manifest.get("allow_missing_experts", False):
        errors.append("actual utility labels forbid allow_missing_experts")

    matrix_dir = resolve(base, manifest["matrix_dir"])
    descriptor_dir = resolve(base, manifest["descriptor_cache"])
    matrix_protocol_path = matrix_dir / "protocol.json"
    descriptor_protocol_path = descriptor_dir / "protocol.json"
    descriptor_artifact_path = descriptor_dir / manifest.get(
        "descriptor_artifact", "descriptors_dev15.npz"
    )
    descriptor_audit_path = descriptor_dir / "audit.json"
    deep_audit_path = descriptor_dir / "deep_raw_audit.json"
    required_paths = (
        matrix_protocol_path, descriptor_protocol_path, descriptor_artifact_path,
        descriptor_audit_path, deep_audit_path,
    )
    for path in required_paths:
        if not path.is_file():
            errors.append(f"required source missing: {path}")
    if errors:
        return {
            "schema_version": 1, "status": "blocked_missing_source",
            "ready": False, "errors": errors, "missing_matrix_entries": [],
        }, None

    matrix = read_json(matrix_protocol_path)
    descriptor = read_json(descriptor_protocol_path)
    descriptor_audit = read_json(descriptor_audit_path)
    deep_audit = read_json(deep_audit_path)
    expected = manifest.get("expected_sha256", {})
    actual_hashes = {
        "matrix_protocol": sha256(matrix_protocol_path),
        "descriptor_protocol": sha256(descriptor_protocol_path),
        "descriptor_artifact": sha256(descriptor_artifact_path),
        "descriptor_audit": sha256(descriptor_audit_path),
        "descriptor_deep_audit": sha256(deep_audit_path),
        "label_manifest": sha256(manifest_path),
    }
    for key, value in expected.items():
        if key.startswith("supplemental_matrix_protocol."):
            continue
        if actual_hashes.get(key) != value:
            errors.append(f"frozen source hash mismatch: {key}")

    scenarios = [str(value) for value in manifest.get("scenarios", [])]
    experts = [str(value) for value in manifest.get("experts", [])]
    identity_name = str(manifest.get("identity_expert", "identity"))
    matrix_scenarios = [str(value) for value in matrix.get("scenarios", [])]
    primary_matrix_experts = [str(value["name"]) for value in matrix.get("experts", [])]
    supplemental_specs = manifest.get("supplemental_experts", {})
    if not isinstance(supplemental_specs, dict):
        errors.append("supplemental_experts must be an expert-to-matrix-dir mapping")
        supplemental_specs = {}
    supplemental_names = [str(value) for value in supplemental_specs]
    matrix_experts = [identity_name] + experts
    descriptor_scenarios = [str(value) for value in descriptor.get("scenarios", [])]
    if scenarios != matrix_scenarios or scenarios != descriptor_scenarios:
        errors.append("scenario list/order differs across manifest, matrix, and descriptors")
    primary_expected = [identity_name] + [
        expert for expert in experts if expert not in supplemental_names
    ]
    if primary_expected != primary_matrix_experts:
        errors.append("primary expert list/order differs from matrix protocol")
    if any(expert not in experts for expert in supplemental_names):
        errors.append("supplemental expert is absent from the declared combined expert list")
    if len(scenarios) != 15 or len(set(scenarios)) != 15:
        errors.append("actual dev15 utility protocol requires 15 unique scenarios")
    if not experts or len(set(experts)) != len(experts) or identity_name in experts:
        errors.append("non-identity experts must be unique and non-empty")
    for field in ("seed", "subset_seed", "corruption_seed"):
        if matrix.get(field) != 1109 or descriptor.get(field) != 1109:
            errors.append(f"{field} must equal 1109 in matrix and descriptor protocols")
    if matrix.get("backend") != manifest.get("backend", "stateless_clip"):
        errors.append("matrix backend differs from label manifest")
    if matrix.get("n") != descriptor.get("n_per_scenario"):
        errors.append("per-scenario sample count differs")
    n = int(matrix.get("n", 0))
    if n <= 0 or descriptor.get("total_rows") != n * len(scenarios):
        errors.append("descriptor total row count is inconsistent")
    for field in ("ids_sha256",):
        if matrix.get(field) != descriptor.get(field):
            errors.append(f"matrix/descriptor {field} mismatch")
    if matrix.get("cache_protocol_sha256") != descriptor.get("input_cache_protocol_sha256"):
        errors.append("matrix/descriptor corruption-cache protocol hash mismatch")
    if matrix.get("cache_file_sha256") != descriptor.get("input_array_sha256"):
        errors.append("matrix/descriptor per-scenario corruption array hashes differ")
    ids_source = Path(matrix.get("ids_source", ""))
    if not ids_source.is_file() or sha256(ids_source) != matrix.get("ids_source_sha256"):
        errors.append("matrix ID source file is missing or hash-mismatched")
    if descriptor_audit.get("source_protocol_sha256") != actual_hashes["descriptor_protocol"]:
        errors.append("descriptor build audit does not bind current protocol")
    artifact = descriptor_audit.get("artifact", {})
    if artifact.get("sha256") != actual_hashes["descriptor_artifact"]:
        errors.append("descriptor build audit artifact hash mismatch")
    if deep_audit.get("descriptor_protocol_sha256") != actual_hashes["descriptor_protocol"]:
        errors.append("deep descriptor audit does not bind current protocol")
    if deep_audit.get("build_audit_sha256") != actual_hashes["descriptor_audit"]:
        errors.append("deep descriptor audit does not bind current build audit")
    if deep_audit.get("descriptor_artifact_sha256") != actual_hashes["descriptor_artifact"]:
        errors.append("deep descriptor audit artifact hash mismatch")
    if deep_audit.get("status") != "deep_raw_audit_passed":
        errors.append("deep descriptor audit is not passed")

    matrix_sources_checked = _source_checks(matrix.get("source_sha256", {}), errors, "matrix")
    descriptor_sources_checked = _source_checks(
        descriptor.get("source_sha256", {}), errors, "descriptor"
    )
    matrix_manifest_source = Path(matrix.get("manifest_source", ""))
    if (
        not matrix_manifest_source.is_file()
        or sha256(matrix_manifest_source) != matrix.get("manifest_sha256")
    ):
        errors.append("matrix runner manifest source is missing or hash-mismatched")

    descriptor_maps: dict[str, dict[int, str]] = {}
    descriptor_shards: dict[str, Path] = {}
    descriptor_order_hashes: dict[str, str] = {}
    try:
        with np.load(descriptor_artifact_path, allow_pickle=False) as stored:
            npz_scenarios = [str(value) for value in stored["scenarios"].tolist()]
            samples = stored["samples"].astype(np.int64)
            raw_hashes = _decode_hashes(stored["raw_sha256"])
            keys = [str(value) for value in stored["keys"].tolist()]
            features = stored["features"]
            source_protocol_sha = str(stored["source_protocol_sha256"].item())
            if source_protocol_sha != actual_hashes["descriptor_protocol"]:
                errors.append("descriptor NPZ source protocol hash mismatch")
            expected_scenario_rows = [scenario for scenario in scenarios for _ in range(n)]
            expected_samples = [sample for _scenario in scenarios for sample in range(n)]
            expected_keys = [
                f"{scenario}#{sample}" for scenario in scenarios for sample in range(n)
            ]
            if npz_scenarios != expected_scenario_rows:
                errors.append("descriptor NPZ scenario order is not exact")
            if samples.tolist() != expected_samples:
                errors.append("descriptor NPZ sample order is not exact 0..n-1 per scenario")
            if keys != expected_keys:
                errors.append("descriptor NPZ scenario/sample keys are not exact")
            if features.shape != (n * len(scenarios), 18) or not np.isfinite(features).all():
                errors.append("descriptor feature matrix shape/finite check failed")
            for scenario_index, scenario in enumerate(scenarios):
                start, stop = scenario_index * n, (scenario_index + 1) * n
                values = raw_hashes[start:stop]
                descriptor_maps[scenario] = dict(zip(range(n), values))
                descriptor_order_hashes[scenario] = hashlib.sha256(
                    np.asarray(values, dtype="S64").tobytes()
                ).hexdigest()
    except (KeyError, ValueError, OSError) as error:
        errors.append(f"descriptor NPZ validation failed: {error}")

    for scenario in scenarios:
        shard_entry = descriptor_audit.get("shards", {}).get(scenario)
        if not isinstance(shard_entry, dict):
            errors.append(f"descriptor shard audit missing: {scenario}")
            continue
        shard = Path(shard_entry.get("path", ""))
        descriptor_shards[scenario] = shard
        if not shard.is_file() or sha256(shard) != shard_entry.get("sha256"):
            errors.append(f"descriptor shard missing/hash mismatch: {scenario}")
        if shard_entry.get("raw_hash_sequence_sha256") != descriptor_order_hashes.get(scenario):
            errors.append(f"descriptor shard raw-hash order mismatch: {scenario}")
        if not shard.is_file():
            continue
        try:
            with np.load(shard, allow_pickle=False) as stored:
                shard_samples = stored["samples"].astype(np.int64)
                shard_raw = _decode_hashes(stored["raw_sha256"])
                shard_scenario = str(stored["scenario"].item())
                shard_protocol = str(stored["source_protocol_sha256"].item())
                shard_array = str(stored["source_array_sha256"].item())
                shard_risks = stored["risks"]
                shard_features = stored["features"]
            if shard_samples.tolist() != list(range(n)):
                errors.append(f"descriptor shard sample order mismatch: {scenario}")
            if shard_raw != [descriptor_maps.get(scenario, {}).get(sample) for sample in range(n)]:
                errors.append(f"descriptor shard raw hashes differ from artifact: {scenario}")
            if shard_scenario != scenario:
                errors.append(f"descriptor shard scenario mismatch: {scenario}")
            if shard_protocol != actual_hashes["descriptor_protocol"]:
                errors.append(f"descriptor shard source protocol mismatch: {scenario}")
            if shard_array != descriptor.get("input_array_sha256", {}).get(scenario):
                errors.append(f"descriptor shard source array mismatch: {scenario}")
            if shard_risks.shape != (n, 4, 2) or not np.isfinite(shard_risks).all():
                errors.append(f"descriptor shard risk shape/finite check failed: {scenario}")
            if shard_features.shape != (n, 18) or not np.isfinite(shard_features).all():
                errors.append(f"descriptor shard feature shape/finite check failed: {scenario}")
        except (KeyError, ValueError, OSError) as error:
            errors.append(f"descriptor shard validation failed: {scenario}: {error}")

    indexed: dict[
        tuple[str, str], tuple[Path, dict[str, Any], dict[str, Any], str]
    ] = {}
    supplemental_identity: dict[str, tuple[Path, dict[str, Any]]] = {}
    malformed: list[str] = []
    for path in sorted(matrix_dir.glob("*.json")):
        if path == matrix_protocol_path:
            continue
        try:
            payload = read_json(path)
        except (ValueError, json.JSONDecodeError, OSError):
            malformed.append(path.name)
            continue
        protocol = payload.get("protocol")
        records = payload.get("records")
        if not isinstance(protocol, dict) or not isinstance(records, list):
            continue
        key = (str(protocol.get("expert")), str(protocol.get("scenario")))
        if key in indexed:
            errors.append(f"duplicate matrix result: {key}")
        indexed[key] = (path, payload, matrix, "primary")
    if malformed:
        errors.append(f"malformed/in-progress matrix JSON files: {malformed}")

    supplemental_sources_checked = 0
    supplemental_protocol_hashes: dict[str, str] = {}
    for supplemental_name, directory_value in supplemental_specs.items():
        supplemental_dir = resolve(base, directory_value)
        supplemental_protocol_path = supplemental_dir / "protocol.json"
        if not supplemental_protocol_path.is_file():
            errors.append(f"supplemental matrix protocol missing: {supplemental_name}")
            continue
        supplemental = read_json(supplemental_protocol_path)
        file_hash = sha256(supplemental_protocol_path)
        supplemental_protocol_hashes[supplemental_name] = file_hash
        expected_key = f"supplemental_matrix_protocol.{supplemental_name}"
        if expected.get(expected_key) != file_hash:
            errors.append(f"frozen source hash mismatch: {expected_key}")
        for field in (
            "backend", "seed", "subset_seed", "corruption_seed", "ids_source_sha256",
            "ids_sha256", "n", "offset", "cache_protocol_sha256", "cache_file_sha256",
            "scenarios", "batch_size", "clip_checkpoint_sha256", "meta_sha256",
        ):
            if supplemental.get(field) != matrix.get(field):
                errors.append(f"supplemental/main protocol mismatch: {supplemental_name}.{field}")
        supplemental_expert_names = [
            str(value["name"]) for value in supplemental.get("experts", [])
        ]
        if supplemental_expert_names != [identity_name, supplemental_name]:
            errors.append(
                f"supplemental matrix must contain only identity and {supplemental_name}"
            )
        supplemental_sources_checked += _source_checks(
            supplemental.get("source_sha256", {}), errors,
            f"supplemental[{supplemental_name}]",
        )
        source_manifest = Path(supplemental.get("manifest_source", ""))
        if (
            not source_manifest.is_file()
            or sha256(source_manifest) != supplemental.get("manifest_sha256")
        ):
            errors.append(
                f"supplemental runner manifest missing/hash mismatch: {supplemental_name}"
            )
        for path in sorted(supplemental_dir.glob("*.json")):
            if path == supplemental_protocol_path:
                continue
            try:
                payload = read_json(path)
            except (ValueError, json.JSONDecodeError, OSError):
                malformed.append(str(path))
                continue
            protocol = payload.get("protocol")
            records = payload.get("records")
            if not isinstance(protocol, dict) or not isinstance(records, list):
                continue
            expert = str(protocol.get("expert"))
            scenario = str(protocol.get("scenario"))
            if expert == identity_name:
                if scenario in supplemental_identity:
                    errors.append(f"duplicate supplemental identity: {supplemental_name}|{scenario}")
                supplemental_identity[scenario] = (path, payload)
                continue
            key = (expert, scenario)
            if key in indexed:
                errors.append(f"duplicate combined matrix result: {key}")
            indexed[key] = (path, payload, supplemental, f"supplemental:{supplemental_name}")
    if malformed:
        errors.append(f"malformed/in-progress supplemental JSON files: {malformed}")

    expected_pairs = [(expert, scenario) for expert in matrix_experts for scenario in scenarios]
    missing = [f"{expert}|{scenario}" for expert, scenario in expected_pairs if (expert, scenario) not in indexed]
    extras = [
        f"{expert}|{scenario}" for expert, scenario in indexed
        if (expert, scenario) not in set(expected_pairs)
    ]
    if extras:
        errors.append(f"unexpected matrix entries: {extras}")

    result_hashes: dict[str, str] = {}
    pair_checks = 0
    for expert, scenario in expected_pairs:
        value = indexed.get((expert, scenario))
        if value is None:
            continue
        path, payload, source_matrix, source_kind = value
        result_hashes[str(path.resolve())] = sha256(path)
        protocol = payload["protocol"]
        records = payload["records"]
        expected_result_protocol_hash = canonical_sha256(source_matrix)
        if protocol.get("protocol_sha256") != expected_result_protocol_hash:
            errors.append(f"result protocol hash mismatch: {expert}|{scenario}")
        if protocol.get("backend") != source_matrix.get("backend"):
            errors.append(f"result backend mismatch: {expert}|{scenario}")
        if protocol.get("seed") != 1109:
            errors.append(f"result seed mismatch: {expert}|{scenario}")
        if protocol.get("cache_file_sha256") != source_matrix.get("cache_file_sha256", {}).get(scenario):
            errors.append(f"result cache hash mismatch: {expert}|{scenario}")
        if len(records) != n:
            errors.append(f"result row count mismatch: {expert}|{scenario}")
            continue
        by_sample: dict[int, dict[str, Any]] = {}
        for position, row in enumerate(records):
            sample = int(row.get("sample", -1))
            if sample in by_sample:
                errors.append(f"duplicate result sample: {expert}|{scenario}|{sample}")
                continue
            by_sample[sample] = row
            if row.get("scenario") != scenario:
                errors.append(f"record scenario mismatch: {expert}|{scenario}|{sample}")
            if row.get("stream_position") != position:
                errors.append(f"record stream order mismatch: {expert}|{scenario}|{sample}")
            expected_raw = descriptor_maps.get(scenario, {}).get(sample)
            if row.get("raw_sha256") != expected_raw:
                errors.append(f"descriptor/result raw hash mismatch: {expert}|{scenario}|{sample}")
        if set(by_sample) != set(range(n)):
            errors.append(f"result sample IDs mismatch: {expert}|{scenario}")
        reconstructed = 100.0 * sum(bool(row.get("correct")) for row in records) / n
        if not math.isclose(float(payload.get("accuracy", math.nan)), reconstructed, abs_tol=1e-10):
            errors.append(f"result accuracy is not reconstructible: {expert}|{scenario}")
        identity = indexed.get((identity_name, scenario))
        if expert != identity_name and identity is not None:
            identity_records = identity[1]["records"]
            left = [
                (row.get("sample"), row.get("target"), row.get("batch"),
                 row.get("stream_position"), row.get("raw_sha256"))
                for row in identity_records
            ]
            right = [
                (row.get("sample"), row.get("target"), row.get("batch"),
                 row.get("stream_position"), row.get("raw_sha256"))
                for row in records
            ]
            if left != right:
                errors.append(f"expert is not exactly paired to identity: {expert}|{scenario}")
            else:
                pair_checks += 1

    supplemental_identity_checks = 0
    for scenario, (path, payload) in supplemental_identity.items():
        primary_identity = indexed.get((identity_name, scenario))
        if primary_identity is None:
            errors.append(f"supplemental identity lacks primary identity: {scenario}")
            continue
        protocol = payload["protocol"]
        matching_name = next(
            (
                name for name, directory in supplemental_specs.items()
                if resolve(base, directory) == path.parent.resolve()
            ),
            None,
        )
        if matching_name is None:
            errors.append(f"cannot identify supplemental identity source: {path}")
            continue
        supplemental_protocol_path = resolve(base, supplemental_specs[matching_name]) / "protocol.json"
        supplemental = read_json(supplemental_protocol_path)
        if protocol.get("protocol_sha256") != canonical_sha256(supplemental):
            errors.append(f"supplemental identity protocol mismatch: {scenario}")
        left = [
            (row.get("sample"), row.get("target"), row.get("prediction"), row.get("correct"),
             row.get("batch"), row.get("stream_position"), row.get("raw_sha256"))
            for row in primary_identity[1]["records"]
        ]
        right = [
            (row.get("sample"), row.get("target"), row.get("prediction"), row.get("correct"),
             row.get("batch"), row.get("stream_position"), row.get("raw_sha256"))
            for row in payload["records"]
        ]
        if left != right:
            errors.append(f"supplemental identity is not identical to primary identity: {scenario}")
        else:
            supplemental_identity_checks += 1
    if supplemental_specs and set(supplemental_identity) != set(scenarios):
        errors.append("supplemental identity does not cover every scenario exactly once")

    ready = not errors and not missing and len(indexed) == len(expected_pairs)
    status = "ready_complete_matrix" if ready else "blocked_incomplete_or_misaligned_matrix"
    report = {
        "schema_version": 1,
        "status": status,
        "ready": ready,
        "label_or_model_generated": False,
        "seed": 1109,
        "scenarios": scenarios,
        "experts_including_identity": matrix_experts,
        "expected_matrix_entries": len(expected_pairs),
        "observed_matrix_entries": len(indexed),
        "missing_matrix_entry_count": len(missing),
        "missing_matrix_entries": missing,
        "errors": errors,
        "available_result_files_deep_checked": len(indexed),
        "available_identity_expert_pairs_checked": pair_checks,
        "source_files_checked": {
            "matrix": matrix_sources_checked,
            "descriptor": descriptor_sources_checked,
            "supplemental": supplemental_sources_checked,
        },
        "supplemental_identity_rows_deduplicated": len(supplemental_identity),
        "supplemental_identity_cross_checks": supplemental_identity_checks,
        "supplemental_protocol_sha256": supplemental_protocol_hashes,
        "hashes": actual_hashes,
        "matrix_claims_complete_cross_product": bool(matrix.get("complete_cross_product")),
        "actual_cross_product_complete": not missing and len(indexed) == len(expected_pairs),
    }
    if not ready:
        return report, None
    expanded = {
        "seed": 1109,
        "experts": experts,
        "scenarios": [
            {
                "name": scenario,
                "risk_npz": str(descriptor_shards[scenario]),
                # Shards are namespaced by the frozen descriptor protocol rather
                # than duplicating the image-ID hash in every NPZ.  The builder
                # may inherit this ID hash only after validate_label_sources has
                # proved that matrix and descriptor protocols agree exactly.
                "ids_sha256": str(descriptor["ids_sha256"]),
                "risk_source_protocol_sha256": actual_hashes["descriptor_protocol"],
                "identity_result": str(indexed[(identity_name, scenario)][0]),
                "expert_results": {
                    expert: str(indexed[(expert, scenario)][0]) for expert in experts
                },
            }
            for scenario in scenarios
        ],
    }
    return report, expanded


def validate_router_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    base = manifest_path.parent
    errors = []
    if manifest.get("schema_version") != 1:
        errors.append("router manifest schema_version must be 1")
    if manifest.get("status") != "development_utility_router_training":
        errors.append("router manifest status mismatch")
    if manifest.get("seed") != 1109:
        errors.append("router training requires seed 1109")
    label_manifest = resolve(base, manifest["utility_label_manifest"])
    if not label_manifest.is_file() or sha256(label_manifest) != manifest.get("utility_label_manifest_sha256"):
        errors.append("utility-label manifest is missing or hash-mismatched")
    dataset = resolve(base, manifest["dataset"])
    dataset_report = resolve(base, manifest["dataset_report"])
    if not dataset.is_file() or not dataset_report.is_file():
        return {
            "schema_version": 1,
            "status": "blocked_waiting_for_utility_labels",
            "ready": False,
            "label_or_model_generated": False,
            "errors": errors,
            "missing": [str(path) for path in (dataset, dataset_report) if not path.is_file()],
        }
    report = read_json(dataset_report)
    if report.get("status") != "development_utility_targets_not_confirmation_evidence":
        errors.append("utility dataset report status mismatch")
    if report.get("seed") != 1109:
        errors.append("utility dataset report seed mismatch")
    if report.get("dataset_sha256") != sha256(dataset):
        errors.append("utility dataset hash mismatch")
    if report.get("source_manifest_sha256") != sha256(label_manifest):
        errors.append("utility dataset was not built from the declared label manifest")
    if not report.get("complete_expert_matrix_required"):
        errors.append("utility dataset was built with incomplete experts allowed")
    expected_sources = manifest.get("source_sha256", {})
    for relative, expected in expected_sources.items():
        path = resolve(base, relative)
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"router source file missing/hash mismatch: {relative}")
    return {
        "schema_version": 1,
        "status": "ready_for_router_training" if not errors else "blocked_router_input_mismatch",
        "ready": not errors,
        "label_or_model_generated": False,
        "errors": errors,
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "dataset_report": str(dataset_report),
        "dataset_report_sha256": sha256(dataset_report),
    }
