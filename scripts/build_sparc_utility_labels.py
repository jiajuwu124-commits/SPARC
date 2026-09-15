#!/usr/bin/env python3
"""Build direct expert-utility targets from matched development runs.

The manifest identifies descriptor files and matched identity/expert records.
Corruption names are retained only as audit metadata; they are never included
in the model input or target.  By default every expert must have a matched run
for every scenario, which prevents a sparse evidence matrix from silently
turning scenario identity into expert availability.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_router_module = load_local_module(
    "sparc_utility_router_local", ROOT / "src/streammint/utility_router.py"
)
_contract_module = load_local_module(
    "sparc_utility_contract_local", ROOT / "src/streammint/utility_pipeline_contract.py"
)
RISK_FEATURE_ORDER = _router_module.RISK_FEATURE_ORDER
risk_features = _router_module.risk_features
validate_label_sources = _contract_module.validate_label_sources


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_records(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    payload = json.loads(path.read_text())
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"missing records: {path}")
    indexed: dict[int, dict[str, Any]] = {}
    for row in records:
        sample = int(row["sample"])
        if sample in indexed:
            raise ValueError(f"duplicate sample {sample}: {path}")
        indexed[sample] = row
    return payload, indexed


def scenario_from_protocol(payload: dict[str, Any]) -> str | None:
    protocol = payload.get("protocol", {})
    return protocol.get("scenario") or (payload.get("records") or [{}])[0].get("domain") \
        or (payload.get("records") or [{}])[0].get("scenario")


def seed_from_protocol(payload: dict[str, Any]) -> int | None:
    protocol = payload.get("protocol", {})
    value = protocol.get("order_seed", protocol.get("seed"))
    return None if value is None else int(value)


def validate_pair(
    scenario: str,
    identity: dict[int, dict[str, Any]],
    expert: dict[int, dict[str, Any]],
    risk_hash_by_sample: dict[int, str],
    identity_path: Path,
    expert_path: Path,
) -> None:
    if set(identity) != set(expert) or set(identity) != set(risk_hash_by_sample):
        raise ValueError(f"sample-set mismatch for {scenario}: {identity_path} vs {expert_path}")
    for sample, base in identity.items():
        candidate = expert[sample]
        for field in ("target", "batch", "stream_position", "raw_sha256"):
            if candidate.get(field) != base.get(field):
                raise ValueError(f"{field} mismatch for {scenario}, sample={sample}")
        raw_hash = str(base.get("raw_sha256"))
        if raw_hash != risk_hash_by_sample[sample]:
            raise ValueError(
                f"descriptor/result raw hash mismatch for {scenario}, sample={sample}; "
                "regenerate descriptors from the exact frozen corruption cache"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--granularity", choices=("sample", "batch"))
    parser.add_argument("--allow-missing-experts", action="store_true")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="deep-audit matrix/descriptor readiness without writing labels",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    source_validation = None
    schema_v2 = manifest.get("schema_version") == 2
    output_from_manifest = None
    report_from_manifest = None
    if schema_v2:
        if args.allow_missing_experts:
            raise ValueError("schema-v2 actual manifests forbid --allow-missing-experts")
        source_validation, expanded = validate_label_sources(manifest_path)
        if args.validate_only:
            print(json.dumps(source_validation, indent=2))
            if not source_validation["ready"]:
                raise SystemExit(2)
            return
        if not source_validation["ready"] or expanded is None:
            raise RuntimeError(
                "utility-label build blocked: expert matrix is incomplete or misaligned; "
                "run --validate-only for the score-free readiness report"
            )
        output_from_manifest = manifest.get("output")
        report_from_manifest = manifest.get("report")
        granularity_from_manifest = manifest.get("granularity", "batch")
        manifest = expanded
        args.granularity = args.granularity or granularity_from_manifest
    elif args.validate_only:
        raise ValueError("--validate-only requires a schema-v2 source manifest")
    args.granularity = args.granularity or "batch"
    if int(manifest.get("seed", 1109)) != 1109:
        raise ValueError("SPARC utility-label protocol requires seed 1109")
    experts = tuple(manifest["experts"])
    if not experts or len(set(experts)) != len(experts) or "identity" in experts:
        raise ValueError("experts must be unique and must not include identity")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("manifest.scenarios must be a non-empty list")

    base = manifest_path.parent
    all_features: list[np.ndarray] = []
    all_utilities: list[np.ndarray] = []
    all_observed: list[np.ndarray] = []
    all_groups: list[str] = []
    all_batches: list[str] = []
    all_samples: list[int] = []
    all_scenarios: list[str] = []
    all_raw_hashes: list[str] = []
    source_hashes: dict[str, str] = {}

    for entry in scenarios:
        name = str(entry["name"])
        risk_path = resolve(base, entry["risk_npz"])
        identity_path = resolve(base, entry["identity_result"])
        source_hashes[str(risk_path)] = sha256(risk_path)
        source_hashes[str(identity_path)] = sha256(identity_path)
        with np.load(risk_path, allow_pickle=False) as stored:
            risks = stored["risks"]
            samples = stored["samples"].astype(np.int64)
            raw_hashes = [value.decode() if isinstance(value, bytes) else str(value)
                          for value in stored["raw_sha256"]]
            if "ids_sha256" in stored.files:
                ids_sha = str(stored["ids_sha256"].item())
                declared_ids_sha = entry.get("ids_sha256")
                if declared_ids_sha is not None and ids_sha != declared_ids_sha:
                    raise ValueError(f"descriptor shard ID namespace mismatch: {risk_path}")
            elif schema_v2:
                # Production descriptor shards are bound to the audited
                # descriptor protocol instead of repeating ids_sha256.  The
                # schema-v2 contract supplies the ID namespace only after it
                # verifies matrix/descriptor ID equality and the shard hash.
                ids_sha = str(entry["ids_sha256"])
                shard_protocol = str(stored["source_protocol_sha256"].item())
                if shard_protocol != entry["risk_source_protocol_sha256"]:
                    raise ValueError(
                        f"descriptor shard source protocol mismatch: {risk_path}"
                    )
            else:
                raise ValueError(
                    f"legacy descriptor shard lacks ids_sha256: {risk_path}"
                )
        if len(set(samples.tolist())) != len(samples):
            raise ValueError(f"duplicate descriptor sample IDs: {risk_path}")
        descriptors = risk_features(risks)
        risk_hash_by_sample = dict(zip(samples.tolist(), raw_hashes))

        identity_payload, identity = load_records(identity_path)
        if scenario_from_protocol(identity_payload) != name:
            raise ValueError(f"identity scenario mismatch: expected {name}")
        if set(identity) != set(samples.tolist()):
            raise ValueError(f"descriptor/identity sample-set mismatch: {name}")
        ordered_identity = [identity[int(sample)] for sample in samples]
        if seed_from_protocol(identity_payload) != 1109:
            raise ValueError(f"identity result does not use seed 1109: {identity_path}")

        utilities = np.full((len(samples), len(experts)), np.nan, dtype=np.float64)
        observed = np.zeros_like(utilities, dtype=bool)
        expert_results = entry.get("expert_results", {})
        for expert_index, expert_name in enumerate(experts):
            value = expert_results.get(expert_name)
            if value is None:
                if not args.allow_missing_experts:
                    raise ValueError(f"missing expert {expert_name!r} for scenario {name!r}")
                continue
            expert_path = resolve(base, value)
            source_hashes[str(expert_path)] = sha256(expert_path)
            expert_payload, expert = load_records(expert_path)
            if scenario_from_protocol(expert_payload) != name:
                raise ValueError(f"expert scenario mismatch: expected {name}, file={expert_path}")
            if seed_from_protocol(expert_payload) != 1109:
                raise ValueError(f"expert result does not use seed 1109: {expert_path}")
            validate_pair(name, identity, expert, risk_hash_by_sample, identity_path, expert_path)
            for row_index, sample in enumerate(samples):
                base_correct = float(bool(identity[int(sample)]["correct"]))
                expert_correct = float(bool(expert[int(sample)]["correct"]))
                utilities[row_index, expert_index] = expert_correct - base_correct
                observed[row_index, expert_index] = True

        if args.granularity == "batch":
            batch_values = np.asarray([int(row["batch"]) for row in ordered_identity])
            for batch in np.unique(batch_values):
                selected = batch_values == batch
                for expert_index in range(len(experts)):
                    available = selected & observed[:, expert_index]
                    if available.any():
                        utilities[selected, expert_index] = utilities[available, expert_index].mean()

        all_features.append(descriptors)
        all_utilities.append(utilities)
        all_observed.append(observed)
        all_groups.extend(f"{ids_sha}:{int(sample)}" for sample in samples)
        all_batches.extend(f"{name}:batch{int(row['batch'])}" for row in ordered_identity)
        all_samples.extend(samples.tolist())
        all_scenarios.extend([name] * len(samples))
        all_raw_hashes.extend(raw_hashes)

    output_value = args.output or (
        resolve(manifest_path.parent, output_from_manifest) if output_from_manifest else None
    )
    if output_value is None:
        raise ValueError("--output or manifest.output is required")
    output = Path(output_value).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.concatenate(all_features),
        utilities=np.concatenate(all_utilities),
        observed=np.concatenate(all_observed),
        groups=np.asarray(all_groups),
        batch_ids=np.asarray(all_batches),
        samples=np.asarray(all_samples, dtype=np.int64),
        scenarios=np.asarray(all_scenarios),
        raw_sha256=np.asarray(all_raw_hashes),
        expert_names=np.asarray(experts),
        feature_order=np.asarray(RISK_FEATURE_ORDER),
        seed=np.asarray(1109),
        granularity=np.asarray(args.granularity),
    )
    report_path = (
        args.report.resolve() if args.report else
        resolve(manifest_path.parent, report_from_manifest) if report_from_manifest else
        output.with_suffix(".manifest.json")
    )
    report = {
        "schema_version": 1,
        "status": "development_utility_targets_not_confirmation_evidence",
        "seed": 1109,
        "observations": int(sum(len(value) for value in all_features)),
        "unique_base_image_groups": len(set(all_groups)),
        "scenarios": [str(entry["name"]) for entry in scenarios],
        "experts": list(experts),
        "granularity": args.granularity,
        "input_to_router": "18D masked-risk descriptor plus optional inference-time batch statistics",
        "target": "matched expert correctness minus identity correctness",
        "corruption_name_used_as_model_input_or_target": False,
        "image_class_label_used_as_model_input": False,
        "development_correctness_used_to_construct_utility_target": True,
        "complete_expert_matrix_required": not args.allow_missing_experts,
        "dataset": str(output),
        "dataset_sha256": sha256(output),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "source_sha256": source_hashes,
        "source_readiness_audit": source_validation,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
