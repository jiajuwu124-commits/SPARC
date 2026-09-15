#!/usr/bin/env python3
"""Strict utility-label builder for multiple supplemental expert matrices.

The original validator deliberately accepts one supplemental identity stream.
This extension validates each supplemental matrix independently against the
same primary identity, then merges only the already validated expert paths.
No score, record, pairing, descriptor, seed, or source-hash check is relaxed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_sparc_utility_labels.py"
CONTRACT = ROOT / "src/streammint/utility_pipeline_contract.py"


def load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


contract = load("sparc_multi_supplement_contract", CONTRACT)
builder = load("sparc_multi_supplement_builder", BUILDER)


def absolute(base: Path, value: str) -> str:
    return str(contract.resolve(base, value))


def validate_multiple(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    manifest_path = manifest_path.resolve()
    manifest = contract.read_json(manifest_path)
    supplements = manifest.get("supplemental_experts", {})
    if not isinstance(supplements, dict) or len(supplements) <= 1:
        return contract.validate_label_sources(manifest_path)
    base = manifest_path.parent
    supplement_names = list(supplements)
    primary_experts = [
        name for name in manifest.get("experts", []) if name not in supplement_names
    ]
    reports: dict[str, Any] = {}
    expanded_parts: dict[str, dict[str, Any]] = {}
    for name in supplement_names:
        expected = {
            key: value for key, value in manifest.get("expected_sha256", {}).items()
            if not key.startswith("supplemental_matrix_protocol.")
        }
        expected[f"supplemental_matrix_protocol.{name}"] = manifest[
            "expected_sha256"
        ][f"supplemental_matrix_protocol.{name}"]
        single = {
            **manifest,
            "matrix_dir": absolute(base, manifest["matrix_dir"]),
            "descriptor_cache": absolute(base, manifest["descriptor_cache"]),
            "experts": [*primary_experts, name],
            "supplemental_experts": {name: absolute(base, supplements[name])},
            "expected_sha256": expected,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=manifest_path.parent, delete=False
        ) as temporary:
            temporary.write(json.dumps(single, indent=2) + "\n")
            temporary_path = Path(temporary.name)
        try:
            report, expanded = contract.validate_label_sources(temporary_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        reports[name] = report
        if not report.get("ready") or expanded is None:
            return {
                "schema_version": 1,
                "status": "blocked_multi_supplement_validation",
                "ready": False,
                "label_or_model_generated": False,
                "seed": 1109,
                "errors": [f"supplemental validation failed: {name}"],
                "per_supplement": reports,
            }, None
        expanded_parts[name] = expanded

    first = expanded_parts[supplement_names[0]]
    merged_scenarios = []
    for scenario_index, reference in enumerate(first["scenarios"]):
        expert_results = {
            key: value for key, value in reference["expert_results"].items()
            if key in primary_experts
        }
        for name in supplement_names:
            candidate = expanded_parts[name]["scenarios"][scenario_index]
            for field in (
                "name", "risk_npz", "ids_sha256", "risk_source_protocol_sha256",
                "identity_result",
            ):
                if candidate[field] != reference[field]:
                    raise RuntimeError(f"supplemental expansion mismatch: {name}.{field}")
            expert_results[name] = candidate["expert_results"][name]
        merged_scenarios.append({**reference, "expert_results": {
            name: expert_results[name] for name in manifest["experts"]
        }})
    expanded = {
        "seed": 1109,
        "experts": list(manifest["experts"]),
        "scenarios": merged_scenarios,
    }
    report = {
        "schema_version": 1,
        "status": "ready_complete_multi_supplement_matrix",
        "ready": True,
        "label_or_model_generated": False,
        "seed": 1109,
        "experts_including_identity": [manifest.get("identity_expert", "identity"), *manifest["experts"]],
        "supplemental_experts": supplement_names,
        "per_supplement": reports,
        "identity_deduplication_rule": (
            "each supplemental identity was independently required to match the same primary identity"
        ),
        "checks_relaxed": [],
    }
    return report, expanded


def main() -> None:
    builder.validate_label_sources = validate_multiple
    builder.main()


if __name__ == "__main__":
    main()
