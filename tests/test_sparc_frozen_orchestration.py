from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sparc_frozen_orchestration_tested",
    ROOT / "scripts/run_sparc_frozen_orchestration.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    else:
        path.write_text(str(value))
    return path


def art(root: Path, path: Path):
    return {"path": str(path.relative_to(root)), "sha256": MODULE.sha256(path)}


def make_fixture(root: Path):
    configs = root / "configs"
    runner = write(root / "scripts/unified.py", "# frozen unified runner\n")
    prompt = write(configs / "prompt.txt", "a photo of a {}\n")
    model = write(root / "models/vitl.pt", "weights")
    environment = write(configs / "environment.json", {
        "status": "frozen_execution_environment", "values": {},
    })
    parent_router = write(root / "models/identity.joblib", "identity")
    plugin_router = write(root / "models/sparc.joblib", "sparc")
    capability = write(configs / "capability.json", {
        "status": "validated_unified_nonlinear_routed_runner",
        "runner_sha256": MODULE.sha256(runner),
        "supported_backends": list(MODULE.BACKENDS),
        "supports_nonlinear_router": True,
        "same_runner_for_parent_and_plugin": True,
    })
    control = write(configs / "control.json", {
        "status": "frozen_identity_control_router_all_abstain",
        "artifact_sha256": MODULE.sha256(parent_router),
    })
    dev = write(configs / "dev_ids.txt", "dev0\ndev1\n")
    test_ids = {"test": ["t0", "t1"]}
    test_paths = {
        split: write(configs / f"test_{split}.txt", "\n".join(ids) + "\n")
        for split, ids in test_ids.items()
    }
    descriptor = write(root / "cache/features.npz", "features")
    conditions = []
    manifest_paths = {}
    model_hash = MODULE.sha256(model)
    prompt_hash = MODULE.sha256(prompt)
    sequence_hashes = {split: MODULE.sequence_sha256(ids) for split, ids in test_ids.items()}
    reset = {backend: "not_applicable" if backend == "clip" else
             "once_before_this_corruption_stream" for backend in MODULE.BACKENDS}
    for split in MODULE.SPLITS:
        for backend in MODULE.BACKENDS:
            for scenario in MODULE.SCENARIOS:
                slug = scenario.replace(":", "_s")
                index = write(root / f"indexes/{split}_{slug}.json", {
                    "split": split, "scenario": scenario, "rows": 2,
                    "records": [
                        {"image_id": image_id, "sample": position,
                         "target": position, "audit_domain": scenario,
                         "raw_sha256": f"{position + 1:064x}"}
                        for position, image_id in enumerate(test_ids[split])
                    ],
                })
                common = {
                    "schema_version": 1, "status": "frozen_for_routed_replay",
                    "seed": 1109, "batch_size": 20, "memory_fraction": 0.68,
                    "backend": {"kind": backend, "clip_checkpoint": art(root, model)},
                    "features": {"kind": "precomputed_descriptors", "artifact": art(root, descriptor)},
                    "expert_bank": [{"name": "x", "kind": "fixed", "operator": "identity"}],
                    "input_stream": {"shuffle": True, "index": art(root, index)},
                    "routing_information_contract": {"mint_reset": reset[backend]},
                    "evaluation_contract": {
                        "split": split, "scenario": scenario,
                        "test_ids_sequence_sha256": sequence_hashes[split],
                        "prompt_sha256": prompt_hash, "model_sha256": model_hash,
                    },
                }
                parent = {**common, "evidence_scope": "parent", "router": {"artifact": art(root, parent_router)}}
                plugin = {**common, "evidence_scope": "plugin", "router": {"artifact": art(root, plugin_router)}}
                parent_path = write(configs / f"{split}_{backend}_{slug}_parent.json", parent)
                plugin_path = write(configs / f"{split}_{backend}_{slug}_plugin.json", plugin)
                manifest_paths[(split, backend, scenario)] = (parent_path, plugin_path)
                conditions.append({
                    "split": split, "backend": backend, "scenario": scenario,
                    "parent": {"manifest": art(root, parent_path),
                               "output_dir": f"outputs/{split}/{backend}/{slug}/parent"},
                    "plugin": {"manifest": art(root, plugin_path),
                               "output_dir": f"outputs/{split}/{backend}/{slug}/plugin"},
                })
    manifest = {
        "schema_version": 1, "status": "frozen_before_evaluation",
        "evidence_scope": "new_untouched_confirmation_not_used_for_selection",
        "runner": art(root, runner), "runner_capability_report": art(root, capability),
        "prompt_contract": art(root, prompt), "model_checkpoint": art(root, model),
        "environment_lock": art(root, environment),
        "identity_control_report": art(root, control),
        "development_id_files": [art(root, dev)],
        "test_id_files": {split: art(root, path) for split, path in test_paths.items()},
        "protocol": {
            "seed": 1109, "batch_size": 20, "n_per_scenario": 2,
            "scenarios": list(MODULE.SCENARIOS), "backends": list(MODULE.BACKENDS),
            "reset_policy": reset, "prompt_sha256": prompt_hash,
            "model_sha256": model_hash, "test_ids_sequence_sha256": sequence_hashes,
        },
        "statistics": {"bootstrap_replicates": 20, "bootstrap_seed": 99173},
        "acceptance": {
            "core_scope": "all_75_backend_corruption_cells",
            "strict_minimum_gain_pp": 0.0,
            "comparison": "strictly_greater_than",
            "preferred_gain_pp": 1.5,
        },
        "execution": {"python": sys.executable, "command_log": "outputs/commands.jsonl"},
        "outputs": {"summary_dir": "outputs/summary"}, "conditions": conditions,
    }
    manifest_path = write(configs / "orchestration.json", manifest)
    return manifest_path, manifest, manifest_paths, test_paths


def rewrite_run_pair(root, manifest, paths, key, mutate):
    parent_path, plugin_path = paths[key]
    for path in (parent_path, plugin_path):
        value = json.loads(path.read_text())
        mutate(value)
        write(path, value)
    condition = next(value for value in manifest["conditions"] if MODULE.condition_key(value) == key)
    condition["parent"]["manifest"] = art(root, parent_path)
    condition["plugin"]["manifest"] = art(root, plugin_path)


def test_complete_frozen_preflight_covers_75_cells():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, _, _, _ = make_fixture(root)
        report, _ = MODULE.preflight(path, native_validate=False)
        assert report["ready"] is True
        assert report["condition_count"] == 75
        assert report["development_test_overlap"] == 0


@pytest.mark.parametrize("field", ["seed", "batch", "model", "prompt", "reset"])
def test_fairness_gate_rejects_asymmetric_or_wrong_contract(field):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, manifest, paths, _ = make_fixture(root)
        key = ("test", "tent", "defocus_blur:5")
        def mutate(value):
            if field == "seed": value["seed"] = 7
            elif field == "batch": value["batch_size"] = 4
            elif field == "model": value["backend"]["clip_checkpoint"]["sha256"] = "0" * 64
            elif field == "prompt": value["evaluation_contract"]["prompt_sha256"] = "0" * 64
            else: value["routing_information_contract"]["mint_reset"] = "never"
        rewrite_run_pair(root, manifest, paths, key, mutate)
        write(path, manifest)
        with pytest.raises(MODULE.GateError):
            MODULE.preflight(path, native_validate=False)


def test_development_test_overlap_is_rejected_before_execution():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, manifest, _, test_paths = make_fixture(root)
        write(test_paths["test"], "dev0\nt1\n")
        manifest["test_id_files"]["test"] = art(root, test_paths["test"])
        write(path, manifest)
        with pytest.raises(MODULE.GateError, match="development/test ID overlap"):
            MODULE.preflight(path, native_validate=False)


def make_output(path: Path, *, runner_sha: str, side: str, changed_sample: bool = False,
                bad_batch: bool = False):
    manifest_sha = "1" * 64 if side == "parent" else "2" * 64
    router_sha = "3" * 64 if side == "parent" else "4" * 64
    protocol = {
        "runner_sha256": runner_sha, "seed": 1109, "batch_size": 20,
        "backend": "clip", "manifest_sha256": manifest_sha,
        "router_sha256": router_sha, "model_sha256": "5" * 64,
        "prompt_sha256": "6" * 64, "test_ids_sequence_sha256": "7" * 64,
        "reset_policy": "not_applicable",
    }
    write(path / "protocol.json", protocol)
    protocol_sha = MODULE.sha256(path / "protocol.json")
    rows = []
    for position in range(2):
        sample = 9 if changed_sample and position == 0 else position
        rows.append({
            "image_id": f"t{position}", "sample": sample, "target": position,
            "raw_sha256": f"{position + 1:064x}", "stream_position": position,
            "batch": 1 if bad_batch and position == 0 else 0,
            "audit_domain": "defocus_blur:5", "correct": bool(position),
            "expert": "identity" if side == "parent" else "gaussian15",
            "decision": "abstain" if side == "parent" else "route",
        })
    result = {
        "protocol": protocol, "protocol_sha256": protocol_sha, "records": rows,
        "records_sha256": MODULE.canonical_sha256(rows), "accuracy": 50.0,
    }
    write(path / "result.json", result)
    write(path / "complete.json", {"result_sha256": MODULE.sha256(path / "result.json")})


def test_output_gate_rejects_mismatched_sample_ids_and_order():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        runner = write(root / "runner.py", "# runner")
        parent_path, plugin_path = root / "parent", root / "plugin"
        make_output(parent_path, runner_sha=MODULE.sha256(runner), side="parent")
        make_output(
            plugin_path, runner_sha=MODULE.sha256(runner), side="plugin",
            changed_sample=True,
        )
        report = {"runner_sha256": MODULE.sha256(runner), "conditions": [{
            "split": "test", "backend": "clip", "scenario": "defocus_blur:5",
            "parent_output": str(parent_path), "plugin_output": str(plugin_path),
            "parent_manifest_sha256": "1" * 64, "plugin_manifest_sha256": "2" * 64,
            "parent_router_sha256": "3" * 64, "plugin_router_sha256": "4" * 64,
        }]}
        protocol = {
            "seed": 1109, "batch_size": 20, "n_per_scenario": 2,
            "model_sha256": "5" * 64, "prompt_sha256": "6" * 64,
            "test_ids_sequence_sha256": {"test": "7" * 64},
            "reset_policy": {"clip": "not_applicable"},
        }
        with pytest.raises(MODULE.GateError, match="not exactly paired"):
            MODULE.audit_outputs(report, protocol)


def test_output_gate_rejects_wrong_batch_even_if_other_fields_match():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        runner = write(root / "runner.py", "# runner")
        output = root / "parent"
        make_output(
            output, runner_sha=MODULE.sha256(runner), side="parent", bad_batch=True
        )
        expected = {
            "runner_sha256": MODULE.sha256(runner), "seed": 1109,
            "batch_size": 20, "backend": "clip", "scenario": "defocus_blur:5",
            "n": 2, "manifest_sha256": "1" * 64, "router_sha256": "3" * 64,
            "model_sha256": "5" * 64, "prompt_sha256": "6" * 64,
            "test_ids_sequence_sha256": "7" * 64, "reset_policy": "not_applicable",
        }
        with pytest.raises(MODULE.GateError, match="order/batch mismatch"):
            MODULE.output_records(output, expected, "parent")
