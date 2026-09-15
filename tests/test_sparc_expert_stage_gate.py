from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_sparc_expert_stage_gate",
    ROOT / "scripts/run_sparc_expert_stage_gate.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


FAKE_EVALUATOR = r'''#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

p = argparse.ArgumentParser()
p.add_argument("--ids", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--status", required=True)
p.add_argument("--backends", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--subset-seed", type=int, required=True)
p.add_argument("--batch-size", type=int, required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--mode", default="normal")
p.add_argument("--scales")
p.add_argument("--frozen", type=Path)
a = p.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=False)
backends = a.backends.split(",")
stage = a.ids.stem
if a.status == "development":
    scales = [float(x) for x in a.scales.split(",")]
else:
    scales = [float(json.loads(a.frozen.read_text())["selected_scale"])]
protocol = {
    "status": a.status, "seed": a.seed, "batch_size": a.batch_size,
    "backends": backends,
    "cache_protocol_sha256": sha(a.cache / "protocol.json"),
    "ids_source": str(a.ids.resolve()),
    "ids_sha256": hashlib.sha256("\n".join(
        line for line in a.ids.read_text().splitlines() if line
    ).encode()).hexdigest(),
    "n": 100,
    "checkpoint_sha256": sha(a.checkpoint), "model": "mock-vit",
}
protocol_path = a.output_dir / "protocol.json"
protocol_path.write_text(json.dumps(protocol, sort_keys=True))
protocol_hash = sha(protocol_path)

def method(scale):
    token = format(scale, ".12g").replace(".", "p")
    return "expert_s" + token

def gain(backend, scale):
    if a.mode == "exact":
        return 2
    if stage == "development":
        return {("clip", .5): 4, ("mint", .5): 2,
                ("clip", 1.): 3, ("mint", 1.): 3}[(backend, scale)]
    return 2 if stage == "diagnostic_a" else 3

def write(backend, name, correct_count, mismatch=False):
    records = []
    for i in range(100):
        raw = ("f" if mismatch and i == 0 else str(i % 10)) * 64
        records.append({
            "sample": i, "target": i % 10, "prediction": i % 10 if i < correct_count else 99,
            "correct": i < correct_count, "scenario": "brightness:5", "batch": i // a.batch_size,
            "stream_position": i, "raw_sha256": raw,
        })
    payload = {
        "protocol": {"protocol_sha256": protocol_hash, "backend": backend,
                     "method": name, "seed": a.seed},
        "accuracy": float(correct_count), "records": records,
    }
    (a.output_dir / f"{backend}_{name}.json").write_text(json.dumps(payload))

for backend in backends:
    write(backend, "identity", 50)
    for scale in scales:
        write(backend, method(scale), 50 + gain(backend, scale),
              mismatch=(a.mode == "mismatch" and backend == backends[0]))
'''


class ExpertStageGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "fake_eval.py").write_text(FAKE_EVALUATOR)
        (self.root / "checkpoint.pt").write_bytes(b"checkpoint")
        for stage in MODULE.STAGES:
            ids = [f"n{i % 10:08d}/{stage}_{i}.JPEG" for i in range(100)]
            (self.root / f"{stage}.txt").write_text("\n".join(ids) + "\n")
            cache = self.root / f"cache_{stage}"
            cache.mkdir()
            ids_hash = MODULE.sha256_bytes("\n".join(ids).encode())
            (cache / "protocol.json").write_text(json.dumps({
                "stage": stage, "seed": 1109, "ids_sha256": ids_hash, "n": 100,
            }))

    def tearDown(self):
        self.temporary.cleanup()

    def manifest_payload(self, *, mode="normal", threshold=1.5):
        return {
            "schema_version": 1,
            "status": "expert_candidate_stage_gate",
            "seed": 1109,
            "batch_size": 20,
            "minimum_gain_pp": threshold,
            "selection_rule": "maximin_development_gain",
            "model_identity": "mock-vit",
            "backends": ["clip", "mint"],
            "identity_method": "identity",
            "candidates": [
                {"scale": 0.5, "method": "expert_s0p5"},
                {"scale": 1.0, "method": "expert_s1"},
            ],
            "splits": {
                stage: {"ids": f"{stage}.txt", "cache": f"cache_{stage}"}
                for stage in MODULE.STAGES
            },
            "artifacts": {"specialist_checkpoint": "checkpoint.pt"},
            "cross_stage_protocol_fields": ["checkpoint_sha256", "model"],
            "expected_protocol": {"model": "mock-vit"},
            "evaluator": {
                "script": "fake_eval.py",
                "common_args": ["--checkpoint", "{manifest_dir}/checkpoint.pt", "--mode", mode],
                "development_args": ["--scales", "{scales_csv}"],
                "diagnostic_args": ["--frozen", "{frozen_manifest}"],
            },
        }

    def write_manifest(self, payload=None, *, yaml_format=False):
        payload = payload or self.manifest_payload()
        path = self.root / ("manifest.yaml" if yaml_format else "manifest.json")
        if yaml_format:
            import yaml
            path.write_text(yaml.safe_dump(payload, sort_keys=False))
        else:
            path.write_text(json.dumps(payload))
        return path

    def test_dry_run_is_read_only_and_records_three_ordered_commands(self):
        manifest_path = self.write_manifest()
        manifest, base = MODULE.validate_manifest(manifest_path)
        output = self.root / "dry"
        plan = MODULE.dry_run_plan(manifest_path, manifest, base, output)
        self.assertFalse(output.exists())
        self.assertEqual([row["stage"] for row in plan["commands"]], list(MODULE.STAGES))
        self.assertIn(str((output / "frozen_scale_manifest.json").resolve()),
                      plan["commands"][1]["argv"])
        self.assertIn("after one development scale is frozen",
                      plan["commands"][1]["note"])
        self.assertEqual(plan["mode"], "dry_run_no_commands_executed")

    def test_maximin_freezes_one_scale_and_resume_runs_nothing(self):
        manifest_path = self.write_manifest()
        output = self.root / "run"
        state = MODULE.run_pipeline(manifest_path, output)
        frozen = json.loads((output / "frozen_scale_manifest.json").read_text())
        self.assertEqual(frozen["selected_scale"], 1.0)
        self.assertTrue(frozen["single_scale_for_all_backends"])
        self.assertEqual(frozen["selection_rule"],
                         "maximize development minimum gain across all declared backends")
        self.assertEqual(state["status"], "complete")
        before = (output / "commands.jsonl").read_text()
        resumed = MODULE.run_pipeline(manifest_path, output, resume=True)
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual((output / "commands.jsonl").read_text(), before)
        self.assertEqual(len(before.splitlines()), 6)

    def test_gain_equal_to_threshold_is_rejected(self):
        manifest_path = self.write_manifest(self.manifest_payload(mode="exact", threshold=2.0))
        with self.assertRaisesRegex(RuntimeError, "not strictly greater"):
            MODULE.run_pipeline(manifest_path, self.root / "exact")

    def test_pairing_mismatch_is_rejected(self):
        manifest_path = self.write_manifest(self.manifest_payload(mode="mismatch"))
        with self.assertRaisesRegex(RuntimeError, "not paired to identity"):
            MODULE.run_pipeline(manifest_path, self.root / "mismatch")

    def test_yaml_manifest_and_hash_change_resume_guard(self):
        manifest_path = self.write_manifest(yaml_format=True)
        manifest, _ = MODULE.validate_manifest(manifest_path)
        self.assertEqual(manifest["backends"], ["clip", "mint"])
        output = self.root / "yaml_run"
        MODULE.run_pipeline(manifest_path, output)
        manifest_path.write_text(manifest_path.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "hashes changed"):
            MODULE.run_pipeline(manifest_path, output, resume=True)

    def test_import_existing_audits_without_invoking_evaluator(self):
        payload = self.manifest_payload()
        original_manifest = self.write_manifest(payload)
        original_output = self.root / "original"
        MODULE.run_pipeline(original_manifest, original_output)
        for stage in MODULE.STAGES:
            stage_state = json.loads(
                (original_output / "stage_gate_state.json").read_text()
            )["stages"][stage]
            payload["splits"][stage]["existing_output"] = stage_state["output_dir"]
        import_manifest = self.root / "import_manifest.json"
        import_manifest.write_text(json.dumps(payload))
        imported_output = self.root / "imported"
        state = MODULE.import_existing_pipeline(import_manifest, imported_output)
        self.assertEqual(state["status"], "complete")
        events = [json.loads(line) for line in
                  (imported_output / "commands.jsonl").read_text().splitlines()]
        self.assertEqual(len(events), 3)
        self.assertTrue(all(event["argv"] is None for event in events))
        self.assertTrue(all(event["event"] == "import_existing_no_evaluator_command"
                            for event in events))
        frozen = json.loads((imported_output / "frozen_scale_manifest.json").read_text())
        self.assertEqual(frozen["selected_scale"], 1.0)
        self.assertFalse(frozen["evaluator_was_invoked_by_stage_gate"])
        before = (imported_output / "commands.jsonl").read_text()
        MODULE.import_existing_pipeline(import_manifest, imported_output, resume=True)
        self.assertEqual((imported_output / "commands.jsonl").read_text(), before)


if __name__ == "__main__":
    unittest.main()
