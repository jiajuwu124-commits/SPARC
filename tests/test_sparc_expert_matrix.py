from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from streammint.brightness_specialist import BrightnessClippingSpecialist


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_sparc_expert_matrix", ROOT / "scripts/run_sparc_expert_matrix.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(tmp_path: Path) -> Path:
    ids = ["n00000001/a.JPEG", "n00000002/b.JPEG"]
    (tmp_path / "ids.txt").write_text("\n".join(ids) + "\n")
    for name in ("meta.bin", "clip.pt"):
        (tmp_path / name).write_bytes(name.encode())
    torch.save({
        "model": {"scenario_count": 1, "width": 4},
        "state_dict": {}, "scenarios": ["snow"], "seed": 1109,
    }, tmp_path / "restorer.pt")
    torch.save({
        "format": "sparc-snow-specialist-v1", "model": {"width": 4},
        "state_dict": {}, "seed": 1109,
    }, tmp_path / "snow.pt")
    brightness_model = BrightnessClippingSpecialist(width=4)
    torch.save({
        "format": "sparc-brightness-clipping-specialist-v1",
        "model": {
            "width": 4,
            "offset": 0.5,
            "saturation_threshold": 254.5 / 255.0,
            "max_context_residual": 0.35,
        },
        "state_dict": brightness_model.state_dict(),
        "seed": 1109,
        "uses_imagenet_class_labels": False,
    }, tmp_path / "brightness.pt")
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    array = np.arange(2 * 224 * 224 * 3, dtype=np.uint8).reshape(2, 224, 224, 3)
    np.save(cache / "snow_s5.npy", array)
    protocol = {
        "schema_version": 1,
        "n": 2,
        "ids_sha256": MODULE.sequence_sha256(ids),
        "subset_seed": 1109,
        "corruption_seed": 1109,
        "files": {
            "snow_s5": {
                "path": "snow_s5.npy",
                "sha256": file_sha(cache / "snow_s5.npy"),
            }
        },
        "source_sha256": {"generator": "f" * 64},
    }
    (cache / "protocol.json").write_text(json.dumps(protocol))
    manifest = {
        "schema_version": 1,
        "status": "development",
        "backend": "stateless_clip",
        "seed": 1109,
        "memory_fraction_limit": 0.68,
        "ids": "ids.txt",
        "cache": "cache",
        "meta": "meta.bin",
        "clip_checkpoint": "clip.pt",
        "batch_size": 2,
        "workers": 0,
        "scenarios": ["snow:5"],
        "experts": [
            {"name": "identity", "type": "identity"},
            {"name": "fixed", "type": "fixed", "operator": "gaussian_1.5"},
            {
                "name": "conditional", "type": "conditional_restorer",
                "checkpoint": "restorer.pt", "residual_scale": 0.75,
                "condition": "snow",
            },
            {"name": "snow", "type": "snow_specialist", "checkpoint": "snow.pt"},
            {
                "name": "brightness",
                "type": "brightness_specialist",
                "checkpoint": "brightness.pt",
                "checkpoint_sha256": file_sha(tmp_path / "brightness.pt"),
                "source_sha256": file_sha(
                    ROOT / "src/streammint/brightness_specialist.py"
                ),
                "residual_scale": 0.75,
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


class ExpertMatrixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_manifest_builds_complete_hashed_protocol_without_gpu(self):
        path = make_fixture(self.tmp_path)
        manifest, base = MODULE.load_manifest(path)
        protocol, ids, cache = MODULE.build_protocol(path, manifest, base)
        self.assertEqual(protocol["seed"], 1109)
        self.assertEqual(protocol["memory_fraction_limit"], 0.68)
        self.assertTrue(protocol["complete_cross_product"])
        self.assertEqual(len(protocol["experts"]) * len(protocol["scenarios"]), 5)
        self.assertEqual(len(protocol["checkpoint_sha256"]), 3)
        self.assertEqual(
            protocol["checkpoint_metadata"]["conditional"]["condition"], "snow"
        )
        self.assertEqual(cache, self.tmp_path / "cache")
        self.assertEqual(ids, ["n00000001/a.JPEG", "n00000002/b.JPEG"])
        self.assertEqual(
            protocol["checkpoint_metadata"]["brightness"]["residual_scale"], 0.75
        )

    def test_brightness_specialist_hashes_are_mandatory_and_verified(self):
        path = make_fixture(self.tmp_path)
        payload = json.loads(path.read_text())
        payload["experts"][-1]["source_sha256"] = "0" * 64
        path.write_text(json.dumps(payload))
        manifest, base = MODULE.load_manifest(path)
        with self.assertRaisesRegex(RuntimeError, "source hash mismatch"):
            MODULE.build_protocol(path, manifest, base)

        path = make_fixture(self.tmp_path)
        payload = json.loads(path.read_text())
        payload["experts"][-1]["checkpoint_sha256"] = "0" * 64
        path.write_text(json.dumps(payload))
        manifest, base = MODULE.load_manifest(path)
        with self.assertRaisesRegex(RuntimeError, "checkpoint hash mismatch"):
            MODULE.build_protocol(path, manifest, base)

    def test_rejects_wrong_seed_and_cache_hash(self):
        path = make_fixture(self.tmp_path)
        payload = json.loads(path.read_text())
        payload["seed"] = 7
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "seed 1109"):
            MODULE.load_manifest(path)

        path = make_fixture(self.tmp_path)
        protocol_path = self.tmp_path / "cache" / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        protocol["files"]["snow_s5"]["sha256"] = "0" * 64
        protocol_path.write_text(json.dumps(protocol))
        manifest, base = MODULE.load_manifest(path)
        with self.assertRaisesRegex(RuntimeError, "cache array hash mismatch"):
            MODULE.build_protocol(path, manifest, base)

    def test_resume_validation_requires_exact_pairing(self):
        path = make_fixture(self.tmp_path)
        manifest, base = MODULE.load_manifest(path)
        protocol, ids, _ = MODULE.build_protocol(path, manifest, base)
        expert = protocol["experts"][0]
        expected = MODULE.run_protocol(protocol, expert, "snow:5")
        records = [
            {
                "sample": index,
                "target": index,
                "prediction": index,
                "correct": True,
                "scenario": "snow:5",
                "batch": 0,
                "stream_position": index,
                "raw_sha256": str(index) * 64,
            }
            for index in range(2)
        ]
        payload = {"protocol": expected, "accuracy": 100.0, "records": records}
        self.assertEqual(MODULE.validate_records(payload, expected, len(ids)), records)
        changed = json.loads(json.dumps(payload))
        changed["records"][0]["raw_sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "raw_sha256"):
            MODULE.validate_records(changed, expected, len(ids), records)

    def test_identity_dataset_preserves_raw_hash(self):
        path = make_fixture(self.tmp_path)
        manifest, base = MODULE.load_manifest(path)
        _, ids, cache = MODULE.build_protocol(path, manifest, base)
        cache_protocol = json.loads((cache / "protocol.json").read_text())
        dataset = MODULE.CachedScenarioDataset(
            cache, cache_protocol, "snow:5", ids,
            {"n00000001": 3, "n00000002": 9}, None,
        )
        image, target, sample, raw_hash = dataset[1]
        expected = np.load(cache / "snow_s5.npy", mmap_mode="r")[1]
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual((target, sample), (9, 1))
        self.assertEqual(raw_hash, hashlib.sha256(expected.tobytes()).hexdigest())

    def test_seeded_loaders_have_identical_order_and_raw_hashes(self):
        path = make_fixture(self.tmp_path)
        manifest, base = MODULE.load_manifest(path)
        _, ids, cache = MODULE.build_protocol(path, manifest, base)
        cache_protocol = json.loads((cache / "protocol.json").read_text())
        dataset = MODULE.CachedScenarioDataset(
            cache, cache_protocol, "snow:5", ids,
            {"n00000001": 3, "n00000002": 9}, None,
        )

        def collect():
            loader = DataLoader(
                dataset, batch_size=1, shuffle=True, num_workers=0,
                generator=torch.Generator().manual_seed(1109),
            )
            return [
                (int(sample.item()), raw_hash[0])
                for _image, _target, sample, raw_hash in loader
            ]

        self.assertEqual(collect(), collect())

    def test_brightness_specialist_loads_and_scales_on_cpu(self):
        path = make_fixture(self.tmp_path)
        manifest, base = MODULE.load_manifest(path)
        expert = manifest["experts"][-1]
        model, metadata = MODULE.load_expert(expert, base, "cpu")
        images = torch.rand((1, 3, 72, 72), generator=torch.Generator().manual_seed(1109))
        native = model(images)
        actual = MODULE.apply_expert(images, expert, model, metadata, "snow:5")
        expected = (images + 0.75 * (native - images)).clamp(0.0, 1.0)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
