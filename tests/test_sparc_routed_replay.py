from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from streammint.brightness_specialist import BrightnessClippingSpecialist


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_sparc_routed_replay", ROOT / "scripts/run_sparc_routed_replay.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)

from streammint.utility_router import RISK_FEATURE_ORDER, UtilityRouter  # noqa: E402


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_hash(path)}


def make_mock_manifest(tmp_path: Path, *, rows: int = 45) -> Path:
    rng = np.random.default_rng(1109)
    images = rng.integers(0, 256, size=(rows, 8, 8, 3), dtype=np.uint8)
    image_path = tmp_path / "images.npy"
    np.save(image_path, images)

    stream = []
    keys = []
    hashes = []
    for index, image in enumerate(images):
        key = f"mixed:{index}"
        raw_hash = hashlib.sha256(image.tobytes()).hexdigest()
        keys.append(key)
        hashes.append(raw_hash)
        stream.append({
            "key": key,
            "sample": index % 13,
            "target": index % 7,
            "audit_domain": "domain-a" if index % 2 else "domain-b",
            "raw_sha256": raw_hash,
            "image": {"format": "npy", "path": str(image_path), "index": index},
        })
    index_path = tmp_path / "stream.json"
    index_path.write_text(json.dumps(stream))

    descriptors = rng.normal(size=(rows, 18)).astype(np.float64)
    feature_path = tmp_path / "features.npz"
    np.savez_compressed(
        feature_path, keys=np.asarray(keys), features=descriptors,
        raw_sha256=np.asarray(hashes, dtype="S64"),
        feature_order=np.asarray(RISK_FEATURE_ORDER),
    )

    expert_names = ("fixed", "conditional", "snow", "brightness")
    router = UtilityRouter(
        expert_names=expert_names,
        mean=np.zeros(55),
        scale=np.ones(55),
        coefficients=np.zeros((4, 55)),
        intercepts=np.asarray((2.0, 1.0, 0.5, 0.25)),
        fitted=np.ones(4, dtype=bool),
        ridge_alpha=1.0,
        feature_mode="mean_std_count",
        utility_threshold=0.0,
        min_margin=0.0,
    )
    router_path = tmp_path / "router.npz"
    router.save(str(router_path))
    conditional = tmp_path / "conditional.pt"
    snow = tmp_path / "snow.pt"
    brightness = tmp_path / "brightness.pt"
    conditional.write_bytes(b"mock conditional checkpoint")
    snow.write_bytes(b"mock snow checkpoint")
    brightness.write_bytes(b"mock brightness checkpoint")

    manifest = {
        "schema_version": 1,
        "status": "draft_not_frozen",
        "seed": 1109,
        "batch_size": 20,
        "memory_fraction": 0.68,
        "backend": {"kind": "mint", "reset": "once_before_stream", "mock_classes": 7},
        "router": {"artifact": artifact(router_path)},
        "features": {"kind": "precomputed_descriptors", "artifact": artifact(feature_path)},
        "expert_bank": [
            {"name": "fixed", "kind": "fixed", "operator": "gaussian_1.5"},
            {
                "name": "conditional", "kind": "conditional_restorer",
                "checkpoint": artifact(conditional), "condition_index": 0,
            },
            {"name": "snow", "kind": "snow_specialist", "checkpoint": artifact(snow)},
            {
                "name": "brightness",
                "kind": "brightness_specialist",
                "checkpoint": artifact(brightness),
                "source_sha256": file_hash(
                    ROOT / "src/streammint/brightness_specialist.py"
                ),
                "residual_scale": 0.75,
            },
        ],
        "input_stream": {
            "shuffle": True,
            "index": artifact(index_path),
            "source_artifacts": [artifact(image_path)],
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_mock_replay_resume_is_record_identical(tmp_path: Path):
    manifest = make_mock_manifest(tmp_path)
    resumed_dir = tmp_path / "resumed"
    assert RUNNER.run(manifest, resumed_dir, mock=True, stop_after_batches=1) is None
    resumed = RUNNER.run(manifest, resumed_dir, mock=True)
    clean = RUNNER.run(manifest, tmp_path / "clean", mock=True)
    assert resumed is not None and clean is not None
    assert resumed["records_sha256"] == clean["records_sha256"]
    assert resumed["records"] == clean["records"]
    assert resumed["protocol"]["status"] == "cpu_mock_no_scientific_evidence"
    assert resumed["expert_counts"] == {
        "identity": 0,
        "fixed": 45,
        "conditional": 0,
        "snow": 0,
        "brightness": 0,
    }


def test_brightness_mock_residual_scale_is_applied():
    images = [np.zeros((2, 2, 3), dtype=np.uint8)]
    experts = [{
        "name": "brightness",
        "kind": "brightness_specialist",
        "residual_scale": 0.75,
    }]
    output = RUNNER.apply_mock_experts(images, np.asarray([0]), experts)
    np.testing.assert_array_equal(output, np.full((1, 2, 2, 3), 3.0))


def test_brightness_source_hash_and_scale_are_validated(tmp_path: Path):
    manifest = make_mock_manifest(tmp_path, rows=2)
    payload = json.loads(manifest.read_text())
    payload["expert_bank"][-1]["source_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        RUNNER.validate_manifest(manifest, mock=True)

    manifest = make_mock_manifest(tmp_path, rows=2)
    payload = json.loads(manifest.read_text())
    payload["expert_bank"][-1]["residual_scale"] = 2.5
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="residual_scale"):
        RUNNER.validate_manifest(manifest, mock=True)


def test_real_brightness_expert_loads_and_scales_on_cpu(tmp_path: Path):
    checkpoint_path = tmp_path / "brightness.pt"
    source_path = ROOT / "src/streammint/brightness_specialist.py"
    model = BrightnessClippingSpecialist(width=4)
    torch.save(
        {
            "format": "sparc-brightness-clipping-specialist-v1",
            "model": {
                "width": 4,
                "offset": 0.5,
                "saturation_threshold": 254.5 / 255.0,
                "max_context_residual": 0.35,
            },
            "state_dict": model.state_dict(),
            "seed": 1109,
            "uses_imagenet_class_labels": False,
        },
        checkpoint_path,
    )
    specification = {
        "name": "brightness",
        "kind": "brightness_specialist",
        "checkpoint": artifact(checkpoint_path),
        "source_sha256": file_hash(source_path),
        "residual_scale": 0.75,
    }
    manifest = {
        "_path": str((tmp_path / "manifest.json").resolve()),
        "expert_bank": [specification],
    }
    experts = RUNNER.RealExperts(manifest, torch.device("cpu"))
    image = np.full((72, 72, 3), (64, 128, 255), dtype=np.uint8)
    actual = experts.apply([image], np.asarray([0]))
    source = torch.from_numpy(image).permute(2, 0, 1).float()[None] / 255.0
    native = model(source)
    expected = (source + 0.75 * (native - source)).clamp(0.0, 1.0)
    torch.testing.assert_close(actual, expected)


def test_real_replay_refuses_unfrozen_manifest(tmp_path: Path):
    manifest = make_mock_manifest(tmp_path, rows=2)
    with pytest.raises(RuntimeError, match="frozen_for_routed_replay"):
        RUNNER.validate_manifest(manifest, mock=False)


def test_source_hash_change_is_rejected(tmp_path: Path):
    manifest = make_mock_manifest(tmp_path, rows=2)
    payload = json.loads(manifest.read_text())
    Path(payload["input_stream"]["source_artifacts"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        RUNNER.validate_manifest(manifest, mock=True)


def test_resume_tampering_is_rejected(tmp_path: Path):
    manifest = make_mock_manifest(tmp_path)
    output = tmp_path / "interrupted"
    assert RUNNER.run(manifest, output, mock=True, stop_after_batches=1) is None
    with (output / "records.partial.jsonl").open("a") as target:
        target.write("\n")
    with pytest.raises(RuntimeError, match="failed hash validation"):
        RUNNER.run(manifest, output, mock=True)


def test_route_api_cannot_receive_domain_or_target():
    router = UtilityRouter(
        expert_names=("fixed",), mean=np.zeros(55), scale=np.ones(55),
        coefficients=np.zeros((1, 55)), intercepts=np.ones(1), fitted=np.ones(1, dtype=bool),
        ridge_alpha=1.0, feature_mode="mean_std_count",
    )
    descriptors = np.zeros((20, 18), dtype=np.float64)
    decisions, predicted = RUNNER.route_batch(router, descriptors)
    np.testing.assert_array_equal(decisions, np.zeros(20, dtype=np.int64))
    assert predicted.shape == (20, 1)
    with pytest.raises(TypeError):
        RUNNER.route_batch(router, descriptors, audit_domain=["snow"] * 20)
