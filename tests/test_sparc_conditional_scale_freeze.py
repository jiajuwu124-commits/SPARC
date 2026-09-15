from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FREEZER = load_script("freeze_sparc_conditional_scale")
WRAPPER = load_script("run_sparc_conditional_frozen_eval")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "conditional.pt"
    torch.save(
        {
            "model": {"scenario_count": 1, "width": 4},
            "state_dict": {},
            "scenarios": ["glass_blur"],
            "seed": 1109,
        },
        path,
    )
    return path


def records(correct: list[bool]) -> list[dict]:
    return [
        {
            "sample": index,
            "target": index,
            "prediction": index if value else -1,
            "correct": value,
            "scenario": "glass_blur:5",
            "batch": index // 2,
            "stream_position": index,
            "raw_sha256": f"{index:x}" * 64,
        }
        for index, value in enumerate(correct)
    ]


def result_payload(
    protocol_sha: str, backend: str, method: str, correct: list[bool]
) -> dict:
    rows = records(correct)
    return {
        "protocol": {
            "protocol_sha256": protocol_sha,
            "backend": backend,
            "method": method,
            "scenario": "glass_blur:5",
            "seed": 1109,
        },
        "accuracy": 100.0 * sum(correct) / len(correct),
        "records": rows,
    }


def development_fixture(tmp_path: Path) -> tuple[Path, Path]:
    model = checkpoint(tmp_path)
    output = tmp_path / "development"
    output.mkdir()
    protocol = {
        "status": "development",
        "seed": 1109,
        "checkpoint_sha256": file_sha(model),
        "scenarios": ["glass_blur:5", "frost:5"],
        "backends": ["clip", "mint"],
        "restorer_scales": [0.5, 1.0],
        "n": 4,
    }
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    protocol_sha = file_sha(protocol_path)
    values = {
        ("clip", "identity"): [True, False, True, False],
        ("mint", "identity"): [True, False, True, False],
        ("clip", "conditional_restorer_scale0p5"): [True, True, True, True],
        ("mint", "conditional_restorer_scale0p5"): [True, True, True, True],
        ("clip", "conditional_restorer"): [True, True, True, True],
        ("mint", "conditional_restorer"): [True, True, True, True],
    }
    for (backend, method), correct in values.items():
        path = output / f"{backend}_{method}.json"
        path.write_text(json.dumps(result_payload(protocol_sha, backend, method, correct)))
    return output, model


def test_maximin_freeze_uses_same_scale_and_smaller_tie(tmp_path: Path):
    development, model = development_fixture(tmp_path)
    frozen = FREEZER.freeze_manifest(
        development,
        model,
        "glass_blur:5",
        ["clip", "mint"],
        1.5,
    )
    assert frozen["selected_residual_scale"] == 0.5
    assert frozen["selected_minimum_gain_pp"] == 50.0
    assert frozen["selected_gain_pp_by_backend"] == {"clip": 50.0, "mint": 50.0}
    assert frozen["uses_diagnostic_or_confirmation_results"] is False


def test_freeze_rejects_when_any_declared_backend_fails_threshold(tmp_path: Path):
    development, model = development_fixture(tmp_path)
    path = development / "mint_conditional_restorer_scale0p5.json"
    payload = json.loads(path.read_text())
    payload["records"] = records([True, False, True, False])
    payload["accuracy"] = 50.0
    path.write_text(json.dumps(payload))
    path = development / "mint_conditional_restorer.json"
    payload = json.loads(path.read_text())
    payload["records"] = records([True, False, True, False])
    payload["accuracy"] = 50.0
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="not strictly greater"):
        FREEZER.freeze_manifest(
            development, model, "glass_blur:5", ["clip", "mint"], 1.5
        )


def diagnostic_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    development, model = development_fixture(tmp_path)
    frozen = FREEZER.freeze_manifest(
        development, model, "glass_blur:5", ["clip", "mint"], 1.5
    )
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen))
    output = tmp_path / "diagnostic"
    output.mkdir()
    protocol = {
        "status": "diagnostic",
        "seed": 1109,
        "checkpoint_sha256": file_sha(model),
        "scenarios": ["glass_blur:5"],
        "backends": ["clip", "mint"],
        "restorer_scales": [0.5],
        "batch_size": 20,
        "n": 4,
    }
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    protocol_sha = file_sha(protocol_path)
    for backend in ("clip", "mint"):
        for method, correct in (
            ("identity", [True, False, True, False]),
            ("conditional_restorer_scale0p5", [True, True, True, True]),
        ):
            path = output / f"{backend}_{method}.json"
            path.write_text(
                json.dumps(result_payload(protocol_sha, backend, method, correct))
            )
    (output / "matched_pair_audit.json").write_text(json.dumps({"passed": True}))
    return output, frozen_path, model


def test_diagnostic_validator_requires_frozen_scale_and_exact_pairs(tmp_path: Path):
    output, frozen, model = diagnostic_fixture(tmp_path)
    audit = WRAPPER.validate_output(output, frozen, model)
    assert audit["status"] == "validated_diagnostic_frozen_scale"
    assert audit["residual_scale"] == 0.5
    assert audit["artifacts"]["clip"]["gain_pp"] == 50.0

    plugin = output / "mint_conditional_restorer_scale0p5.json"
    payload = json.loads(plugin.read_text())
    payload["records"][0]["batch"] = 99
    plugin.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="not exactly paired"):
        WRAPPER.validate_output(output, frozen, model)


def test_diagnostic_validator_rejects_checkpoint_and_scale_tampering(tmp_path: Path):
    output, frozen_path, model = diagnostic_fixture(tmp_path)
    frozen = json.loads(frozen_path.read_text())
    frozen["selected_residual_scale"] = 0.75
    frozen_path.write_text(json.dumps(frozen))
    with pytest.raises(RuntimeError, match="exactly the frozen scale"):
        WRAPPER.validate_output(output, frozen_path, model)

    second = tmp_path / "second"
    second.mkdir()
    output, frozen_path, model = diagnostic_fixture(second)
    model.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checkpoint"):
        WRAPPER.validate_output(output, frozen_path, model)
