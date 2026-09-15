from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_sparc_dev15_descriptor_cache",
    ROOT / "scripts/build_sparc_dev15_descriptor_cache.py",
)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)


def test_frozen_descriptor_definition_is_exactly_18d():
    assert BUILDER.RISK_FEATURE_ORDER == BUILDER.EXPECTED_FEATURE_ORDER
    assert len(BUILDER.RISK_FEATURE_ORDER) == 18
    assert BUILDER.RISK_BANK == (
        "identity",
        "gaussian_1.0",
        "gaussian_1.5",
        "median_3",
    )
    assert len(BUILDER.SCENARIOS) == 15


def test_cpu_extraction_records_raw_hash_and_finite_descriptor():
    rng = np.random.default_rng(1109)
    BUILDER._WORKER_IMAGES = rng.integers(
        0, 256, size=(2, 12, 13, 3), dtype=np.uint8
    )
    index, risks, raw_hash = BUILDER._extract_index(1)
    assert index == 1
    assert risks.shape == (4, 2)
    assert risks.dtype == np.float32
    assert np.isfinite(risks).all()
    assert len(raw_hash) == 64
    features = BUILDER.risk_features(risks[None])
    BUILDER.validate_arrays(
        risks[None],
        features,
        np.asarray([0], dtype=np.int64),
        np.asarray([raw_hash], dtype="S64"),
        1,
    )


def test_array_audit_rejects_nonfinite_and_reordered_samples():
    risks = np.ones((2, 4, 2), dtype=np.float32)
    features = BUILDER.risk_features(risks)
    hashes = np.asarray(["a" * 64, "b" * 64], dtype="S64")
    with pytest.raises(AssertionError):
        BUILDER.validate_arrays(
            risks, features, np.asarray([1, 0], dtype=np.int64), hashes, 2
        )
    features[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="invalid values"):
        BUILDER.validate_arrays(
            risks, features, np.arange(2, dtype=np.int64), hashes, 2
        )


def test_source_protocol_excludes_labels_results_and_untouched_data(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "protocol.json").write_text("{}")
    protocol = BUILDER.source_protocol(
        cache,
        {"ids_source": "development_ids.txt", "ids_sha256": "a" * 64},
        {scenario: "b" * 64 for scenario in BUILDER.SCENARIOS},
        4,
    )
    assert protocol["status"] == "development_only_label_free_descriptor_cache"
    assert protocol["uses_class_labels"] is False
    assert protocol["reads_imagenet_metadata"] is False
    assert protocol["reads_model_predictions"] is False
    assert protocol["reads_results_or_untouched_data"] is False
    assert protocol["total_rows"] == 15000
    assert protocol["descriptor_shape"] == [18]
