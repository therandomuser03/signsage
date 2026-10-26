from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from app.core.config import Settings
from app.core.errors import ModelNotPromotedError
from app.ml.isolated_words import (
    FEATURE_DIM,
    FEATURE_VERSION,
    FRAME_COUNT,
    IsolatedWordAdapter,
    IsolatedWordNet,
    prepare_isolated_features,
    split_isolated_rows,
)
from app.ml.models import registry


@pytest.fixture
def isolated_bundle(tmp_path):
    labels = ["hello", "school"]
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": IsolatedWordNet(len(labels)).state_dict()}, checkpoint)
    evaluation = {
        "status": "NO-GO",
        "signer_independent": False,
        "tests_passed": False,
        "promotion_eligible": False,
    }
    metadata = {
        "format": "signsage_isolated_words_v1",
        "labels": labels,
        "feature_dim": FEATURE_DIM,
        "feature_version": FEATURE_VERSION,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evaluation": evaluation,
        "promotion_eligible": False,
    }
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "evaluation_results.json").write_text(json.dumps(evaluation))
    return tmp_path


def test_preprocessing_matches_live_543_landmark_contract():
    landmarks = np.zeros((8, 543, 3), dtype=np.float32)
    landmarks[:, 23, 0] = 0.4
    landmarks[:, 24, 0] = 0.6
    landmarks[:, 11, 0] = 0.3
    landmarks[:, 12, 0] = 0.7
    result = prepare_isolated_features(landmarks)
    assert result.shape == (FRAME_COUNT, FEATURE_DIM)
    assert result.dtype == np.float32


@pytest.mark.parametrize(
    "landmarks",
    [
        np.zeros((8, 10, 3), dtype=np.float32),
        np.zeros((1, 543, 3), dtype=np.float32),
        np.full((8, 543, 3), np.nan, dtype=np.float32),
    ],
)
def test_preprocessing_rejects_invalid_landmarks(landmarks):
    with pytest.raises(ValueError):
        prepare_isolated_features(landmarks)


def test_split_groups_repeated_source_recordings_and_is_reproducible():
    rows = [
        {
            "normalized_word": label,
            "signer": "shared-recording" if index == 0 else f"{label}-{index}",
            "video_path": f"{label}/{index}.mp4",
        }
        for label in ("hello", "school")
        for index in range(10)
    ]
    first = split_isolated_rows(rows, 42)
    second = split_isolated_rows(rows, 42)
    assert first == second
    placements = {
        split for split, samples in first.items()
        if any(row["signer"] == "shared-recording" for row in samples)
    }
    assert len(placements) == 1
    for split, samples in first.items():
        expected = 6 if split == "train" else 2
        assert sum(row["normalized_word"] == "hello" for row in samples) == (
            expected
        )
        assert sum(row["normalized_word"] == "school" for row in samples) == (
            expected
        )


def test_adapter_loads_real_bundle_format_and_runs_eval_self_test(isolated_bundle):
    adapter = IsolatedWordAdapter.load(isolated_bundle, device="cpu")
    assert adapter.labels == ["hello", "school"]
    assert not adapter.model.training
    assert adapter.promotion_eligible() is False


def test_adapter_rejects_tampered_checkpoint_hash(isolated_bundle):
    metadata_path = isolated_bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["checkpoint_sha256"] = "invalid"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum"):
        IsolatedWordAdapter.load(isolated_bundle, device="cpu")


def test_evaluation_thresholds_prevent_false_go_metadata_from_promoting(isolated_bundle):
    metadata_path = isolated_bundle / "metadata.json"
    evaluation_path = isolated_bundle / "evaluation_results.json"
    metadata = json.loads(metadata_path.read_text())
    evaluation = {
        "status": "GO",
        "signer_independent": True,
        "tests_passed": True,
        "promotion_eligible": True,
        "test": {
            "accuracy": 0.56,
            "macro_recall": 0.56,
            "per_class_recall": {"hello": 1.0, "school": 0.12},
        },
    }
    metadata["evaluation"] = evaluation
    metadata["promotion_eligible"] = True
    metadata_path.write_text(json.dumps(metadata))
    evaluation_path.write_text(json.dumps(evaluation))
    assert IsolatedWordAdapter.load(isolated_bundle, device="cpu").promotion_eligible() is False


def test_registry_caches_isolated_adapter_and_keeps_promotion_gate_closed(isolated_bundle):
    registry.reset_registry()
    settings = Settings(
        model_kind="isolated_words",
        model_bundle_dir=str(isolated_bundle),
        model_promoted=False,
        device="cpu",
    )
    try:
        with pytest.raises(ModelNotPromotedError):
            registry.get_sign_to_text_pipeline(settings)
        first = registry.get_sign_to_text_pipeline(settings, allow_unpromoted=True)
        second = registry.get_sign_to_text_pipeline(settings, allow_unpromoted=True)
        assert first is second
        settings.model_promoted = True
        with pytest.raises(ModelNotPromotedError, match="signer-independent"):
            registry.get_sign_to_text_pipeline(settings)
    finally:
        registry.reset_registry()
