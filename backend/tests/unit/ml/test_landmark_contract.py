"""Unit tests for LandmarkContract value validation.

Counts alone are not the contract: component ORDER and RANGES must match
exactly, or a reordered bundle would silently mistranslate. These tests prove
reordered bundles are rejected, never reinterpreted.
"""
import json

import pytest

from app.ml.contracts.landmark_contract import LandmarkContract, LandmarkContractError
from app.ml.preprocessing.landmarks import NORMALIZATION_VERSION


def _valid_manifest():
    return {
        "total_landmark_count": 543,
        "coordinates_per_landmark": 3,
        "raw_feature_dim": 1629,
        "component_order": [
            "POSE_LANDMARKS",
            "FACE_LANDMARKS",
            "LEFT_HAND_LANDMARKS",
            "RIGHT_HAND_LANDMARKS",
        ],
        "component_ranges": {
            "POSE_LANDMARKS": [0, 33],
            "FACE_LANDMARKS": [33, 501],
            "LEFT_HAND_LANDMARKS": [501, 522],
            "RIGHT_HAND_LANDMARKS": [522, 543],
        },
        "coordinate_order": ["x", "y", "z"],
        "normalization_version": NORMALIZATION_VERSION,
        "max_frames": 160,
        "final_model_input_dim": 3258,
        "full_feature_mask": list(range(1629)),
    }


def _write(tmp_path, overrides=None):
    manifest = _valid_manifest()
    manifest.update(overrides or {})
    path = tmp_path / "landmark_contract.json"
    path.write_text(json.dumps(manifest))
    return path


def test_loads_valid_manifest(tmp_path):
    contract = LandmarkContract.load(_write(tmp_path))
    assert contract.max_frames == 160
    assert contract.component_order[:4] == [
        "POSE_LANDMARKS",
        "FACE_LANDMARKS",
        "LEFT_HAND_LANDMARKS",
        "RIGHT_HAND_LANDMARKS",
    ]
    assert contract.component_ranges["LEFT_HAND_LANDMARKS"] == [501, 522]


def test_loads_real_no_go_bundle_contract():
    from pathlib import Path

    real = Path(__file__).resolve().parents[4] / "research" / "evaluation" / "reports" / (
        "352868924"
    ) / "landmark_contract.json"
    if not real.is_file():
        pytest.skip("reference bundle contract not present")
    contract = LandmarkContract.load(real)  # must not raise
    assert contract.total_landmark_count == 543


def test_rejects_swapped_component_order(tmp_path):
    path = _write(tmp_path, {"component_order": [
        "POSE_LANDMARKS",
        "FACE_LANDMARKS",
        "RIGHT_HAND_LANDMARKS",  # swapped with left hand
        "LEFT_HAND_LANDMARKS",
    ]})
    with pytest.raises(LandmarkContractError):
        LandmarkContract.load(path)


def test_rejects_wrong_component_ranges(tmp_path):
    path = _write(tmp_path, {"component_ranges": {
        "POSE_LANDMARKS": [0, 33],
        "FACE_LANDMARKS": [33, 501],
        "LEFT_HAND_LANDMARKS": [501, 522],
        "RIGHT_HAND_LANDMARKS": [500, 543],  # overlaps face range
    }})
    with pytest.raises(LandmarkContractError):
        LandmarkContract.load(path)


def test_rejects_wrong_coordinate_order(tmp_path):
    path = _write(tmp_path, {"coordinate_order": ["y", "x", "z"]})
    with pytest.raises(LandmarkContractError):
        LandmarkContract.load(path)


def test_allows_trailing_extra_component(tmp_path):
    # POSE_WORLD_LANDMARKS after index 543 is excluded by the mask, not served.
    path = _write(tmp_path, {"component_order": [
        "POSE_LANDMARKS",
        "FACE_LANDMARKS",
        "LEFT_HAND_LANDMARKS",
        "RIGHT_HAND_LANDMARKS",
        "POSE_WORLD_LANDMARKS",
    ]})
    contract = LandmarkContract.load(path)  # must not raise
    assert contract.component_order[4] == "POSE_WORLD_LANDMARKS"


def test_rejects_bad_max_frames(tmp_path):
    for bad in (0, 1, -160, "160", None):
        with pytest.raises(LandmarkContractError):
            LandmarkContract.load(_write(tmp_path, {"max_frames": bad}))


def test_missing_file_raises(tmp_path):
    with pytest.raises(LandmarkContractError):
        LandmarkContract.load(tmp_path / "does_not_exist.json")
