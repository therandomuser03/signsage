"""Unit tests for app.ml.preprocessing -- pure numpy, no model/checkpoint needed."""
import numpy as np
import pytest

from app.ml.preprocessing.landmarks import RAW_FEATURE_DIM, TOTAL_LANDMARKS, assert_landmark_shape
from app.ml.preprocessing.normalization import normalize_pose_sequence
from app.ml.preprocessing.sequence import compute_motion_energy, prepare_live_landmarks, resize_sequence


def test_assert_landmark_shape_accepts_correct_shape():
    arr = np.zeros((5, TOTAL_LANDMARKS, 3), dtype=np.float32)
    assert_landmark_shape(arr)  # should not raise


def test_assert_landmark_shape_rejects_wrong_landmark_count():
    arr = np.zeros((5, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        assert_landmark_shape(arr)


def test_normalize_pose_sequence_centers_on_hip_midpoint():
    x = np.zeros((3, TOTAL_LANDMARKS, 3), dtype=np.float32)
    x[:, 23, :] = 1.0
    x[:, 24, :] = 3.0  # hip midpoint = 2.0
    x[:, 0, :] = 5.0  # nonzero body spread, so scale != 1.0
    normalized = normalize_pose_sequence(x.reshape(3, -1))
    result = normalized.reshape(3, TOTAL_LANDMARKS, 3)
    hip_mid = (result[:, 23, :] + result[:, 24, :]) / 2
    assert np.allclose(hip_mid, 0.0, atol=1e-5)


def test_normalize_pose_sequence_handles_degenerate_input_without_nan():
    x = np.zeros((3, RAW_FEATURE_DIM), dtype=np.float32)  # fully static, zero spread
    out = normalize_pose_sequence(x)
    assert np.isfinite(out).all()


def test_resize_sequence_pads_short_sequences():
    seq = np.ones((3, 8), dtype=np.float32)
    out = resize_sequence(seq, target_len=5)
    assert out.shape == (5, 8)
    assert np.all(out[3:] == 0.0)


def test_resize_sequence_downsamples_long_sequences():
    seq = np.arange(20 * 8, dtype=np.float32).reshape(20, 8)
    out = resize_sequence(seq, target_len=5)
    assert out.shape == (5, 8)


def test_resize_sequence_passthrough_when_already_target_length():
    seq = np.ones((5, 8), dtype=np.float32)
    out = resize_sequence(seq, target_len=5)
    assert np.array_equal(out, seq)


def test_prepare_live_landmarks_rejects_too_few_frames():
    arr = np.zeros((1, TOTAL_LANDMARKS, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        prepare_live_landmarks(arr, max_frames=160)


def test_prepare_live_landmarks_returns_expected_shapes():
    arr = np.random.default_rng(0).normal(size=(40, TOTAL_LANDMARKS, 3)).astype(np.float32)
    processed, true_len = prepare_live_landmarks(arr, max_frames=160)
    assert processed.shape == (160, RAW_FEATURE_DIM)
    assert true_len == 40


def test_compute_motion_energy_is_zero_for_static_landmarks():
    arr = np.ones((10, TOTAL_LANDMARKS, 3), dtype=np.float32) * 0.5
    assert compute_motion_energy(arr) == 0.0


def test_compute_motion_energy_is_positive_for_moving_landmarks():
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(10, TOTAL_LANDMARKS, 3)).astype(np.float32)
    assert compute_motion_energy(arr) > 0.0
