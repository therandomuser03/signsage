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


def test_assert_landmark_shape_rejects_non_array():
    with pytest.raises(ValueError):
        assert_landmark_shape([[0.0, 0.0, 0.0]] * 543)  # list, not ndarray


def test_assert_landmark_dtype_accepts_int_and_float():
    from app.ml.preprocessing.landmarks import assert_landmark_dtype

    assert_landmark_dtype(np.zeros((2, TOTAL_LANDMARKS, 3), dtype=np.float32))
    assert_landmark_dtype(np.zeros((2, TOTAL_LANDMARKS, 3), dtype=np.int64))


def test_assert_landmark_dtype_rejects_bool_str_object():
    from app.ml.preprocessing.landmarks import assert_landmark_dtype

    for bad in (
        np.zeros((2, TOTAL_LANDMARKS, 3), dtype=bool),
        np.array([["a", "b", "c"]] * 2, dtype=str),
        np.empty((2, TOTAL_LANDMARKS, 3), dtype=object),
    ):
        with pytest.raises(ValueError):
            assert_landmark_dtype(bad)


def test_normalize_pose_sequence_rejects_flat_dim_mismatch():
    with pytest.raises(ValueError):
        normalize_pose_sequence(np.zeros((3, 100), dtype=np.float32))


def test_normalize_pose_sequence_matches_independent_computation():
    """Golden test: center + uniform scale recomputed with pure-python
    statistics (independent of the numpy implementation under test)."""
    import statistics

    t = 2
    x = np.zeros((t, TOTAL_LANDMARKS, 3), dtype=np.float64)
    # Frame 0: hips at (2,0,0)/(4,0,0), everything else at (1,0,0).
    x[0, 23, :] = (2.0, 0.0, 0.0)
    x[0, 24, :] = (4.0, 0.0, 0.0)
    x[0, :, 0] += 1.0
    x[0, 23, 0] = 2.0
    x[0, 24, 0] = 4.0
    # Frame 1: hips at origin, everything else at (4,0,0).
    x[1, :, 0] = 4.0
    x[1, 23, :] = 0.0
    x[1, 24, :] = 0.0

    out = normalize_pose_sequence(x.reshape(t, -1)).reshape(t, TOTAL_LANDMARKS, 3)

    expected = np.zeros_like(x)
    scales = []
    for f in range(t):
        frame = x[f]
        cx = (frame[23, 0] + frame[24, 0]) / 2
        cy = (frame[23, 1] + frame[24, 1]) / 2
        cz = (frame[23, 2] + frame[24, 2]) / 2
        centered = frame - np.array([cx, cy, cz])
        expected[f] = centered
        body = centered[:33, :].ravel().tolist()
        scales.append(statistics.pstdev(body))
    scale = statistics.median(s for s in scales if s > 1e-6)
    expected = expected / scale
    assert np.allclose(out, expected, atol=1e-5)
    # Hip midpoint is exactly the root: zero in every frame.
    hip_mid = (out[:, 23, :] + out[:, 24, :]) / 2
    assert np.allclose(hip_mid, 0.0, atol=1e-6)


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


def test_prepare_live_landmarks_pads_with_zeros_and_reports_true_len():
    arr = np.random.default_rng(2).normal(size=(40, TOTAL_LANDMARKS, 3)).astype(np.float32)
    processed, true_len = prepare_live_landmarks(arr, max_frames=160)
    assert true_len == 40
    assert processed.shape == (160, RAW_FEATURE_DIM)
    assert np.all(processed[40:] == 0.0)  # padding is exact zeros
    assert np.isfinite(processed).all()


def test_prepare_live_landmarks_downsamples_long_sequences():
    arr = np.random.default_rng(3).normal(size=(200, TOTAL_LANDMARKS, 3)).astype(np.float32)
    processed, true_len = prepare_live_landmarks(arr, max_frames=160)
    assert processed.shape == (160, RAW_FEATURE_DIM)
    assert true_len == 160
    assert np.isfinite(processed).all()


def test_prepare_live_landmarks_rejects_bad_representations():
    good = np.zeros((5, TOTAL_LANDMARKS, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        prepare_live_landmarks(np.zeros((5, 10, 3), dtype=np.float32), max_frames=160)
    with pytest.raises(ValueError):
        prepare_live_landmarks(np.zeros((5, TOTAL_LANDMARKS, 3), dtype=bool), max_frames=160)
    with pytest.raises(ValueError):
        prepare_live_landmarks(good, max_frames=1)
    with pytest.raises(ValueError):
        prepare_live_landmarks(good, max_frames="160")


def test_resize_sequence_rejects_bad_target():
    seq = np.ones((5, 8), dtype=np.float32)
    for bad in (0, -3, "160", None):
        with pytest.raises(ValueError):
            resize_sequence(seq, target_len=bad)


def test_compute_motion_energy_rejects_misshapen_input():
    with pytest.raises(ValueError):
        compute_motion_energy(np.zeros((10, 100, 3), dtype=np.float32))


def test_compute_motion_energy_is_zero_for_static_landmarks():
    arr = np.ones((10, TOTAL_LANDMARKS, 3), dtype=np.float32) * 0.5
    assert compute_motion_energy(arr) == 0.0


def test_compute_motion_energy_is_positive_for_moving_landmarks():
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(10, TOTAL_LANDMARKS, 3)).astype(np.float32)
    assert compute_motion_energy(arr) > 0.0
