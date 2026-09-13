import numpy as np
import pytest

from app.services.pose_preprocessing import (
    FULL_FEATURE_DIM,
    NOFACE_FEATURE_INDICES,
    normalize_pose_sequence,
    prepare_landmarks,
    resize_sequence,
)


def make_points(frames=5):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(frames, 543, 3)).astype(np.float32)
    # Make hips finite and non-degenerate.
    x[:, 23, :] += np.array([10, 20, 30], dtype=np.float32)
    x[:, 24, :] += np.array([10, 20, 30], dtype=np.float32)
    return x


def test_normalization_shape_and_finiteness():
    x = make_points()
    out = normalize_pose_sequence(x)
    assert out.shape == (5, FULL_FEATURE_DIM)
    assert np.isfinite(out).all()


def test_resize_padding_and_true_length():
    x = np.ones((3, 4), dtype=np.float32)
    out, true_len = resize_sequence(x, 5)
    assert out.shape == (5, 4)
    assert true_len == 3
    assert np.all(out[3:] == 0)


def test_prepare_normalized_full_and_noface():
    x = make_points(4)
    normalized = normalize_pose_sequence(x).reshape(4, 543, 3)

    full = prepare_landmarks(
        normalized,
        max_frames=8,
        no_face=False,
        input_mode="normalized",
    )
    noface = prepare_landmarks(
        normalized,
        max_frames=8,
        no_face=True,
        input_mode="normalized",
    )
    assert full.features.shape == (8, 1629)
    assert noface.features.shape == (8, 225)
    assert np.array_equal(
        NOFACE_FEATURE_INDICES,
        np.arange(468 * 3, 543 * 3),
    )


def test_invalid_shape():
    with pytest.raises(ValueError):
        prepare_landmarks(
            np.zeros((4, 542, 3), dtype=np.float32),
            max_frames=160,
        )


def test_nonfinite_rejected_for_live_contract():
    x = make_points(2)
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        prepare_landmarks(x, max_frames=160, input_mode="normalized")
