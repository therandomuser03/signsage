"""Unit tests for research/track_b/features_v3.py

Requirements verified:
  (1) output shape (60, 122), no NaN/Inf
  (2) swapping the input L/R slots gives IDENTICAL output (hand-label independence)
  (3) a mirrored copy of a clip gives near-identical output (canonicalisation)
  (4) the all-missing-hand case -> all zeros, masks 0
  (5) short / long clip resample

Run:  python test_features_v3.py
      (or pytest test_features_v3.py)
"""
import numpy as np
import features_v3 as fv

N_FRAMES = fv.N_FRAMES
FEAT_DIM = fv.FEAT_DIM


def _make_clip(T=24):
    """SignSage (T,543,3): moving hand on image-LEFT, static hand on image-RIGHT."""
    h = np.zeros((T, 543, 3), dtype=np.float64)
    h[:, 11, 0] = 10.0; h[:, 12, 0] = 12.0   # shoulders -> midpoint x=11
    h[:, 11, 1] = 0.5; h[:, 12, 1] = 0.5
    h[:, 501, 0] = np.arange(T) * 0.25; h[:, 501, 1] = 0.5     # slot0 wrist (left, moving)
    h[:, 509, 0] = np.arange(T) * 0.25 + 1.0; h[:, 509, 1] = 0.5  # slot0 index
    h[:, 522, 0] = 20.0; h[:, 522, 1] = 0.5                    # slot1 wrist (right, static)
    h[:, 530, 0] = 21.0; h[:, 530, 1] = 0.5                    # slot1 index
    h[:, 34, 0] = 11.0; h[:, 34, 1] = 0.4                      # face nose
    h[:, 185, 0] = 11.0; h[:, 185, 1] = 0.7                    # face chin
    h[:, 46, 0] = 11.0; h[:, 46, 1] = 0.5                      # face mouth
    h[:, 43, 0] = 11.0; h[:, 43, 1] = 0.3                      # face forehead
    return h


def test_shape_and_no_nan():
    f = fv.extract_features(_make_clip(), 'signsage')
    assert f.shape == (N_FRAMES, FEAT_DIM), f.shape
    assert f.dtype == np.float32
    assert not np.isnan(f).any()
    assert not np.isinf(f).any()


def test_all_missing_hand():
    h = np.zeros((24, 543, 3), dtype=np.float64)
    h[:, 11, 0] = 10.0; h[:, 12, 0] = 12.0
    h[:, 11, 1] = 0.5; h[:, 12, 1] = 0.5
    h[:, 34, 0] = 11.0; h[:, 34, 1] = 0.4
    h[:, 185, 0] = 11.0; h[:, 185, 1] = 0.7
    f = fv.extract_features(h, 'signsage')
    assert f.shape == (N_FRAMES, FEAT_DIM)
    assert not np.isnan(f).any()
    assert np.allclose(f, 0.0), f[np.abs(f) > 0][:5]
    assert np.all(f[:, 84] == 0.0) and np.all(f[:, 85] == 0.0)


def test_lr_slot_swap_identical():
    h = _make_clip()
    f = fv.extract_features(h, 'signsage')
    h_swapped = h.copy()
    h_swapped[:, 501:522, :] = h[:, 522:543, :]
    h_swapped[:, 522:543, :] = h[:, 501:522, :]
    f_swapped = fv.extract_features(h_swapped, 'signsage')
    assert np.allclose(f, f_swapped, atol=1e-5), np.abs(f - f_swapped).max()


def test_mirrored_copy_near_identical():
    h = _make_clip()
    f = fv.extract_features(h, 'signsage')
    h_mirror = h.copy()
    mid = 11.0
    h_mirror[:, 33:500, 0] = 2 * mid - h[:, 33:500, 0]
    h_mirror[:, 501:543, 0] = 2 * mid - h[:, 501:543, 0]
    f_mirror = fv.extract_features(h_mirror, 'signsage')
    assert np.allclose(f, f_mirror, atol=1e-4), np.abs(f - f_mirror).max()


def test_resample_short_and_long():
    f_short = fv.extract_features(_make_clip(T=5), 'signsage')
    assert f_short.shape == (N_FRAMES, FEAT_DIM)
    assert np.allclose(f_short[-1], f_short[4])
    f_long = fv.extract_features(_make_clip(T=120), 'signsage')
    assert f_long.shape == (N_FRAMES, FEAT_DIM)


if __name__ == "__main__":
    tests = [
        test_shape_and_no_nan,
        test_all_missing_hand,
        test_lr_slot_swap_identical,
        test_mirrored_copy_near_identical,
        test_resample_short_and_long,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    if failed:
        raise SystemExit(1)
    print("test_features_v3: ALL OK")
