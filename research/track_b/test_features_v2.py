"""Unit tests for research/track_b/features_v2.py

Run:  python test_features_v2.py
      (or pytest test_features_v2.py if pytest is installed)
Covers: output shape (60, 122), no NaN/Inf, and the all-missing-hand case
(every hand-derived channel must be exactly 0 with mask=0, no NaN).
"""
import numpy as np
import features_v2 as fv

N_FRAMES = fv.N_FRAMES
FEAT_DIM = fv.FEAT_DIM


def _make_landmarks(T=24):
    """SignSage (T,543,3): moving left hand, static right hand, shoulders, face."""
    h = np.zeros((T, 543, 3), dtype=np.float64)
    h[:, 11, 0] = 10.0; h[:, 12, 0] = 12.0   # shoulders
    h[:, 11, 1] = 0.5;  h[:, 12, 1] = 0.5
    h[:, 501, 0] = np.arange(T); h[:, 501, 1] = 0.5          # l wrist moves
    h[:, 509, 0] = np.arange(T) + 1.0; h[:, 509, 1] = 0.5    # l index tip
    h[:, 522, 0] = 20.0; h[:, 522, 1] = 0.5                  # r wrist static
    h[:, 530, 0] = 21.0; h[:, 530, 1] = 0.5                  # r index tip
    h[:, 34, 0] = 11.0; h[:, 34, 1] = 0.4                    # face nose (abs 33+1)
    h[:, 185, 0] = 11.0; h[:, 185, 1] = 0.7                  # face chin (abs 33+152)
    return h


def test_shape_and_no_nan():
    f = fv.extract_features(_make_landmarks(), 'signsage')
    assert f.shape == (N_FRAMES, FEAT_DIM), f.shape
    assert f.dtype == np.float32
    assert not np.isnan(f).any()
    assert not np.isinf(f).any()
    # both hands present all frames -> masks are 1
    assert np.all(f[:, 84] == 1.0) and np.all(f[:, 85] == 1.0)


def test_all_missing_hand():
    h = np.zeros((24, 543, 3), dtype=np.float64)
    h[:, 11, 0] = 10.0; h[:, 12, 0] = 12.0
    h[:, 11, 1] = 0.5; h[:, 12, 1] = 0.5
    h[:, 34, 0] = 11.0; h[:, 34, 1] = 0.4   # face present
    h[:, 185, 0] = 11.0; h[:, 185, 1] = 0.7
    # both hands all-zero -> missing
    f = fv.extract_features(h, 'signsage')
    assert f.shape == (N_FRAMES, FEAT_DIM)
    assert not np.isnan(f).any()
    assert not np.isinf(f).any()
    assert np.allclose(f, 0.0), f[np.abs(f) > 0][:5]
    assert np.all(f[:, 84] == 0.0) and np.all(f[:, 85] == 0.0)


def test_one_hand_missing():
    h = _make_landmarks()
    # blank out the right hand entirely
    h[:, 522:543, :] = 0.0
    f = fv.extract_features(h, 'signsage')
    assert f.shape == (N_FRAMES, FEAT_DIM)
    assert not np.isnan(f).any()
    assert np.all(f[:, 85] == 0.0)            # right mask 0
    assert np.all(f[:, 84] == 1.0)            # left mask 1
    # right-hand channels all zero
    assert np.allclose(f[:, 42:84], 0.0)
    assert np.allclose(f[:, 120:122], 0.0)    # right wrist velocity zero
    # left hand is moving, so left velocity should be nonzero
    assert np.abs(f[:, 118:120]).max() > 0


def test_short_gap_interpolated_long_gap_zero():
    T = 20
    h = np.zeros((T, 543, 3), dtype=np.float64)
    h[:, 11, 0] = 10.0; h[:, 12, 0] = 12.0
    h[:, 11, 1] = 0.5; h[:, 12, 1] = 0.5
    h[:, 34, 0] = 11.0; h[:, 34, 1] = 0.4
    h[:, 185, 0] = 11.0; h[:, 185, 1] = 0.7
    h[0, 501, 0] = 0.0;  h[0, 501, 1] = 0.5     # present
    h[6, 501, 0] = 10.0; h[6, 501, 1] = 0.5     # present (gap 5 between)
    h[8, 501, 0] = 100.0; h[8, 501, 1] = 0.5    # present
    h[15, 501, 0] = 100.0; h[15, 501, 1] = 0.5  # present (gap 6 between)
    f = fv.extract_features(h, 'signsage', n_frames=T)
    lwrist_x = f[:, 0] * 2.0 + 11.0             # undo normalisation
    for t in range(1, 6):                       # gap of 5 -> interpolated
        expected = (t / 6.0) * 10.0
        assert abs(lwrist_x[t] - expected) < 1e-4, (t, lwrist_x[t], expected)
        assert f[t, 84] == 1.0                  # interpolated -> mask 1
    for t in range(9, 15):                      # gap of 6 -> stays missing
        assert f[t, 0] == 0.0, (t, f[t, 0])
        assert f[t, 84] == 0.0                  # mask 0


def test_resample_short_and_long():
    # short clip T=5 -> edge pad to 60
    h = _make_landmarks(T=5)
    f = fv.extract_features(h, 'signsage')
    assert f.shape == (N_FRAMES, FEAT_DIM)
    assert np.allclose(f[-1], f[4])
    # long clip T=120 -> subsample to 60
    h = _make_landmarks(T=120)
    f = fv.extract_features(h, 'signsage')
    assert f.shape == (N_FRAMES, FEAT_DIM)


if __name__ == "__main__":
    tests = [
        test_shape_and_no_nan,
        test_all_missing_hand,
        test_one_hand_missing,
        test_short_gap_interpolated_long_gap_zero,
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
    print("test_features_v2: ALL OK")
