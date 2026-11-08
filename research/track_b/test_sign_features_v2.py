"""Unit test: sign_features_v2.py matches features_v2.py output exactly.

Run:  python test_sign_features_v2.py
"""
import numpy as np
import features_v2 as fv
import sign_features_v2 as sfv


def _random_landmarks(T=37, seed=3):
    rng = np.random.RandomState(seed)
    h = np.zeros((T, 543, 3), dtype=np.float64)
    # random pose shoulders + face + hands with some missing frames
    h[:, 11, 0] = 10.0 + rng.randn(T) * 0.1
    h[:, 12, 0] = 12.0 + rng.randn(T) * 0.1
    h[:, 11, 1] = 0.5; h[:, 12, 1] = 0.5
    h[:, 34, 0] = 11.0 + rng.randn(T) * 0.05
    h[:, 34, 1] = 0.4 + rng.randn(T) * 0.05
    h[:, 185, 0] = 11.0 + rng.randn(T) * 0.05
    h[:, 185, 1] = 0.7 + rng.randn(T) * 0.05
    # hands: present in a subset of frames
    present = rng.rand(T) > 0.3
    for t in range(T):
        if present[t]:
            h[t, 501, 0] = 5.0 + t * 0.2 + rng.randn() * 0.1
            h[t, 501, 1] = 0.5
            h[t, 509, 0] = h[t, 501, 0] + 1.0
            h[t, 509, 1] = 0.5
            h[t, 522, 0] = 18.0 + rng.randn() * 0.1
            h[t, 522, 1] = 0.5
            h[t, 530, 0] = h[t, 522, 0] + 1.0
            h[t, 530, 1] = 0.5
    return h


def main():
    h = _random_landmarks()
    a = fv.extract_features(h, 'signsage')
    b = sfv.extract_features(h, 'signsage')
    assert a.shape == b.shape == (fv.N_FRAMES, fv.FEAT_DIM), (a.shape, b.shape)
    assert np.allclose(a, b, atol=1e-5), np.abs(a - b).max()
    # all-missing-hand case
    h2 = np.zeros((24, 543, 3))
    h2[:, 11, 0] = 10.0; h2[:, 12, 0] = 12.0
    h2[:, 11, 1] = 0.5; h2[:, 12, 1] = 0.5
    h2[:, 34, 0] = 11.0; h2[:, 34, 1] = 0.4
    h2[:, 185, 0] = 11.0; h2[:, 185, 1] = 0.7
    a2 = fv.extract_features(h2, 'signsage')
    b2 = sfv.extract_features(h2, 'signsage')
    assert np.allclose(a2, b2, atol=1e-5)
    assert np.allclose(a2, 0.0)
    assert np.all(a2[:, 84] == 0.0) and np.all(a2[:, 85] == 0.0)
    print("test_sign_features_v2: sign_features_v2 matches features_v2 exactly. OK")


if __name__ == "__main__":
    main()
