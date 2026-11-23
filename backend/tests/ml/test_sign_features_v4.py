"""Serving-side sign_features_v4 must reproduce the training features exactly (golden vector) and behave
sensibly on live-style captures (fewer frames, different position, different video aspect)."""
from pathlib import Path

import numpy as np

from app.ml.preprocessing.sign_features_v3 import extract_features as v3_extract
from app.ml.preprocessing.sign_features_v4 import (
    EXTRACT_FRAMES,
    extract_features_v4,
    pose_face_reference,
    prepare_landmarks,
    resample_landmarks,
)

GOLDEN = Path(__file__).parent / "data" / "fdmse_v4_golden.npz"


def _physical_clip(n: int, t0: float = 0.0, t1: float = 1.0):
    """A signing-like clip in physical units (image height = 1), sampled at n uniform times in [t0, t1]."""
    t = np.linspace(t0, t1, n)
    arr = np.zeros((n, 543, 3), np.float32)
    base = np.array([0.9, 0.5])
    arr[:, 11, :2] = base + (-0.25, 0.0)
    arr[:, 12, :2] = base + (0.25, 0.0)
    head = base + (0.0, -0.35)
    arr[:, 0, :2] = head
    arr[:, 2, :2] = head + (-0.04, -0.05)
    arr[:, 5, :2] = head + (0.04, -0.05)
    arr[:, 9, :2] = head + (-0.03, 0.06)
    arr[:, 10, :2] = head + (0.03, 0.06)
    ang = np.linspace(0, 2 * np.pi, 21)
    for slot, (sx, k) in enumerate(((0.3, 1.0), (-0.3, 1.7))):
        wx = base[0] + sx + 0.12 * np.sin(2 * np.pi * k * t)
        wy = base[1] - 0.1 + 0.12 * np.cos(2 * np.pi * t)
        sl = slice(501, 522) if slot == 0 else slice(522, 543)
        arr[:, sl, 0] = wx[:, None] + 0.06 * np.cos(ang)[None]
        arr[:, sl, 1] = wy[:, None] + 0.06 * np.sin(ang)[None]
    return arr


def _to_normalized(phys: np.ndarray, aspect: float, shift=(0.0, 0.0)) -> np.ndarray:
    out = phys.copy()
    nz = out != 0
    out[..., 0] = np.where(nz[..., 0], (out[..., 0] + shift[0]) / aspect, 0.0)
    out[..., 1] = np.where(nz[..., 1], out[..., 1] + shift[1], 0.0)
    return out


def test_golden_vector_matches_training_features():
    g = np.load(GOLDEN)
    got = extract_features_v4(g["landmarks"], float(g["aspect"]))
    assert got.shape == (60, 122) and got.dtype == np.float32
    np.testing.assert_allclose(got, g["expected"], atol=1e-5)


def test_live_face_mesh_is_ignored():
    g = np.load(GOLDEN)
    noisy = g["landmarks"].copy()
    noisy[:, 33:501] = np.random.RandomState(3).rand(len(noisy), 468, 3)
    np.testing.assert_allclose(extract_features_v4(noisy, float(g["aspect"])),
                               extract_features_v4(g["landmarks"], float(g["aspect"])), atol=1e-6)


def test_face_reference_follows_geometry():
    pose = np.zeros((1, 33, 3), np.float32)
    pose[0, 0, :2], pose[0, 2, :2], pose[0, 5, :2] = (0.5, 0.25), (0.48, 0.20), (0.52, 0.20)
    pose[0, 9, :2], pose[0, 10, :2] = (0.48, 0.30), (0.52, 0.30)
    ref = pose_face_reference(pose)
    assert np.allclose(ref[152][0], (0.50, 0.36)) and np.allclose(ref[10][0], (0.50, 0.08))   # chin below, forehead above
    assert np.allclose(ref[1][0], (0.50, 0.25)) and np.allclose(ref[13][0], (0.50, 0.30))


def test_resample_is_identity_at_48_and_has_right_shape_otherwise():
    clip = _to_normalized(_physical_clip(48), 16 / 9)
    assert np.array_equal(resample_landmarks(clip, 48), clip)
    short = _to_normalized(_physical_clip(25), 16 / 9)
    out = resample_landmarks(short, 48)
    assert out.shape == (48, 543, 3)
    assert np.allclose(out[0], short[0]) and np.allclose(out[-1], short[-1])           # endpoints preserved


def test_resample_never_invents_a_missing_hand():
    clip = _to_normalized(_physical_clip(30), 16 / 9)
    clip[10:20, 501:522] = 0                                                            # left hand missing for 10 frames
    out = resample_landmarks(clip, 48)
    present = np.any(out[:, 501:522] != 0, axis=(1, 2))
    # frames deep inside the gap must stay missing; frames well outside must be present
    assert not present[int(15 / 29 * 47)] and present[0] and present[-1]


def test_resample_does_not_blend_a_hand_that_jumped():
    clip = _to_normalized(_physical_clip(30), 16 / 9)
    clip[5, 501:522, :2] += 0.5                                                         # slot swap / glitch: huge jump
    out = resample_landmarks(clip, 48)
    # no output hand frame may lie between the jumped position and its neighbours (would be a ghost hand)
    xs = out[:, 501:522, 0].mean(axis=1)
    assert np.all(np.abs(np.diff(xs)) < 0.6)


def test_missing_pose_frames_are_filled_from_neighbours():
    clip = _to_normalized(_physical_clip(48), 16 / 9)
    clip[20:23, 0:33] = 0
    prepared = prepare_landmarks(clip, 16 / 9)
    assert np.all(np.any(prepared[:, 0:33] != 0, axis=(1, 2)))


def test_features_do_not_depend_on_where_the_signer_stands():
    a = _to_normalized(_physical_clip(48), 16 / 9)
    b = _to_normalized(_physical_clip(48), 16 / 9, shift=(0.25, 0.08))
    np.testing.assert_allclose(extract_features_v4(a, 16 / 9), extract_features_v4(b, 16 / 9), atol=1e-3)


def test_aspect_correction_makes_features_camera_independent():
    phys = _physical_clip(48)
    f169 = extract_features_v4(_to_normalized(phys, 16 / 9), 16 / 9)
    f43 = extract_features_v4(_to_normalized(phys, 4 / 3), 4 / 3)
    np.testing.assert_allclose(f169, f43, atol=1e-3)
    raw169 = extract_features_v4(_to_normalized(phys, 16 / 9), None)
    raw43 = extract_features_v4(_to_normalized(phys, 4 / 3), None)
    assert np.abs(raw169 - raw43).max() > 0.05                                           # without it the cameras disagree


def test_live_frame_rate_is_handled_by_resampling():
    """~10 fps for 3 s (30 frames) must give features close to the 48-frame training layout of the same motion;
    feeding sign_features_v3 the 30 raw frames directly (the old behaviour) must be clearly worse."""
    aspect = 16 / 9
    reference = extract_features_v4(_to_normalized(_physical_clip(48), aspect), aspect)
    live = _to_normalized(_physical_clip(30), aspect)
    ours = extract_features_v4(live, aspect)
    naive_input = prepare_landmarks(live, aspect, n_frames=30)                           # same prep, but NO resample to 48
    naive = v3_extract(naive_input, layout="signsage")
    err_ours, err_naive = float(np.abs(ours - reference).mean()), float(np.abs(naive - reference).mean())
    assert err_ours < 0.05, err_ours
    assert err_ours < 0.5 * err_naive, (err_ours, err_naive)
    assert EXTRACT_FRAMES == 48
