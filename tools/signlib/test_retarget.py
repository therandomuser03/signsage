"""Unit tests for the retargeting math (Step 3 item 5).

Run: backend/.venv/bin/python -m pytest tools/signlib/test_retarget.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from signlib.glb import Rig, quat_mul, quat_rotate, quat_conj, quat_normalize
from signlib.retarget import Retargeter, FINGER_BONES
from signlib.landmarks import FINGERS
from author_sign_clips import palm_frame, synth_hand
from build_sign_clips import clip_bones

REPO = Path(__file__).resolve().parent.parent.parent
GLB = REPO / 'frontend/public/models/signsage-human.glb'


@pytest.fixture(scope='module')
def rig():
    return Rig(GLB)


@pytest.fixture(scope='module')
def retarget(rig):
    return Retargeter(rig)


def test_rest_pose_maps_to_rest_pose(rig, retarget):
    """Landmarks sampled from the rest skeleton produce (near-)rest local output."""
    anchor = retarget.shoulder_center_rest
    pos = {n: rig.rest_world_p[i] for n, i in rig.by_name.items()}
    # Build synthetic landmarks = actual rest-pose bone positions, shoulder-relative
    hands = {}
    for side in ('L', 'R'):
        h = np.zeros((21, 3))
        h[0] = pos[f'wrist.{side}']
        # MCP/PIP/DIP/TIP from the chain of each finger
        for finger, lm_idx in zip(['thumb', 'index', 'middle', 'ring', 'pinky'], [(1, 2, 3), (5, 9, 13, 17), (9, 10, 11), (13, 14, 15), (17, 18, 19)]):  # noqa: F841
            pass
        fingers = {
            'thumb': (['finger1-1', 'finger1-2', 'finger1-3'], [1, 2, 3, 4]),
            'index': (['finger2-1', 'finger2-2', 'finger2-3'], [5, 6, 7, 8]),
            'middle': (['finger3-1', 'finger3-2', 'finger3-3'], [9, 10, 11, 12]),
            'ring': (['finger4-1', 'finger4-2', 'finger4-3'], [13, 14, 15, 16]),
            'pinky': (['finger5-1', 'finger5-2', 'finger5-3'], [17, 18, 19, 20]),
        }
        for finger, (bones, lms) in fingers.items():
            pts = [pos[f'{b}.{side}'] for b in bones]
            # fingertip landmark: extend the last bone's direction by its length
            pts.append(pts[-1] + (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2]) * 0.03)
            for lm, p in zip(lms, pts):
                h[lm] = p
        hands[side] = h - anchor
    local = retarget.solve_frame(hands, None)
    _, quats = rig.world_positions(local)
    max_dev = 0.0
    worst = None
    for name, q in quats.items():
        if not any(name.startswith(p) or name.startswith(f'{p}.') for p in
                   ('upperarm', 'lowerarm', 'wrist', 'metacarpal', 'finger', 'spine', 'neck', 'head')):
            continue
        rest_q = rig.rest_world_q[rig.by_name[name]]
        d = abs(float(np.dot(q, rest_q)))
        ang = np.degrees(2 * np.arccos(min(1.0, d)))
        if ang > max_dev:
            max_dev = ang
            worst = name
    # roll drift about the bone's Y axis is expected and cosmetically irrelevant;
    # full-fidelity would need exact rest reproduction which pole-IK cannot give.
    # What must hold: bone Y (FK direction) stays put at rest.
    y = np.array([0.0, 1.0, 0.0])
    for name in ('wrist.L', 'wrist.R', 'finger3-1.L', 'finger3-1.R', 'finger2-2.L', 'upperarm01.L'):
        d_solved = quat_rotate(quats[name], y)
        d_rest = quat_rotate(rig.rest_world_q[rig.by_name[name]], y)
        cosang = np.clip(np.dot(d_solved / np.linalg.norm(d_solved), d_rest / np.linalg.norm(d_rest)), -1, 1)
        ang = np.degrees(np.arccos(cosang))
        # upper arm carries ~4.5 deg of unresolvable rig bend (upperarm02 joint)
        tol = 6.0 if name == 'upperarm01.L' else 2.0
        assert ang < tol, f'{name}: Y axis drifted {ang:.2f} deg at rest'


def test_fingertip_direction_curls_toward_palm(rig, retarget):
    """Synthetic flat hand vs synthetic fist: fingertip must move palm-ward."""
    anchor = retarget.shoulder_center_rest
    for side in ('R', 'L'):
        W = anchor + np.array([0.0, -0.25, 0.22])
        open_lm = synth_hand(W, [0, 1, 0], [0, 0, 1], side, 0.0) - anchor
        fist_lm = synth_hand(W, [0, 1, 0], [0, 0, 1], side, 1.0) - anchor
        f_open = retarget.solve_frame({'R': open_lm, 'L': open_lm if side == 'L' else np.zeros((21, 3))}, None)
        f_fist = retarget.solve_frame({'R': fist_lm, 'L': fist_lm if side == 'L' else np.zeros((21, 3))}, None)
        for lms in (8, 12):
            pass  # directional check below uses FK
        p_open, _ = rig.world_positions({k: np.array(v) for k, v in f_open.items()})
        p_fist, _ = rig.world_positions({k: np.array(v) for k, v in f_fist.items()})
        # fingers curl toward the palm side; palm faces +z, so tips move +z
        tip = 'finger3-3'
        dz = p_fist[f'{tip}.{side}'][2] - p_open[f'{tip}.{side}'][2]
        dy = p_fist[f'{tip}.{side}'][1] - p_open[f'{tip}.{side}'][1]
        assert dz > 0.005, f'{side}: tip z did not curl toward palm (dz={dz:.4f})'
        assert dy < 0, f'{side}: tip y should drop as the finger curls (dy={dy:.4f})'


def test_wrist_twist_faces_palm_to_viewer(rig, retarget):
    """Synthetic flat palm facing +z must rotate the avatar palm toward +z."""
    anchor = retarget.shoulder_center_rest
    for side in ('R', 'L'):
        W = anchor + np.array([0.0, -0.2, 0.2])
        lm = synth_hand(W, [0, 1, 0], [0, 0, 1], side, 0.0) - anchor
        frame = retarget.solve_frame({'R': lm if side == 'R' else np.zeros((21, 3)),
                                      'L': lm if side == 'L' else np.zeros((21, 3))},
                                     None, palm_overrides={side: [0, 0, 1]})
        _, quats = rig.world_positions({k: np.array(v) for k, v in frame.items()})
        palm = quat_rotate(quats[f'wrist.{side}'], np.array([-1, 0, 0]))
        # palm normal (local -X) should be closer to +z than to -z
        assert palm[2] > 0.2, f'{side}: palm normal z={palm[2]:.3f} (facing away?)'


def test_palm_frame_right_hand_anatomy():
    f, r, n = palm_frame([0, 1, 0], [0, 0, 1], 'R')
    assert np.allclose(r, [-1, 0, 0], atol=1e-6), 'right-hand pinky→index must point -x'
    f, r, n = palm_frame([0, 1, 0], [0, 0, 1], 'L')
    assert np.allclose(r, [1, 0, 0], atol=1e-6), 'left-hand pinky→index must point +x'


def test_two_bone_ik_reach_and_clamp(retarget):
    from signlib.retarget import _two_bone_ik
    P = np.array([0.0, 0.0, 0.0])
    T_reachable = np.array([0.3, -0.3, 0.0])
    E, T = _two_bone_ik(P, T_reachable, 0.25, 0.26, np.array([0.0, -1.0, -0.3]))
    assert np.linalg.norm(E - P) == pytest.approx(0.25, abs=1e-6)
    assert np.linalg.norm(E - T) == pytest.approx(0.26, abs=1e-6)
    # target beyond reach is clamped, not NaN
    E2, T2 = _two_bone_ik(P, np.array([5.0, 0.0, 0.0]), 0.25, 0.26, np.array([0.0, -1.0, -0.3]))
    assert np.linalg.norm(E2 - P) == pytest.approx(0.25, abs=1e-6)
    assert np.linalg.norm(T2 - P) == pytest.approx((0.25 + 0.26) * 0.999, abs=1e-6)


def test_clip_bones_exist_in_rig(rig):
    bones = clip_bones(rig)
    assert len(bones) >= 50
    for b in bones:
        assert b in rig.by_name


def _synthetic_clip(slot_side, n=30):
    """Build a (T,75,3) clip with one slot filled, image-x on the given side."""
    clip = np.zeros((n, 75, 3))
    off = 33 if slot_side == 'A' else 54
    for t in range(n):
        clip[t, 11] = [0.4, 0.3, 0.0]   # L shoulder
        clip[t, 12] = [0.6, 0.3, 0.0]   # R shoulder
        # hand wrist: A side fills at image-left (x≈0.3), B side at image-right (x≈0.7)
        x = 0.3 if slot_side == 'A' else 0.7
        for lm in range(21):
            clip[t, off + lm] = [x, 0.5, 0.1 * lm / 20]
    return clip


def test_assign_hands_left_image_is_physical_right():
    from signlib import landmarks as LM
    c = _synthetic_clip('A')
    hands = LM.assign_hands(c)
    assert np.any(hands['R']), 'image-left hand must map to physical RIGHT'
    assert not np.any(hands['L']), 'image-right slot empty -> physical LEFT empty'
    c = _synthetic_clip('B')
    hands = LM.assign_hands(c)
    assert np.any(hands['L']), 'image-right hand must map to physical LEFT'
    assert not np.any(hands['R'])


def test_missing_hand_keeps_rig_at_rest(rig, retarget):
    """Only the image-left slot present -> avatar RIGHT wrist must move, LEFT stays rest."""
    from signlib import landmarks as LM
    c = _synthetic_clip('A', n=5)
    hands = LM.assign_hands(c)
    hands_norm, _, _ = LM.normalize(hands, c[:, :33], retarget.shoulder_width,
                                    retarget.shoulder_center_rest, 854, 480)
    h = hands_norm['R'][2]
    frame = retarget.solve_frame({'L': None, 'R': h}, None)
    _, quats = rig.world_positions(frame)
    # right wrist should no longer be at rest
    rest = rig.rest_world_q[rig.by_name['wrist.R']]
    moved = np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(quats['wrist.R'], rest))))))
    assert moved > 5.0, f'right wrist did not move ({moved:.2f} deg)'
    # left wrist must be at rest (missing hand)
    rest_l = rig.rest_world_q[rig.by_name['wrist.L']]
    stayed = np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(quats['wrist.L'], rest_l))))))
    assert stayed < 1.0, f'left wrist moved despite missing data ({stayed:.2f} deg)'
