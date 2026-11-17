"""Retarget normalized MediaPipe landmarks onto the MakeHuman GLB rig.

Strategy per frame:
  * arms  — analytic two-bone IK in the plane P->T with a pole vector
            (elbows point down & slightly back). Upper arm rotates about
            upperarm01 only; lowerarm01 carries the elbow. Twist preserved
            from rest via minimal-arc swing composition.
  * wrist — swing to aim the hand at the middle-MCP direction, plus twist
            about that axis to align the palm normal with the landmark palm
            normal (cross(indexMCP-wrist, pinkyMCP-wrist), sign per side).
  * fingers — per-bone minimal-arc swing from the rest bone direction onto
            the corresponding landmark segment direction (twist kept minimal).
  * metacarpals — same swing, damped to 60% to avoid hand deformation noise.

All rotations are expressed as absolute local quaternions for output frames.
"""
import numpy as np
from . import glb
from .glb import (quat_mul, quat_conj, quat_normalize, quat_slerp, quat_rotate,
                  quat_from_axis_angle, swing_arc, Rig)
from .landmarks import FINGERS, WRIST

# MakeHuman finger numbering: finger1 = thumb .. finger5 = pinky
FINGER_BONES = {
    'thumb': ['finger1-1', 'finger1-2', 'finger1-3'],
    'index': ['finger2-1', 'finger2-2', 'finger2-3'],
    'middle': ['finger3-1', 'finger3-2', 'finger3-3'],
    'ring': ['finger4-1', 'finger4-2', 'finger4-3'],
    'pinky': ['finger5-1', 'finger5-2', 'finger5-3'],
}
METACARPAL_BONES = {'index': 'metacarpal1', 'middle': 'metacarpal2', 'ring': 'metacarpal3', 'pinky': 'metacarpal4'}
METACARPAL_LM = {'index': 5, 'middle': 9, 'ring': 13, 'pinky': 17}
# MediaPipe finger joint landmarks (proximal→distal): mcp..tip
FINGER_LM = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20],
}
# The palm faces along local -X for both hands in this MakeHuman rig
# (verified against rest pose: n_lm ≈ -(wrist local X in world), both sides).
PALM_LOCAL_AXIS = np.array([-1.0, 0.0, 0.0])

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def _two_bone_ik(P, T, l1, l2, pole):
    """Return (elbow_pos, ) for shoulder P, wrist target T, link lengths."""
    dvec = T - P
    d = np.linalg.norm(dvec)
    d = np.clip(d, abs(l1 - l2) + 1e-4, (l1 + l2) * 0.999)
    dvec = _unit(dvec) * d
    proj = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = np.sqrt(max(l1 * l1 - proj * proj, 1e-12))
    perp = pole - np.dot(pole, dvec) / np.dot(dvec, dvec) * dvec
    if np.linalg.norm(perp) < 1e-6:
        perp = np.array([0.0, 0.0, -1.0]) - np.dot(np.array([0.0, 0.0, -1.0]), dvec) * _unit(dvec)
    perp = _unit(perp)
    return P + dvec * (proj / d) + perp * h, P + dvec

def _swing_between(a, b):
    return swing_arc(_unit(a), _unit(b))

class Retargeter:
    def __init__(self, rig: Rig):
        self.rig = rig
        r = rig
        self.shoulder_center_rest = (r.rest_world_p[r.by_name['shoulder01.L']] + r.rest_world_p[r.by_name['shoulder01.R']]) / 2.0
        self.shoulder_width = np.linalg.norm(
            r.rest_world_p[r.by_name['shoulder01.L']] - r.rest_world_p[r.by_name['shoulder01.R']])
        # link lengths from rest
        self.L1 = {}
        self.L2 = {}
        self.poles = {}
        for s in ('L', 'R'):
            ua = r.rest_world_p[r.by_name[f'upperarm01.{s}']]
            la = r.rest_world_p[r.by_name[f'lowerarm01.{s}']]
            wr = r.rest_world_p[r.by_name[f'wrist.{s}']]
            self.L1[s] = np.linalg.norm(la - ua)
            self.L2[s] = np.linalg.norm(wr - la)
            # pole reference = rest elbow's offset from the shoulder→wrist axis.
            # Keeps the rest pose reproduced exactly and elbows pointing the
            # anatomical way (out & slightly back) during motion.
            d = wr - ua
            dist = np.linalg.norm(d)
            dd = d / dist
            proj = (self.L1[s] ** 2 - self.L2[s] ** 2 + dist ** 2) / (2 * dist)
            perp = la - (ua + dd * proj)
            n = np.linalg.norm(perp)
            self.poles[s] = perp / n if n > 1e-9 else np.array([0.0, -1.0, -0.35])
        # bind-reference palm-aim directions (wrist→MCP) per finger, used to
        # retarget deltas instead of absolute orientations (the MakeHuman rig's
        # metacarpal axes diverge ~30° from wrist→MCP at bind pose; aiming
        # absolutely would dislocate fingers at rest)
        self.mc_bind_dir = {}
        self.wrist_bind_dir = {}
        for side in ('L', 'R'):
            w = r.rest_world_p[r.by_name[f'wrist.{side}']]
            mid_mcp = r.rest_world_p[r.by_name[f'finger3-1.{side}']]
            self.wrist_bind_dir[side] = _unit(mid_mcp - w)
            for finger, mc_bone in METACARPAL_BONES.items():
                mc_child = {'index': 'finger2-1', 'middle': 'finger3-1',
                            'ring': 'finger4-1', 'pinky': 'finger5-1'}[finger]
                mcp = r.rest_world_p[r.by_name[f'{mc_child}.{side}']]
                self.mc_bind_dir[(finger, side)] = _unit(mcp - w)

    # ---------- per-frame solve ----------
    def solve_frame(self, hands_norm, pose_norm, palm_overrides=None):
        """hands_norm: {'L','R' -> (21,3) meters, shoulder-center relative}
           pose_norm: (33,3) same frame (unused; kept for call-site symmetry)
           palm_overrides: optional {'L','R' -> n (3,)} exact palm normals for
           authored clips (skips the sign/branch heuristic entirely).
           Returns dict boneName -> local xyzw (full overrides)."""
        rig = self.rig
        anchor = self.shoulder_center_rest
        local = {name: q.copy() for name, q in ((n, rig.rest_local_q[rig.by_name[n]]) for n in rig.names)}

        for side in ('L', 'R'):
            h = hands_norm.get(side)
            if h is None or not np.any(h):
                continue  # stay at rest
            wrist_t = anchor + h[WRIST]
            P = rig.rest_world_p[rig.by_name[f'upperarm01.{side}']]
            # pole: rest-elbow reference (out & slightly back) — anatomical
            pole = self.poles[side]
            E, Tt = _two_bone_ik(P, wrist_t, self.L1[side], self.L2[side], pole)

            # --- upper arm: swing rest world dir onto P->E ---
            rest_upper_dir = rig.rest_world_dir(f'upperarm01.{side}')
            target_upper_dir = _unit(E - P)
            swing_u = _swing_between(rest_upper_dir, target_upper_dir)
            # --- lower arm carried through upper swing, then aimed at wrist ---
            rest_lower_dir = rig.rest_world_dir(f'lowerarm01.{side}')
            carried = quat_mul(swing_u, rig.rest_world_q[rig.by_name[f'lowerarm01.{side}']])
            carried_dir = quat_rotate(carried, np.array([0, 1, 0]))
            target_lower_dir = _unit(Tt - E)
            swing_l = _swing_between(carried_dir, target_lower_dir)
            world_lower = quat_mul(swing_l, carried)
            world_upper = quat_mul(swing_u, rig.rest_world_q[rig.by_name[f'upperarm01.{side}']])

            # upperarm02 follows upperarm01 exactly (rigid pair): local stays rest
            # convert world orientations to local
            clav = rig.by_name[f'shoulder01.{side}']
            world_clav = rig.rest_world_q[clav]  # clavicle/shoulder01 unrotated
            local[f'upperarm01.{side}'] = quat_normalize(quat_mul(quat_conj(world_clav), world_upper))
            world_upperarm02 = quat_mul(world_upper, rig.rest_local_q[rig.by_name[f'upperarm02.{side}']])
            local[f'lowerarm01.{side}'] = quat_normalize(quat_mul(quat_conj(world_upperarm02), world_lower))
            # lowerarm02 local stays rest -> rigid forearm

            # --- wrist ---
            world_lower02 = quat_mul(world_lower, rig.rest_local_q[rig.by_name[f'lowerarm02.{side}']])
            wrist_rest = rig.rest_world_q[rig.by_name[f'wrist.{side}']]
            carried_wrist = quat_mul(world_lower02, rig.rest_local_q[rig.by_name[f'wrist.{side}']])
            f_dir = _unit(h[9] - h[0])  # wrist -> middle MCP (palm aim)
            swing_now = _swing_between(quat_rotate(carried_wrist, np.array([0, 1, 0])), f_dir)
            swing_bind = _swing_between(rig.rest_world_dir(f'wrist.{side}'), self.wrist_bind_dir[side])
            swing_w = quat_mul(swing_now, quat_conj(swing_bind))
            after_swing = quat_mul(swing_w, carried_wrist)
            # twist: align palm normal (local Z) to landmark palm normal
            idx, pnk = h[5], h[17]
            override = (palm_overrides or {}).get(side)
            if override is not None:
                n_lm = _unit(np.asarray(override, dtype=np.float64))
            else:
                # Chirality differs between hands: for landmark data wound like
                # the rig's (real data), cross(idx-w, pnk-w) is palm-ward; for
                # mirror-wound data (our synthetic authored hands) it is not.
                # The flip matches the mirror-wound convention; the carried-
                # nearest-branch pick corrects any residual mismatch for real
                # data (rest-anchored, can never introduce a >90° flip alone).
                n_lm = np.cross(idx - h[0], pnk - h[0])
                if side == 'R':
                    n_lm = -n_lm
            n_now = quat_rotate(after_swing, PALM_LOCAL_AXIS)
            n_now_p = n_now - np.dot(n_now, f_dir) * f_dir
            if n_lm is not None and np.linalg.norm(n_now_p) > 1e-6:
                a = _unit(n_now_p)
                n_lm_p = n_lm - np.dot(n_lm, f_dir) * f_dir
                if override is not None:
                    # exact authored orientation: no branch relaxation
                    b = n_lm_p / (np.linalg.norm(n_lm_p) + 1e-12)
                    cosang = np.clip(np.dot(a, b), -1, 1)
                    ang = np.arccos(cosang)
                    if np.dot(np.cross(a, b), f_dir) < 0:
                        ang = -ang
                    twist = quat_from_axis_angle(f_dir, ang)
                else:
                    cand = []
                    for sgn in (+1.0, -1.0):
                        b = sgn * n_lm_p / (np.linalg.norm(n_lm_p) + 1e-12)
                        cosang = np.clip(np.dot(a, b), -1, 1)
                        ang = np.arccos(cosang)
                        if np.dot(np.cross(a, b), f_dir) < 0:
                            ang = -ang
                        cand.append(ang)
                    ang = cand[0] if abs(cand[0]) <= abs(cand[1]) else cand[1]
                    twist = quat_from_axis_angle(f_dir, ang)
            else:
                twist = np.array([0, 0, 0, 1.0])
            world_wrist = quat_mul(twist, after_swing)
            local[f'wrist.{side}'] = quat_normalize(quat_mul(quat_conj(world_lower02), world_wrist))

            # --- fingers ---
            parent_world = world_wrist  # fingers are children of wrist
            for finger, bones in FINGER_BONES.items():
                lms = FINGER_LM[finger]
                chain_dirs = []
                for k in range(3):
                    a_lm = lms[k]; b_lm = lms[k + 1]
                    chain_dirs.append(h[b_lm] - h[a_lm])
                # metacarpal (non-thumb): damped swing tracked through the carried parent
                if finger != 'thumb':
                    mc_bone = METACARPAL_BONES[finger]
                    mc_name = f'{mc_bone}.{side}'
                    rest_mc = rig.rest_world_q[rig.by_name[mc_name]]
                    rel_mc = quat_mul(quat_conj(rig.rest_world_q[rig.by_name[f'wrist.{side}']]), rest_mc)
                    carried = quat_mul(parent_world, rel_mc)
                    carried_dir = quat_rotate(carried, np.array([0, 1, 0]))
                    mc_dir_lm = _unit(h[METACARPAL_LM[finger]] - h[0])
                    # delta from bind: identity at bind pose, corrects rig axes mismatch
                    swing_now = _swing_between(carried_dir, mc_dir_lm)
                    swing_bind = _swing_between(rig.rest_world_dir(mc_name),
                                                self.mc_bind_dir[(finger, side)])
                    swing_mc = quat_mul(swing_now, quat_conj(swing_bind))
                    # damp to 60%: palm bones barely move in ISL
                    damped = quat_slerp(np.array([0, 0, 0, 1.0]), swing_mc, 0.6)
                    world_mc = quat_mul(damped, carried)
                    local[mc_name] = quat_normalize(quat_mul(quat_conj(parent_world), world_mc))
                    pw = world_mc
                    rest_pw = rig.rest_world_q[rig.by_name[mc_name]]
                else:
                    pw = parent_world
                    rest_pw = rig.rest_world_q[rig.by_name[f'wrist.{side}']]
                for k, bone in enumerate(bones):
                    name = f'{bone}.{side}'
                    rest_q = rig.rest_world_q[rig.by_name[name]]
                    rel = quat_mul(quat_conj(rest_pw), rest_q)
                    carried = quat_mul(pw, rel)
                    carried_dir = quat_rotate(carried, np.array([0, 1, 0]))
                    target_dir = _unit(chain_dirs[k])
                    swing = _swing_between(carried_dir, target_dir)
                    world_bone = quat_mul(swing, carried)
                    local[name] = quat_normalize(quat_mul(quat_conj(pw), world_bone))
                    pw = world_bone
                    rest_pw = rest_q
        return local

    def local_to_world(self, local):
        return self.rig.world_positions(local)

def smooth_quats(qseq, window=5):
    """Hemisphere-continuity fix + simple centered moving average (per component), renormalized."""
    qs = [q.copy() for q in qseq]
    for i in range(1, len(qs)):
        if np.dot(qs[i - 1], qs[i]) < 0:
            qs[i] = -qs[i]
    out = []
    half = window // 2
    for i in range(len(qs)):
        acc = np.zeros(4)
        cnt = 0
        for j in range(max(0, i - half), min(len(qs), i + half + 1)):
            acc += qs[j]
            cnt += 1
        out.append(quat_normalize(acc / cnt))
    return out

def ease_frames(frames_local, rest_local, blend_in=8, blend_out=8):
    """Prepend/append slerp-from-rest transition frames. frames_local: list of dicts."""
    bones = [b for b in frames_local[0]]
    out = []
    for k in range(blend_in):
        t = (k + 1) / (blend_in + 1)
        frame = {}
        for b in bones:
            frame[b] = quat_slerp(rest_local[b], frames_local[0][b], t)
        out.append(frame)
    out.extend(frames_local)
    last = frames_local[-1]
    for k in range(blend_out):
        t = (k + 1) / (blend_out + 1)
        frame = {}
        for b in bones:
            frame[b] = quat_slerp(last[b], rest_local[b], t)
        out.append(frame)
    return out
