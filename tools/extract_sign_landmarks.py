#!/usr/bin/env python3
"""Extract MediaPipe Pose(33) + Hands(2x21) landmarks from isolated-sign videos.

Same layout as research/openhands_pilot/extract_poses.py: (T, 75, 3) float32,
image-normalized x,y in [0,1], z = MediaPipe relative depth.
  [0:33]   pose landmarks (MediaPipe Pose order)
  [33:54]  left-hand landmarks (MediaPipe Hands order)
  [54:75]  right-hand landmarks
Handedness labels are kept as reported by MediaPipe for THIS unmirrored setup
the CLI flag --mirror-x can flip x if the source video is mirrored.
Missing detections stay zero; per-frame hand presence = any nonzero.

Usage: extract_sign_landmarks.py OUT_DIR name=path.mp4 [name=path.mp4 ...]
"""
import os, sys, time
import numpy as np
import cv2
from pathlib import Path
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    HandLandmarker, HandLandmarkerOptions,
)

REPO = Path(__file__).resolve().parent.parent
POSE_MODEL = REPO / "backend/data/isl/mediapipe/pose_landmarker_lite.task"
HAND_MODEL = REPO / "backend/data/isl/mediapipe/hand_landmarker.task"

def main():
    out_dir = Path(sys.argv[1])
    pairs = [a.split("=", 1) for a in sys.argv[2:]]
    out_dir.mkdir(parents=True, exist_ok=True)

    pose_l = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO, num_poses=1))
    hand_l = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3))

    ts_global = [0]
    def next_ts(fps):
        ts_global[0] += int(1000 / fps)
        return ts_global[0]

    for name, vpath in pairs:
        out_path = out_dir / f"{name}.npy"
        if out_path.exists():
            print(f"skip {out_path} (exists)")
            continue
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts = next_ts(fps)
            f75 = np.zeros((75, 3), dtype=np.float32)
            p = pose_l.detect_for_video(mp_img, ts)
            if p.pose_landmarks:
                for i, lm in enumerate(p.pose_landmarks[0]):
                    if i < 33:
                        f75[i] = [lm.x, lm.y, lm.z]
            h = hand_l.detect_for_video(mp_img, ts)
            if h.hand_landmarks:
                for idx, lms in enumerate(h.hand_landmarks):
                    label = "Left"
                    if h.handedness and idx < len(h.handedness) and h.handedness[idx]:
                        label = h.handedness[idx][0].category_name
                    off = 33 if label == "Left" else 54
                    for i, lm in enumerate(lms):
                        if i < 21:
                            f75[off + i] = [lm.x, lm.y, lm.z]
            frames.append(f75)
        cap.release()
        arr = np.array(frames, dtype=np.float32)
        np.save(out_path, arr)
        lp = np.any(arr[:, 33:54] != 0, axis=(1, 2)).mean() if len(arr) else 0
        rp = np.any(arr[:, 54:75] != 0, axis=(1, 2)).mean() if len(arr) else 0
        print(f"{name}: {arr.shape} L-hand {lp:.0%} R-hand {rp:.0%}")

    pose_l.close()
    hand_l.close()

if __name__ == "__main__":
    main()
