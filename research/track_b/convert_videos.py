"""Convert ISL reference sign videos (videos/*.mp4) into landmark clips in the
same (T,543,3) format as the other converted sessions.

Uses the SAME extractor as the SignSage frontend / recorder (vendored
@mediapipe/tasks-vision pose/face/hand landmarker models, VIDEO mode), via
the Python tasks API:

  pose_landmarker_lite.task / face_landmarker.task / hand_landmarker.task
  from research/track_b/recorder/assets/models/

Layout (matches frontend/backend contract):
  0-32   pose (33)
  33-500 face (468)
  501-521 left hand (21)
  522-542 right hand (21)
Missing parts are stored as all-zero landmarks (same convention as the app).
Coordinates are NOT mirrored; handedness follows the image ("Left" = appears
on the left), exactly like the app/recorder.

Purpose: the videos depict the correct ISL sign for each of the 9 vocabulary
words (hello, me, you, yes, no, help, water, where + rest). They are external
reference material (mixed sources/signers/quality), NOT the app's own captures.

Output (gitignored):
  research/track_b/data/s4_converted/<label>__S4__take01__<ts>.npy
  research/track_b/data/s4_converted/<label>__S4__take01__<ts>.json
  research/track_b/data/s4_converted/manifest.csv

This script only reads videos/ and writes into research/track_b/data/s4_converted/.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

REPO_ROOT = Path(__file__).resolve().parents[2]  # research/track_b -> repo root
TRACK_B = Path(__file__).resolve().parent
MODELS_DIR = TRACK_B / "recorder" / "assets" / "models"
OUT_DIR = TRACK_B / "data" / "s4_converted"
VIDEOS_DIR = REPO_ROOT / "videos"

LABELS = ["hello", "me", "you", "yes", "no", "help", "water", "where", "rest"]
POSE_LM, FACE_LM, HAND_LM = 33, 468, 21


def zero_fill(marks, count: int) -> list[list[float]]:
    out = []
    for i in range(count):
        m = marks[i] if i < len(marks) else None
        out.append([float(m.x), float(m.y), float(m.z)] if m else [0.0, 0.0, 0.0])
    return out


def landmarks_for_frame(pose_landmarker, face_landmarker, hands_landmarker,
                        mp_image, ts_ms: int) -> tuple[np.ndarray, bool, bool, bool]:
    pose_res = pose_landmarker.detect_for_video(mp_image, ts_ms)
    face_res = face_landmarker.detect_for_video(mp_image, ts_ms)
    hand_res = hands_landmarker.detect_for_video(mp_image, ts_ms)

    pose = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else []
    face = face_res.face_landmarks[0] if face_res.face_landmarks else []

    # Handedness labels follow the image; fill left then right, like the app.
    hand_lms = hand_res.hand_landmarks or []
    handed = hand_res.handedness or []
    left_marks = right_marks = []
    for i, marks in enumerate(hand_lms):
        label = handed[i][0].category_name if handed and i < len(handed) else None
        if label == "Left" and not left_marks:
            left_marks = marks
        elif label == "Right" and not right_marks:
            right_marks = marks
        elif not left_marks:
            left_marks = marks
        elif not right_marks:
            right_marks = marks

    pts = (
        zero_fill(pose, POSE_LM)
        + zero_fill(face, FACE_LM)
        + zero_fill(left_marks, HAND_LM)
        + zero_fill(right_marks, HAND_LM)
    )
    arr = np.array(pts, dtype=np.float32)
    hand_ok = bool(left_marks) or bool(right_marks)
    return arr, bool(pose), bool(face), hand_ok


def main() -> int:
    videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files in {VIDEOS_DIR}")
        return 1

    for name in ("pose_landmarker_lite.task", "face_landmarker.task", "hand_landmarker.task"):
        if not (MODELS_DIR / name).is_file():
            print(f"Missing model asset: {MODELS_DIR / name}")
            return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def make_landmarkers():
        pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODELS_DIR / "pose_landmarker_lite.task")),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
            )
        )
        face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODELS_DIR / "face_landmarker.task")),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            )
        )
        hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODELS_DIR / "hand_landmarker.task")),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )
        return pose, face, hands

    rows = []
    for video in videos:
        label = video.stem
        if label not in LABELS:
            print(f"  (skip {video.name}: not in vocabulary {LABELS})")
            continue

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"  (skip {video.name}: cannot open)")
            continue

        # Fresh landmarkers per video so each VIDEO-mode timestamp starts at 0.
        pose, face, hands = make_landmarkers()

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frames = []
        n_pose = n_face = n_hand = n_left = n_right = 0
        ts_ms = 0
        step_ms = max(1, round(1000.0 / fps))
        ok_read = True
        while ok_read:
            ok_read, frame = cap.read()
            if not ok_read:
                break
            ts_ms += step_ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            arr, pose_ok, face_ok, hand_ok = landmarks_for_frame(
                pose, face, hands, mp_image, ts_ms)
            frames.append(arr)
            n_pose += pose_ok
            n_face += face_ok
            n_hand += hand_ok
            n_left += bool(np.any(arr[501:522] != 0.0))
            n_right += bool(np.any(arr[522:543] != 0.0))
        cap.release()
        pose.close(); face.close(); hands.close()

        if len(frames) < 2:
            print(f"  (skip {video.name}: only {len(frames)} frames decoded)")
            continue

        arr = np.stack(frames).astype(np.float32)
        mtime = datetime.fromtimestamp(video.stat().st_mtime)
        stamp = mtime.strftime("%Y%m%d%H%M%S")
        out_npy = OUT_DIR / f"{label}__S4__take01__{stamp}.npy"
        out_json = OUT_DIR / f"{label}__S4__take01__{stamp}.json"
        np.save(out_npy, arr)

        extractor_meta = {
            "name": "@mediapipe/tasks-vision PoseLandmarker + FaceLandmarker + HandLandmarker",
            "version": mp.__version__,
            "running_mode": "VIDEO",
            "note": (
                "External ISL reference video converted with the same tasks-vision "
                "landmarker models as the app/recorder (Python tasks API, CPU delegate). "
                "Different signer(s)/sources; coordinates NOT mirrored; handedness follows the image."
            ),
            "options": {
                "pose": {"numPoses": 1},
                "face": {"numFaces": 1},
                "hands": {"numHands": 2},
            },
            "models": ["pose_landmarker_lite.task", "face_landmarker.task", "hand_landmarker.task"],
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"extractor": extractor_meta, "source_video": video.name}, f, indent=2)

        n = len(frames)
        summary = {
            "json": str(out_json),
            "npy": str(out_npy),
            "label": label,
            "session": "S4",
            "take": 1,
            "timestamp": stamp,
            "effective_fps": round(float(fps), 1),
            "frames": n,
            "frames_reported": total,
            "left_miss_frac": round(1.0 - n_left / n, 6),
            "right_miss_frac": round(1.0 - n_right / n, 6),
            "left_miss_frac_reported": round(1.0 - n_left / n, 6),
            "right_miss_frac_reported": round(1.0 - n_right / n, 6),
            "shape": f"({n}, 543, 3)",
        }
        rows.append(summary)
        print(
            f"{label:6} T={n:3} fps={fps:.1f} pose={n_pose / n:.2f} face={n_face / n:.2f} "
            f"hand_any={n_hand / n:.2f} left_det={n_left / n:.2f} right_det={n_right / n:.2f}"
        )

    manifest = OUT_DIR / "manifest.csv"
    fieldnames = ["json", "npy", "label", "session", "take", "timestamp",
                  "effective_fps", "frames", "frames_reported",
                  "left_miss_frac", "right_miss_frac",
                  "left_miss_frac_reported", "right_miss_frac_reported", "shape"]
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {manifest} ({len(rows)} clips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
