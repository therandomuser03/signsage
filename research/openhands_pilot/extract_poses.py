#!/usr/bin/env python3
"""Extract MediaPipe pose + hand landmarks from real ISL clips and cache to disk.

Saves (T, 75, 3) float32 arrays to extracted_poses/<clip_basename>.npy.
ponytail: ceiling=local disk cache; upgrade path = stream directly or integrate into video pipeline.
"""
import os, sys, glob, time, cv2
import numpy as np
from pathlib import Path
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    HandLandmarker, HandLandmarkerOptions
)

POSE_MODEL = "/mnt/D/Codes/signsage/backend/data/isl/mediapipe/pose_landmarker_lite.task"
HAND_MODEL = "/mnt/D/Codes/signsage/backend/data/isl/mediapipe/hand_landmarker.task"
OUT_DIR = os.path.join(os.path.dirname(__file__), "extracted_poses")
os.makedirs(OUT_DIR, exist_ok=True)

def find_clips():
    video_files = glob.glob("/mnt/D/Codes/signsage/backend/data/isl/videos/**/*.mp4", recursive=True)
    words = ["friend", "hello", "hospital", "market", "school", "thank_you"]
    unique_clips = {}
    for vf in video_files:
        fname = os.path.basename(vf)
        for w in words:
            if fname.startswith(w + "__INCLUDE__"):
                unique_clips[fname] = vf
    return sorted(unique_clips.items())

def extract_clip(video_path, pose_l, hand_l, get_next_ts):
    cap = cv2.VideoCapture(video_path)
    frames_75 = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        ts = get_next_ts()
        f75 = np.zeros((75, 3), dtype=np.float32)
        p_res = pose_l.detect_for_video(mp_img, ts)
        if p_res.pose_landmarks:
            for i, lm in enumerate(p_res.pose_landmarks[0]):
                if i < 33: f75[i] = [lm.x, lm.y, lm.z]
            f75[33] = f75[15] # fallback left wrist
            f75[54] = f75[16] # fallback right wrist
        h_res = hand_l.detect_for_video(mp_img, ts)
        if h_res.hand_landmarks:
            for idx, h_lms in enumerate(h_res.hand_landmarks):
                label = "Left"
                if h_res.handedness and idx < len(h_res.handedness) and len(h_res.handedness[idx]) > 0:
                    label = h_res.handedness[idx][0].category_name
                offset = 33 if label == "Left" else 54
                for i, lm in enumerate(h_lms):
                    if i < 21: f75[offset + i] = [lm.x, lm.y, lm.z]
        frames_75.append(f75)
    cap.release()
    return np.array(frames_75, dtype=np.float32)

def main():
    clips = find_clips()
    print(f"Total clips to extract: {len(clips)}")
    
    pose_l = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=RunningMode.VIDEO, num_poses=1))
    hand_l = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.1, min_hand_presence_confidence=0.1, min_tracking_confidence=0.1))

    current_ts = [0]
    def get_next_ts():
        current_ts[0] += 33
        return current_ts[0]

    t_start = time.perf_counter()
    extracted = 0
    for idx, (fname, vpath) in enumerate(clips, 1):
        out_path = os.path.join(OUT_DIR, fname.replace(".mp4", ".npy"))
        if os.path.exists(out_path):
            continue
        t0 = time.perf_counter()
        arr = extract_clip(vpath, pose_l, hand_l, get_next_ts)
        np.save(out_path, arr)
        extracted += 1
        dur = time.perf_counter() - t0
        print(f"[{idx}/{len(clips)}] Extracted {fname}: {arr.shape} in {dur:.2f}s", flush=True)

    pose_l.close()
    hand_l.close()
    total_time = time.perf_counter() - t_start
    print(f"Done! Extracted {extracted} clips in {total_time:.1f}s.")

if __name__ == "__main__":
    main()
