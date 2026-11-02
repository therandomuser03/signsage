#!/usr/bin/env python3
"""Predict top-5 ISL word labels for a given .mp4 clip using OpenHands INCLUDE checkpoints.

Usage:
  python predict_clip.py path/to/clip.mp4 [--model stgcn|lstm|bert|slgcn] [--top_k 5]

ponytail: ceiling=no batching, single clip only; upgrade path = add batch loader for eval set.
"""
import sys, os, time, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import omegaconf
import cv2

# Add openhands_repo to path
REPO_DIR = os.path.join(os.path.dirname(__file__), "openhands_repo")
sys.path.insert(0, REPO_DIR)

from openhands.models.loader import get_model
from openhands.datasets.pose_transforms import PoseSelect, CenterAndScaleNormalize, PoseUniformSubsampling
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    HandLandmarker, HandLandmarkerOptions
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_MAP = {
    "stgcn": os.path.join(BASE_DIR, "checkpoints/include_stgcn/include/st_gcn/epoch=215-step=23543.ckpt"),
    "lstm":  os.path.join(BASE_DIR, "checkpoints/include_lstm/include/lstm/epoch=294-step=63719.ckpt"),
    "bert":  os.path.join(BASE_DIR, "checkpoints/include_bert/include/bert/epoch=328-step=18094.ckpt"),
    "slgcn": os.path.join(BASE_DIR, "checkpoints/include_slgcn/include/sl_gcn/epoch=112-step=12203.ckpt"),
}

CONFIG_MAP = {
    "stgcn": os.path.join(BASE_DIR, "checkpoints/include_stgcn/include/st_gcn/config.yaml"),
    "lstm":  os.path.join(BASE_DIR, "checkpoints/include_lstm/include/lstm/config.yaml"),
    "bert":  os.path.join(BASE_DIR, "checkpoints/include_bert/include/bert/config.yaml"),
    "slgcn": os.path.join(BASE_DIR, "checkpoints/include_slgcn/include/sl_gcn/config.yaml"),
}

SPLIT_CSV = os.path.join(BASE_DIR, "checkpoints/include_metadata/Train_Test_Split/train_include.csv")
POSE_TASK = "/mnt/D/Codes/signsage/backend/data/isl/mediapipe/pose_landmarker_lite.task"
HAND_TASK = "/mnt/D/Codes/signsage/backend/data/isl/mediapipe/hand_landmarker.task"

def extract_pose_from_video(video_path: str) -> np.ndarray:
    """Extract (T, 75, 3) float32 landmarks using MediaPipe tasks."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pose_l = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_TASK),
        running_mode=RunningMode.VIDEO, num_poses=1))
    hand_l = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_TASK),
        running_mode=RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.1, min_hand_presence_confidence=0.1, min_tracking_confidence=0.1))

    frames_75 = []
    frame_idx = 0
    prev_ts = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        ts = max(prev_ts + 1, int(frame_idx * 1000 / fps))
        prev_ts = ts
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
        frame_idx += 1

    cap.release()
    pose_l.close()
    hand_l.close()
    return np.array(frames_75, dtype=np.float32)

def predict(video_path: str, model_name: str = "stgcn", top_k: int = 5):
    t_start = time.perf_counter()
    raw_kps = extract_pose_from_video(video_path)
    t_pose = time.perf_counter()

    # Preprocess
    kps2d = raw_kps[:, :, :2].astype(np.float32)
    data = {"frames": torch.tensor(kps2d).permute(2, 0, 1)} # (2, T, 75)
    
    pose_select = PoseSelect(preset="mediapipe_holistic_minimal_27")
    center_scale = CenterAndScaleNormalize(reference_points_preset="shoulder_mediapipe_holistic_minimal_27")
    data = pose_select(data)
    if model_name == "bert":
        uniform_sub = PoseUniformSubsampling(num_frames=120)
        data = uniform_sub(data)
    data = center_scale(data)

    inp = data["frames"].unsqueeze(0) # (1, 2, T, 27)

    # Load model
    cfg = omegaconf.OmegaConf.load(CONFIG_MAP[model_name])
    model = get_model(cfg.model, in_channels=2, num_class=263)
    ckpt = torch.load(CHECKPOINT_MAP[model_name], map_location="cpu", weights_only=False)
    state = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.eval()

    t_infer_start = time.perf_counter()
    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=-1)[0]
        topk_v, topk_i = probs.topk(top_k)
    t_end = time.perf_counter()

    df = pd.read_csv(SPLIT_CSV)
    glosses = sorted(df["Word"].unique())

    results = [(glosses[idx], float(val)) for val, idx in zip(topk_v.tolist(), topk_i.tolist())]
    timings = {
        "pose_extract_sec": t_pose - t_start,
        "model_infer_sec": t_end - t_infer_start,
        "total_sec": t_end - t_start,
    }
    return results, timings

def main():
    p = argparse.ArgumentParser(description="Predict top-5 ISL word labels for an .mp4 clip")
    p.add_argument("video", help="Path to .mp4 video file")
    p.add_argument("--model", choices=list(CHECKPOINT_MAP), default="stgcn", help="Model architecture")
    p.add_argument("--top_k", type=int, default=5, help="Number of predictions to display")
    args = p.parse_args()

    print(f"Loading video: {args.video}")
    print(f"Model: {args.model.upper()}")
    results, timings = predict(args.video, args.model, args.top_k)

    print(f"\nPredictions:")
    for rank, (label, prob) in enumerate(results, 1):
        print(f"  {rank}. {label:30s}  (confidence: {prob*100:5.2f}%)")

    print(f"\nTimings:")
    print(f"  Pose extraction: {timings['pose_extract_sec']:.2f}s")
    print(f"  Model inference: {timings['model_infer_sec']*1000:.1f}ms")
    print(f"  Total latency:   {timings['total_sec']:.2f}s")

if __name__ == "__main__":
    main()
