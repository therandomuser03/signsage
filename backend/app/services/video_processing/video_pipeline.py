"""Video-to-landmarks pipeline for isolated-word recognition.

Reads an MP4 file, decodes frames with OpenCV, extracts MediaPipe
pose and hand landmarks, and builds the canonical (T, 543, 3) representation
expected by IsolatedWordAdapter.
"""
from __future__ import annotations

import tempfile
import os
from pathlib import Path

import cv2
import numpy as np

from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
    HandLandmarker,
    HandLandmarkerOptions,
)


# Maximum frames to process from a video (matches adapter contract).
MAX_FRAMES = 64


def process_video_file(video_path: str | Path, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Process an MP4/video file and return (T, 543, 3) landmark array."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    pose_options = PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "isl" / "mediapipe" / "pose_landmarker_lite.task")
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    hand_options = HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "isl" / "mediapipe" / "hand_landmarker.task")
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )

    pose_landmarker = PoseLandmarker.create_from_options(pose_options)
    hand_landmarker = HandLandmarker.create_from_options(hand_options)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / 15))
    frame_index = 0
    previous_timestamp_ms = 0
    frames_543: list[np.ndarray] = []
    try:
        while cap.isOpened() and len(frames_543) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % stride == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

                previous_timestamp_ms = max(previous_timestamp_ms + 1, round(frame_index * 1000 / fps))
                timestamp_ms = previous_timestamp_ms

                frame_543 = np.zeros((543, 3), dtype=np.float32)

                # Pose — VIDEO mode for temporal continuity (matches training)
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                if pose_result.pose_landmarks:
                    pose_points = np.array(
                        [[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]],
                        dtype=np.float32,
                    )
                    n_points = min(len(pose_points), 33)
                    frame_543[0:n_points, :] = pose_points[:n_points, :]

                # Hands — VIDEO mode, label-based assignment, no overwrite
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                if hand_result.hand_landmarks:
                    for index, hand_landmarks in enumerate(hand_result.hand_landmarks or []):
                        hand_points = np.array(
                            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                            dtype=np.float32,
                        )
                        label = "Left"
                        if (hand_result.handedness and
                            index < len(hand_result.handedness) and
                            len(hand_result.handedness[index]) > 0):
                            label = hand_result.handedness[index][0].category_name
                        start_idx = 501 if label == "Left" else 522
                        if not frame_543[start_idx:start_idx + 21].any():
                            n_points = min(len(hand_points), 21)
                            frame_543[start_idx:start_idx + n_points, :] = hand_points[:n_points, :]

                frames_543.append(frame_543)
            frame_index += 1
    finally:
        cap.release()
        pose_landmarker.close()
        hand_landmarker.close()

    if len(frames_543) == 0:
        raise ValueError("No frames could be decoded from the video.")
    if len(frames_543) < 2:
        raise ValueError(f"Insufficient frames for inference: got {len(frames_543)}, need at least 2.")

    return np.stack(frames_543, axis=0)
