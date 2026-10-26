"""Run genuine held-out video inference with a trained isolated-word bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app.ml.isolated_words import IsolatedWordAdapter
from train_isolated_words import ASSETS, extract_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--save-request")
    parser.add_argument("--all-test", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    data_dir, bundle = Path(args.data_dir), Path(args.bundle)
    split = json.loads((bundle / "split_manifest.json").read_text(encoding="utf-8"))
    if not 0 <= args.sample_index < len(split["test"]):
        parser.error(f"--sample-index must be between 0 and {len(split['test']) - 1}")
    assets_dir = data_dir / "mediapipe"
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(assets_dir / "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    ))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(assets_dir / "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    ))
    started = time.perf_counter()
    adapter = IsolatedWordAdapter.load(bundle, device="cpu")
    load_seconds = time.perf_counter() - started
    records = []
    try:
        samples = split["test"] if args.all_test else [split["test"][args.sample_index]]
        for sample_index, sample in enumerate(samples):
            video = data_dir / "videos" / sample["video_path"]
            if hashlib.sha256(video.read_bytes()).hexdigest() != sample["sha256"]:
                raise RuntimeError(f"Held-out sample checksum mismatch: {video}")
            started = time.perf_counter()
            landmarks = extract_video(video, pose, hands, sample_index * 1_000_000)
            extraction_seconds = time.perf_counter() - started
            started = time.perf_counter()
            result = adapter.transcribe(landmarks)
            inference_seconds = time.perf_counter() - started
            records.append({
                "video_path": sample["video_path"],
                "expected_label": sample["label"],
                "text": result.text,
                "accepted": result.accepted,
                "reason": result.reason,
                "uncalibrated_softmax_score": result.confidence,
                "correct": result.accepted and result.text == sample["label"],
                "video_landmark_extraction_seconds": extraction_seconds,
                "inference_seconds": inference_seconds,
                "frame_shape": list(landmarks.shape),
            })
            selected_for_request = sample["video_path"] == split["test"][args.sample_index]["video_path"]
            if selected_for_request and args.save_request:
                payload = {
                    "frames": [{"landmarks": frame.tolist()} for frame in landmarks],
                    "save_history": False,
                }
                Path(args.save_request).write_text(json.dumps(payload), encoding="utf-8")
    finally:
        pose.close()
        hands.close()
    print(json.dumps({
        "dataset": "vidit031/isl-isolated-40words",
        "revision": "255ce7fd10da5d05bc30d59668ba256e5f02531c",
        "source": "INCLUDE",
        "license": "CC-BY-4.0",
        "model_load_seconds": load_seconds,
        "device": str(adapter.device),
        "test_sample_count": len(records),
        "accepted_count": sum(record["accepted"] for record in records),
        "accepted_correct_count": sum(record["correct"] for record in records),
        "samples": records,
    }, indent=2))


if __name__ == "__main__":
    main()
