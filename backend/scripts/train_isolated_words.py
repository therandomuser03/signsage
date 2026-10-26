"""Train a small INCLUDE-only isolated-word pilot from the published aggregate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
from pathlib import PurePosixPath

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.isolated_words import (
    FEATURE_DIM,
    FEATURE_VERSION,
    FRAME_COUNT,
    IsolatedWordNet,
    prepare_isolated_features,
    split_isolated_rows,
)

REVISION = "255ce7fd10da5d05bc30d59668ba256e5f02531c"
DATASET = "https://huggingface.co/datasets/vidit031/isl-isolated-40words"
ASSETS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def fetch(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)


def source_rows() -> list[dict[str, str]]:
    url = f"{DATASET}/resolve/{REVISION}/metadata.csv"
    with urllib.request.urlopen(url, timeout=30) as response:
        rows = list(csv.DictReader(response.read().decode("utf-8").splitlines()))
    rows = [
        row for row in rows
        if row["dataset"] == "INCLUDE" and row["review_status"] == "accepted"
    ]
    if not rows or any(row["license"] != "CC-BY-4.0" for row in rows):
        raise RuntimeError("Refusing dataset subset with missing or unexpected per-row license")
    expected = {
        "friend": 20, "hello": 21, "hospital": 20,
        "market": 20, "school": 20, "thank you": 21,
    }
    actual: dict[str, int] = defaultdict(int)
    for row in rows:
        actual[row["normalized_word"]] += 1
    if dict(actual) != expected:
        raise RuntimeError(f"Pinned INCLUDE subset changed: expected {expected}, received {dict(actual)}")
    return rows


def download_videos(rows: list[dict[str, str]], data_dir: Path) -> list[Path]:
    downloaded = []
    for row in rows:
        relative = PurePosixPath(row["video_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe dataset path: {row['video_path']}")
        path = data_dir / "videos" / Path(*relative.parts)
        fetch(f"{DATASET}/resolve/{REVISION}/{row['video_path']}", path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {row['video_path']}")
        downloaded.append(path)
    return downloaded


def _landmarks(points, count: int) -> np.ndarray:
    result = np.zeros((count, 3), dtype=np.float32)
    for index, point in enumerate(points[:count]):
        result[index] = (point.x, point.y, point.z)
    return result


def extract_video(path: Path, pose, hands, time_offset_ms: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / 15))
    frames = []
    frame_index = 0
    previous_timestamp_ms = 0
    while len(frames) < FRAME_COUNT:
        ok, bgr = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            previous_timestamp_ms = max(previous_timestamp_ms + 1, round(frame_index * 1000 / fps))
            timestamp_ms = time_offset_ms + previous_timestamp_ms
            pose_result = pose.detect_for_video(image, timestamp_ms)
            hand_result = hands.detect_for_video(image, timestamp_ms)
            row = np.zeros((543, 3), dtype=np.float32)
            if pose_result.pose_landmarks:
                row[:33] = _landmarks(pose_result.pose_landmarks[0], 33)
            for index, points in enumerate(hand_result.hand_landmarks or []):
                label = hand_result.handedness[index][0].category_name
                start = 501 if label == "Left" else 522
                if not row[start:start + 21].any():
                    row[start:start + 21] = _landmarks(points, 21)
            frames.append(row)
        frame_index += 1
    capture.release()
    if len(frames) < 2:
        raise RuntimeError(f"Video has fewer than 2 readable frames: {path}")
    return np.stack(frames)


def metrics(logits: torch.Tensor, targets: torch.Tensor, labels: list[str]) -> dict:
    predicted = logits.argmax(dim=1)
    matrix = torch.zeros((len(labels), len(labels)), dtype=torch.int64)
    for expected, actual in zip(targets, predicted):
        matrix[int(expected), int(actual)] += 1
    recalls = {
        label: float(matrix[index, index] / matrix[index].sum())
        for index, label in enumerate(labels)
    }
    return {
        "accuracy": float((predicted == targets).float().mean()),
        "macro_recall": sum(recalls.values()) / len(recalls),
        "per_class_recall": recalls,
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.getenv("SIGNSAGE_ISL_DATA_DIR"))
    parser.add_argument("--output", default=os.getenv("SIGNSAGE_ISL_BUNDLE_DIR"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if not args.data_dir or not args.output:
        parser.error("Set SIGNSAGE_ISL_DATA_DIR and SIGNSAGE_ISL_BUNDLE_DIR or pass both paths")
    data_dir, bundle_dir = Path(args.data_dir), Path(args.output)
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty bundle directory: {bundle_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    rows = source_rows()
    paths = download_videos(rows, data_dir)
    assets_dir = data_dir / "mediapipe"
    assets = {}
    for name, url in ASSETS.items():
        assets[name] = assets_dir / name
        fetch(url, assets[name])
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(assets["pose_landmarker_lite.task"])),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    ))
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(assets["hand_landmarker.task"])),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    ))
    try:
        raw_features = {
            row["video_path"]: prepare_isolated_features(
                extract_video(path, pose, hands, video_index * 1_000_000)
            )
            for video_index, (row, path) in enumerate(zip(rows, paths))
        }
    finally:
        pose.close()
        hands.close()

    labels = sorted({row["normalized_word"] for row in rows})
    label_ids = {label: index for index, label in enumerate(labels)}
    splits = split_isolated_rows(rows, args.seed)

    def tensors(split: str):
        samples = splits[split]
        x = torch.tensor(np.stack([raw_features[row["video_path"]] for row in samples]))
        y = torch.tensor([label_ids[row["normalized_word"]] for row in samples])
        return x, y

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x, train_y = tensors("train")
    valid_x, valid_y = tensors("validation")
    test_x, test_y = tensors("test")
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=16, shuffle=True)
    model = IsolatedWordNet(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy, best_state, patience = -1.0, None, 12
    stale_epochs = 0
    for _epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            valid_logits = model(valid_x.to(device)).cpu()
        score = metrics(valid_logits, valid_y, labels)["accuracy"]
        if score > best_accuracy:
            best_accuracy = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        test_logits = model(test_x.to(device)).cpu()
    evaluation = {
        "status": "NO-GO",
        "test": metrics(test_logits, test_y, labels),
        "validation_accuracy": best_accuracy,
        "split_counts": {name: len(values) for name, values in splits.items()},
        "signer_independent": False,
        "tests_passed": False,
        "promotion_eligible": False,
        "promotion_blocker": "No verified human-signer identities or signer-independent split.",
    }
    checkpoint = bundle_dir / "model.pt"
    torch.save({"state_dict": best_state}, checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata = {
        "format": "signsage_isolated_words_v1",
        "model_scope": "isolated_word_pilot",
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "dataset_license": "CC-BY-4.0 (INCLUDE-only accepted rows; attribution required)",
        "source_repository": "https://github.com/AI4Bharat/INCLUDE",
        "source_record": "https://zenodo.org/records/4010759",
        "labels": labels,
        "feature_dim": FEATURE_DIM,
        "frame_count": FRAME_COUNT,
        "feature_version": FEATURE_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "confidence_threshold": 0.5,
        "promotion_eligible": False,
        "split_seed": args.seed,
        "split_policy": "per-class shuffled clips; no human-signer identity in source metadata",
        "evaluation": evaluation,
    }
    split_manifest = {
        split: [
            {
                "video_path": row["video_path"],
                "label": row["normalized_word"],
                "sha256": row["sha256"],
            }
            for row in samples
        ]
        for split, samples in splits.items()
    }
    (bundle_dir / "labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    (bundle_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (bundle_dir / "evaluation_results.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"device": str(device), "labels": labels, "evaluation": evaluation}, indent=2))


if __name__ == "__main__":
    main()
