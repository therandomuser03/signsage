"""Step 1 extractor + Step 4 extraction: MediaPipe Tasks Vision (frontend parity).

Reads video frames sequentially, detects with PoseLandmarker / FaceLandmarker /
HandLandmarker (local model assets in /mnt/D/track_b_venv/assets/), outputs (T,543,3)
array in SignSage component order with missing = [0,0,0].
Resumable: skips existing .npy in output dir.
"""
import os, glob, time, cv2
import numpy as np

# Use the installed venv
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions

ASSETS = "/mnt/D/track_b_venv/assets"

def create_extractors():
    # Tasks-vision parity: IMAGE mode (offline proxy for frontend VIDEO);
    # temporal smoothing from VIDEO mode not reproduced — documented.
    pose = PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(ASSETS, "pose_landmarker_lite.task"),
                                    delegate=BaseOptions.Delegate.CPU),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5))
    face = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(ASSETS,"face_landmarker.task"),
                                    delegate=BaseOptions.Delegate.CPU),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1))
    hands = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(ASSETS,"hand_landmarker.task"),
                                    delegate=BaseOptions.Delegate.CPU),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=2))
    return pose, face, hands


def to_points(marks, count):
    out = []
    for i in range(count):
        m = marks[i] if marks and i < len(marks) else None
        out.append([m.x, m.y, m.z] if m else [0.0, 0.0, 0.0])
    return np.array(out, dtype=np.float32)


def extract_video(video_path, out_path):
    if os.path.exists(out_path):
        return None, "skipped (exists)"
    pose_ex, face_ex, hands_ex = create_extractors()
    cap = cv2.VideoCapture(video_path)
    frames = []
    ts = 0.0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Tasks API detect() accepts np.uint8 RGB array directly in 1.2.0
        rgb_u8 = rgb.astype(np.uint8)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb_u8)
        pose_res = pose_ex.detect(mp_img)
        face_res = face_ex.detect(mp_img)
        hand_res = hands_ex.detect(mp_img)

        pose_r = pose_res.landmarks if hasattr(pose_res, 'landmarks') else (pose_res.pose_landmarks if hasattr(pose_res, 'pose_landmarks') else [])
        pose = pose_r[0] if pose_r else []
        face_r = face_res.face_landmarks if hasattr(face_res, 'face_landmarks') else (face_res.landmarks if hasattr(face_res, 'landmarks') else [])
        face = face_r[0] if face_r else []
        hands = (hand_res.landmarks if hasattr(hand_res, 'landmarks') else [])
        handed = (hand_res.handedness if hasattr(hand_res, 'handedness') else [])
        left_hand = []
        right_hand = []
        for marks, h in zip(hands, handed):
            label = h[0].category_name if h and h[0] else None
            if label == "Left" and not left_hand:
                left_hand = marks
            elif label == "Right" and not right_hand:
                right_hand = marks
            elif not left_hand:
                left_hand = marks
            elif not right_hand:
                right_hand = marks

        pose_pts = to_points(pose, 33)
        face_pts = to_points(face, 468)
        lh_pts = to_points(left_hand, 21)
        rh_pts = to_points(right_hand, 21)
        full = np.vstack([pose_pts, face_pts, lh_pts, rh_pts])  # (543,3)
        frames.append(full)
    cap.release()
    pose_ex.close(); face_ex.close(); hands_ex.close()
    arr = np.stack(frames)  # (T,543,3)
    np.save(out_path, arr.astype(np.float32))
    # stats on stacked array
    fr = arr.shape[0]
    left_frame_missing = np.mean(np.any(arr[:, 501:522] == 0, axis=(1,2)))
    right_frame_missing = np.mean(np.any(arr[:, 522:543] == 0, axis=(1,2)))
    return arr, f"frames={fr} left_miss_frac={left_frame_missing:.2f} right_miss_frac={right_frame_missing:.2f}"

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 1:
        video_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/D/Codes/signsage/backend/data/isl/videos/friend__INCLUDE__00546__MVI_3826.mp4"
        out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/test_extract.npy"
        arr, info = extract_video(video_path, out_path)
        if arr is not None:
            print("EXTRACTED", out_path, "shape", arr.shape, "dtype", arr.dtype, info)
        else:
            print("SKIPPED", info)
