"""Fast batch extraction with reused Tasks-vision extractors (Step 4)."""
import os, glob, time, sys
sys.path.insert(0, "/mnt/D/Codes/signsage/research/track_b")
from extract_tasks import create_extractors, to_points
import cv2, numpy as np

ASSETS_DIR = "/mnt/D/Codes/signsage/research/track_b/data/include_clips"
OUT_DIR = "/mnt/D/Codes/signsage/research/track_b/data/include_landmarks"
os.makedirs(OUT_DIR, exist_ok=True)

# Find all local INCLUDE videos across flat + subdirs
videos = []
for root, dirs, files in os.walk("/mnt/D/Codes/signsage/backend/data/isl/videos"):
    for f in files:
        if f.endswith(".mp4") and "__INCLUDE__" in f:
            videos.append(os.path.join(root, f))

# Deduplicate by basename
seen = set()
unique_videos = []
for v in videos:
    b = os.path.basename(v)
    if b not in seen:
        seen.add(b)
        unique_videos.append(v)
print(f"Batch extracting {len(unique_videos)} unique INCLUDE clips to {OUT_DIR}")

pose_ex, face_ex, hands_ex = create_extractors()
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

start = time.time()
done = 0
skip = 0
fail = 0
missing_stats = []

for idx, path in enumerate(sorted(unique_videos)):
    basename = os.path.basename(path).replace(".mp4", "")
    out_path = os.path.join(OUT_DIR, basename + ".npy")
    if os.path.exists(out_path):
        skip += 1
        continue
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                break
            rgb_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
            mp_img = Image(image_format=ImageFormat.SRGB, data=rgb_u8)
            pose_res = pose_ex.detect(mp_img)
            face_res = face_ex.detect(mp_img)
            hand_res = hands_ex.detect(mp_img)
            pose_r = pose_res.landmarks if hasattr(pose_res, 'landmarks') else (pose_res.pose_landmarks if hasattr(pose_res, 'pose_landmarks') else [])
            pose = pose_r[0] if pose_r else []
            face_r = face_res.face_landmarks if hasattr(face_res, 'face_landmarks') else (face_res.landmarks if hasattr(face_res, 'landmarks') else [])
            face = face_r[0] if face_r else []
            hands = (hand_res.hand_landmarks if hasattr(hand_res, 'hand_landmarks') else (hand_res.landmarks if hasattr(hand_res, 'landmarks') else []))
            handed = (hand_res.handedness if hasattr(hand_res, 'handedness') else [])
            left_hand = []; right_hand = []
            for marks, h in zip(hands, handed):
                label = h[0].category_name if h and h[0] else None
                if label == "Left" and not left_hand: left_hand = marks
                elif label == "Right" and not right_hand: right_hand = marks
                elif not left_hand: left_hand = marks
                elif not right_hand: right_hand = marks
            pose_pts = to_points(pose, 33)
            face_pts = to_points(face, 468)
            lh_pts = to_points(left_hand, 21)
            rh_pts = to_points(right_hand, 21)
            full = np.vstack([pose_pts, face_pts, lh_pts, rh_pts]).astype(np.float32)
            frames.append(full)
        cap.release()
        arr = np.stack(frames) if frames else np.zeros((0, 543, 3), dtype=np.float32)
        np.save(out_path, arr)
        done += 1
        if done % 10 == 0:
            print(f"  done={done} skip={skip} fail={fail} last={basename} frames={arr.shape[0]}")
        # Per-clip missing stats (simplified)
        if arr.shape[0] > 0:
            left_miss = float(np.mean(np.any(arr[:, 501:522] == 0, axis=(1,2))))
            right_miss = float(np.mean(np.any(arr[:, 522:543] == 0, axis=(1,2))))
            missing_stats.append((basename, arr.shape[0], left_miss, right_miss))
    except Exception as e:
        cap.release()
        fail += 1
        print(f"  FAIL {basename}: {type(e).__name__} {str(e)[:100]}")

pose_ex.close(); face_ex.close(); hands_ex.close()
elapsed = time.time() - start
print(f"\n=== BATCH SUMMARY === done={done} skip={skip} fail={fail} elapsed={elapsed:.1f}s ({elapsed/60:.1f}min)")
# Report missing-hand stats: flag >0.40
high_miss = [(b, fr, lm, rm) for b, fr, lm, rm in missing_stats if lm > 0.40 or rm > 0.40]
print(f"Clips with hands missing >40% frames: {len(high_miss)}")
for item in high_miss[:5]:
    print(f"  {item[0]} frames={item[1]} left_miss={item[2]:.2f} right_miss={item[3]:.2f}")
# Save stats
import json
with open("/mnt/D/Codes/signsage/research/track_b/data/extraction_stats.json", "w") as f:
    json.dump({"done": done, "skip": skip, "fail": fail, "elapsed_sec": elapsed,
               "high_miss_flags": [[b,fr,lm,rm] for b,fr,lm,rm in high_miss]}, f)
print("Wrote extraction_stats.json")
