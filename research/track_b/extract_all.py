"""Step 4: Extract landmarks for ALL INCLUDE clips (17-word vocab).

Uses the Tasks-Vision extractor (extract_tasks.py) matching frontend parity:
  - mediapipe.tasks PoseLandmarker / FaceLandmarker / HandLandmarker
  - IMAGE mode, CPU delegate (frontend uses VIDEO mode + GPU delegate)
  - Output: (T,543,3) float32 in SignSage component order
  - Missing landmarks = [0,0,0] zeros
  - Resumable: skips existing .npy

Reads clips from:
  - backend/data/isl/videos/ (121 pre-existing local INCLUDE clips)
  - research/track_b/data/include_clips/ (190+ downloaded Zenodo clips)

Writes to:
  - research/track_b/data/include_landmarks/*.npy
  - research/track_b/data/extraction_stats.json (per-clip log)
"""
import os, sys, glob, time, json, csv, re
sys.path.insert(0, "/mnt/Drive-D/Codes/signsage/research/track_b")
from extract_tasks import create_extractors, to_points, extract_video

import cv2, numpy as np
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

VIDEOS_DIR = "/mnt/Drive-D/Codes/signsage/backend/data/isl/videos"
CLIPS_DIR = "/mnt/Drive-D/Codes/signsage/research/track_b/data/include_clips"
OUT_DIR = "/mnt/Drive-D/Codes/signsage/research/track_b/data/include_landmarks"
STATS_PATH = "/mnt/Drive-D/Codes/signsage/research/track_b/data/extraction_stats.json"
os.makedirs(OUT_DIR, exist_ok=True)

VOCAB = ['hello','thank you','you','friend','hospital','school','market',
         'doctor','medicine','good','bad','morning','night','today',
         'tomorrow','mother','father']
NUM = re.compile(r'^\d+\.\s*')
def norm(w):
    return re.sub(r'\s+', ' ', NUM.sub('', w or '')).strip().lower()

# Build video -> word mapping from CSV
csv_paths = [
    "/mnt/Drive-D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/train_include.csv",
    "/mnt/Drive-D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/test_include.csv",
]
video_to_word = {}
video_to_split = {}
for p in csv_paths:
    fname = os.path.basename(p)  # train_include.csv or test_include.csv
    split = "train" if fname.startswith("train_") else ("test" if fname.startswith("test_") else "?")
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = norm(row["Word"])
            if w in VOCAB:
                vid = row["Video"].replace(".MOV",".mp4").replace(".MP4",".mp4")
                video_to_word[vid] = w
                video_to_split[vid] = split

# Collect all video paths
# 1. Local videos in backend/data/isl/videos (subdirs + flat)
video_paths = []  # (path, basename, word, split)
for root, dirs, files in os.walk(VIDEOS_DIR):
    for f in files:
        if f.endswith(".mp4") and "__INCLUDE__" in f:
            word_part = f.split("__")[0].lower()
            if word_part in VOCAB:
                video_paths.append((os.path.join(root, f), f, word_part, video_to_split.get(f, "?")))

# 2. Downloaded clips in include_clips (flat MOV/mp4)
for f in os.listdir(CLIPS_DIR):
    if f.endswith((".MOV", ".mp4", ".MP4")):
        # Map via CSV video name
        vid_key = f.replace(".MOV", ".mp4").replace(".MP4", ".mp4")
        word = video_to_word.get(vid_key, "unknown")
        split = video_to_split.get(vid_key, "?")
        video_paths.append((os.path.join(CLIPS_DIR, f), f, word, split))

# Deduplicate by basename
seen = set()
unique = []
for p, b, w, s in video_paths:
    if b not in seen:
        seen.add(b)
        unique.append((p, b, w, s))

print(f"Total unique INCLUDE clips for extraction: {len(unique)}")
# Per-word counts
from collections import Counter
wc = Counter(w for _,_,w,_ in unique)
for v in VOCAB:
    print(f"  {v:14} -> {wc.get(v,0)} clips")

# Initialize extractors (reused across all clips)
pose_ex, face_ex, hands_ex = create_extractors()

start = time.time()
done = 0
skip = 0
fail = 0
stats = []
high_miss = []

for idx, (path, basename, word, split) in enumerate(sorted(unique, key=lambda x: x[1])):
    stem = os.path.splitext(basename)[0]
    out_path = os.path.join(OUT_DIR, stem + ".npy")
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
        fr = arr.shape[0]
        left_miss = float(np.mean(np.any(arr[:, 501:522] == 0, axis=(1,2)))) if fr > 0 else 1.0
        right_miss = float(np.mean(np.any(arr[:, 522:543] == 0, axis=(1,2)))) if fr > 0 else 1.0
        stats.append({"file": basename, "word": word, "split": split, "frames": fr,
                      "left_miss_frac": left_miss, "right_miss_frac": right_miss})
        if left_miss > 0.40 or right_miss > 0.40:
            high_miss.append([basename, fr, left_miss, right_miss])
        if done % 10 == 0:
            print(f"  done={done} skip={skip} fail={fail} last={basename} frames={fr} l_miss={left_miss:.2f} r_miss={right_miss:.2f}")
    except Exception as e:
        cap.release()
        fail += 1
        print(f"  FAIL {basename}: {type(e).__name__} {str(e)[:100]}")
        stats.append({"file": basename, "word": word, "split": split, "frames": 0,
                      "left_miss_frac": 1.0, "right_miss_frac": 1.0, "error": str(e)[:200]})

pose_ex.close(); face_ex.close(); hands_ex.close()
elapsed = time.time() - start

result = {
    "done": done, "skip": skip, "fail": fail, "total_clips": len(unique),
    "elapsed_sec": elapsed, "elapsed_min": elapsed / 60.0,
    "high_miss_flags": high_miss,
    "per_clip": stats,
}
with open(STATS_PATH, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n=== EXTRACTION SUMMARY ===")
print(f"done={done} skip={skip} fail={fail} total={len(unique)}")
print(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"Clips with hands missing >40% frames: {len(high_miss)}")
for item in high_miss[:10]:
    print(f"  {item[0]} frames={item[1]} left_miss={item[2]:.2f} right_miss={item[3]:.2f}")
print(f"Wrote {STATS_PATH}")