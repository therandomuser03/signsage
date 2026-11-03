import csv, os, glob
from collections import Counter

# Map downloaded basename back to word via CSV video filenames
csv_paths = [
    "/mnt/D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/train_include.csv",
    "/mnt/D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/test_include.csv",
]
video_to_word = {}
NUM = __import__("re").compile(r"^\d+\.\s*")
for p in csv_paths:
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = __import__("re").sub(r"\s+", " ", NUM.sub("", row["Word"] or "")).strip().lower()
            vid = row["Video"]  # e.g., MVI_0029.MOV
            video_to_word[vid.replace(".MOV",".mp4").replace(".MP4",".mp4")] = w

downloaded = glob.glob("/mnt/D/Codes/signsage/research/track_b/data/include_clips/*.MOV")
downloaded += glob.glob("/mnt/D/Codes/signsage/research/track_b/data/include_clips/*.mp4")
c = Counter()
for fp in downloaded:
    base = os.path.basename(fp).replace(".MOV",".mp4").replace(".MP4",".mp4")
    # Try exact match on video name (e.g., MVI_0029 -> MVI_0029)
    for vid_key, w in video_to_word.items():
        if vid_key.startswith(base) or base.startswith(vid_key.replace(".mp4","")):
            c[w] += 1
            break

print("Downloaded members matched to words:", dict(c))
print("Total downloaded files:", len(downloaded))

# Local videos by word
local_dir = "/mnt/D/Codes/signsage/backend/data/isl/videos"
local_words = Counter()
for sub in os.listdir(local_dir):
    p = os.path.join(local_dir, sub)
    if os.path.isdir(p):
        for f in os.listdir(p):
            if f.endswith(".mp4") and "__INCLUDE__" in f:
                word_part = f.split("__")[0].lower()
                local_words[word_part] += 1
# Also flat files
for f in os.listdir(local_dir):
    if f.endswith(".mp4") and "__INCLUDE__" in f:
        word_part = f.split("__")[0].lower()
        local_words[word_part] += 1

print("Local by word (subdirs + flat):", dict(local_words))
