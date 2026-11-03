"""Step 2: Build clip map for the 17-word vocabulary from train_include.csv / test_include.csv.
Reuses CSVs from research/openhands_pilot/openhands_repo metadata.
"""
import csv
import os
from collections import defaultdict

TRAIN = "/mnt/D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/train_include.csv"
TEST = "/mnt/D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/test_include.csv"

VIDEOS_DIR = "/mnt/D/Codes/signsage/backend/data/isl/videos"

VOCAB = [
    "hello", "thank you", "you", "friend", "hospital", "school", "market",
    "doctor", "medicine", "good", "bad", "morning", "night", "today",
    "tomorrow", "mother", "father",
]

NUM_PREFIX_RE = __import__("re").compile(r"^\d+\.\s*")


def norm(word):
    return __import__("re").sub(r"\s+", " ", NUM_PREFIX_RE.sub("", word or "")).strip().lower()


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["_split"] = os.path.basename(os.path.dirname(path)).replace("_include.csv", "")
            rows.append(row)
    return rows


train_rows = load(TRAIN)
test_rows = load(TEST)
all_rows = train_rows + test_rows

# index by normalized word -> list of (split, category, word, video, filepath)
by_word = defaultdict(list)
for r in all_rows:
    by_word[norm(r["Word"])].append(r)

# existing local videos
local = set(os.listdir(VIDEOS_DIR))
local_include = {f for f in local if f.endswith(".mp4") and "__INCLUDE__" in f}

print("=" * 70)
print("STEP 2 - 17-word vocabulary clip map")
print("=" * 70)
total = 0
for w in VOCAB:
    rows = by_word.get(w, [])
    if not rows:
        print(f"[MISSING-IN-CSV] {w!r:14} -> no CSV rows")
        continue
    n_train = sum(1 for r in train_rows if norm(r["Word"]) == w)
    n_test = sum(1 for r in test_rows if norm(r["Word"]) == w)
    # representative category + filepath from first train row (or test)
    rep = None
    src = train_rows if n_train else test_rows
    for r in src:
        if norm(r["Word"]) == w:
            rep = r
            break
    cat = rep["Category"] if rep else "?"
    fp = rep["FilePath"] if rep else "?"
    video = rep["Video"] if rep else "?"
    # check local existence (both flat and subdir)
    local_names = {f for f in local_include if video in f or w in f.lower()}
    local_exists = any(video in f for f in local_include)
    total += len(rows)
    print(f"[{'LOCAL' if local_exists else 'DOWNLOAD'}] {w!r:14} cat={cat:16} train={n_train} test={n_test} total={len(rows):3}  fp={fp}  video={video}  local={local_exists}")

print(f"\nTOTAL 17-word clips in CSV: {total}")
print(f"Local INCLUDE videos present: {len(local_include)}")

# Show unique categories among the 17 words
cats = set()
for w in VOCAB:
    for r in by_word.get(w, []):
        cats.add(r["Category"])
print(f"Categories involved: {sorted(cats)}")
