"""Step 1: Match target ISL concepts to INCLUDE labels with clip counts.
Reads train_include.csv / test_include.csv from the OpenHands repo metadata.
"""
import csv
import re
from collections import defaultdict, Counter

TRAIN = "/mnt/Drive-D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/train_include.csv"
TEST = "/mnt/Drive-D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/test_include.csv"

TARGETS = [
    "hello", "thank you", "water", "hospital", "school", "friend", "market",
    "you", "i/me", "doctor", "medicine", "help", "please", "sorry", "yes", "no",
    "good", "bad", "morning", "night", "today", "tomorrow", "mother", "father",
    "home", "food", "eat", "drink", "go", "come", "name", "what", "where", "how",
    "need", "want", "understand",
]

NUM_PREFIX_RE = re.compile(r"^\d+\.\s*")


def norm(word):
    """Strip 'NN.' prefix, lowercase, collapse whitespace."""
    w = NUM_PREFIX_RE.sub("", word or "")
    return re.sub(r"\s+", " ", w).strip().lower()


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


train_rows = load(TRAIN)
test_rows = load(TEST)

# Index all INCLUDE labels by normalized word -> list of rows
label_rows = defaultdict(list)
for row in train_rows + test_rows:
    label_rows[norm(row["Word"])].append(row)

print("=" * 70)
print("STEP 1 - Target concept -> INCLUDE label matching (clip counts)")
print("=" * 70)

matched_any = False
for t in TARGETS:
    rows = label_rows.get(t, [])
    if not rows:
        print(f"[MISS] {t!r:22} -> no INCLUDE label")
        continue
    matched_any = True
    # representative label (first occurrence) + counts
    rep = rows[0]["Word"]
    n_train = sum(1 for r in rows if r.get("_split") == "train") if False else None
    # determine split by source file: we loaded combined; recompute cleanly
    tr = sum(1 for r in train_rows if norm(r["Word"]) == t)
    te = sum(1 for r in test_rows if norm(r["Word"]) == t)
    print(f"[HIT]  {t!r:22} -> INCLUDE '{rep}'  train={tr}  test={te}  total={tr+te}")

if not matched_any:
    print("(none matched)")

print("\n" + "=" * 70)
print("STEP 1 - 40 INCLUDE labels with most clips (train+test)")
print("=" * 70)
label_counts = Counter()
for row in train_rows + test_rows:
    label_counts[row["Word"]] += 1
for word, cnt in label_counts.most_common(40):
    tr = sum(1 for r in train_rows if r["Word"] == word)
    te = sum(1 for r in test_rows if r["Word"] == word)
    print(f"  {cnt:4d}  train={tr:3d}  test={te:3d}  {word}")

print("\n" + "=" * 70)
print("STEP 1 - Avatar vocabulary (sign_vocabulary.json) vs INCLUDE labels")
print("=" * 70)
AVATAR = ["HELLO", "THANK_YOU", "SORRY", "PLEASE", "YES", "NO", "HELP", "WATER",
          "FOOD", "NAME", "YOU", "ME", "WHERE", "HOW"]
include_norm_set = {norm(w) for w in label_counts}
for aid in AVATAR:
    n = aid.lower().replace("_", " ")
    hit = n in include_norm_set
    # find representative INCLUDE label(s)
    reps = [w for w in label_counts if norm(w) == n]
    cnt = sum(label_counts[w] for w in reps)
    print(f"  avatar {aid:12} -> norm {n!r:16} in INCLUDE={hit}  labels={reps}  total_clips={cnt}")
