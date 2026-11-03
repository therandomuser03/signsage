"""Step 3: Download missing INCLUDE members from Zenodo via RemoteZip range requests.
Only downloads members needed for the 17-word vocabulary; skips existing.
Keeps all on /mnt/D; reports counts, failures, total downloaded bytes.
"""
import csv, os, time, sys
from collections import defaultdict
sys.path.insert(0, "/mnt/D/Codes/signsage/research/track_b")

TRAIN = "/mnt/D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/train_include.csv"
TEST = "/mnt/D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/test_include.csv"
OUT_DIR = "/mnt/D/Codes/signsage/research/track_b/data/include_clips"
os.makedirs(OUT_DIR, exist_ok=True)

VOCAB = [
    "hello","thank you","you","friend","hospital","school","market",
    "doctor","medicine","good","bad","morning","night","today","tomorrow","mother","father"
]
NUM = __import__("re").compile(r"^\d+\.\s*")

def norm(w):
    return __import__("re").sub(r"\s+", " ", NUM.sub("", w or "")).strip().lower()

# Build needed file paths (category/member/video.ext) from CSV
needed = defaultdict(set)  # word -> set of zip-member paths
for path in (TRAIN, TEST):
    split = os.path.basename(os.path.dirname(path)).replace("_include.csv","")
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = norm(row["Word"])
            if w in VOCAB:
                needed[w].add(row["FilePath"])  # e.g., "Greetings/55. Thank you/MVI_0006.MOV"

total_needed_members = sum(len(s) for s in needed.values())
print("STEP 3 - Needed members for 17 words:", total_needed_members)
for w in VOCAB:
    print(f"  {w:14} -> {len(needed[w]):2} members")

# Zenodo file list (from /tmp/zenodo_full.json already fetched)
import json
with open("/tmp/zenodo_full.json") as f:
    zenodo = json.load(f)

# Build URL -> file_path mapping for needed categories
urls = {}
for f in zenodo.get("files", []):
    key = f["key"]  # e.g., "Greetings_1of2.zip"
    url = f["links"]["self"]  # content URL
    # Derive category from key
    cat = key.split("_")[0]  # "Greetings"
    urls.setdefault(cat, []).append(url)

needed_cats = set()
# Derive category from file paths
for word, members in needed.items():
    for m in members:
        cat = m.split("/")[0]
        needed_cats.add(cat)

print(f"\nNeeded category zips: {sorted(needed_cats)}")

# Download via RemoteZip
import remotezip, time
import numpy as np

downloaded = 0
download_bytes = 0
failed = []

start_t = time.time()
for cat in sorted(needed_cats):
    urls_for_cat = urls.get(cat, [])
    print(f"\n--- Category {cat} ({len(urls_for_cat)} zip parts) ---")
    for url in urls_for_cat:
        try:
            with remotezip.RemoteZip(url) as z:
                all_info = z.infolist()
                members = [info for info in all_info if info.filename.startswith(cat + "/")]
                # Filter to members needed for this word
                for word in VOCAB:
                    needed_set = needed[word]
                    for info in members:
                        fname = info.filename
                        # Match if file path contains word or video ID
                        # Simplest: check if fname is in needed_set for this word's category
                        # But need to handle category path correctly
                        # If fname starts with cat + "/", check if any needed member has same basename or path
                        basename = os.path.basename(fname)
                        # Try direct match on full path
                        for need_path in needed_set:
                            if fname == need_path or fname.endswith("/" + basename) or basename == os.path.basename(need_path):
                                # Extract this member
                                dest = os.path.join(OUT_DIR, basename.replace(".MOV",".mp4").replace(".MP4",".mp4"))
                                # Actually keep original extension; just save as basename
                                dest = os.path.join(OUT_DIR, basename)
                                if os.path.exists(dest):
                                    continue
                                with open(dest, "wb") as out_f:
                                    with z.open(info.filename) as src_f:
                                        out_f.write(src_f.read())
                                sz = os.path.getsize(dest)
                                downloaded += 1
                                download_bytes += sz
                                print(f"  OK {fname} -> {dest} ({sz/1024:.1f}KB)")
                                # Remove from needed to avoid duplicate work
                                needed[word].discard(need_path)
        except Exception as e:
            print(f"  FAIL {cat} url error: {type(e).__name__} {str(e)[:120]}")
            failed.append((cat, url, str(e)[:200]))
    # After all parts for this category, check remaining needed members
    remaining = sum(len(needed[w]) for w in VOCAB)
    if remaining == 0:
        print("  All members downloaded.")
        break

elapsed = time.time() - start_t
print(f"\n=== DOWNLOAD SUMMARY ===")
print(f"Downloaded members: {downloaded}")
print(f"Total bytes downloaded: {download_bytes/1e6:.2f} MB")
print(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"Failed attempts: {len(failed)}")
for f in failed[:5]:
    print(f"  {f[0]} {f[1]}: {f[2]}")

# Report remaining missing
remaining_by_word = {w: len(needed[w]) for w in VOCAB}
remaining_total = sum(remaining_by_word.values())
print(f"Remaining members: {remaining_total}")
for w in VOCAB:
    if remaining_by_word[w] > 0:
        print(f"  {w}: {remaining_by_word[w]} still missing")
