"""Step 7: Export model artifacts to research/track_b/export/.

Exports:
  - state dict (.pt) of the best 1D-CNN (17-class)
  - JSON spec: labels, feature spec, normalisation constants,
    extractor name/version/options
  - Golden test vector: one (T,543,3) .npy plus expected logits
ONNX: skipped (would exceed 15 minutes; not required).
"""
import os, sys, glob, json, csv, re, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/mnt/Drive-D/Codes/signsage/research/track_b")
from features import extract_features, POSE_JOINTS, N_FRAMES, FEAT_DIM

EXPORT_DIR = "/mnt/Drive-D/Codes/signsage/research/track_b/export"
LANDMARKS_DIR = "/mnt/Drive-D/Codes/signsage/research/track_b/data/include_landmarks"
CSV_TRAIN = "/mnt/Drive-D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/train_include.csv"
CSV_TEST = "/mnt/Drive-D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/test_include.csv"

VOCAB = ['hello','thank you','you','friend','hospital','school','market',
         'doctor','medicine','good','bad','morning','night','today',
         'tomorrow','mother','father']
NUM = re.compile(r'^\d+\.\s*')
def norm(w):
    return re.sub(r'\s+', ' ', NUM.sub('', w or '')).strip().lower()

os.makedirs(EXPORT_DIR, exist_ok=True)

labels = sorted(VOCAB)
n_classes = len(labels)
label_to_idx = {l: i for i, l in enumerate(labels)}

# ---- Build train data ----
vid_word = {}
vid_split = {}
for path, split in [(CSV_TRAIN, 'train'), (CSV_TEST, 'test')]:
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = norm(row['Word'])
            if w in VOCAB:
                vid = row['Video'].replace('.MOV', '.mp4').replace('.MP4', '.mp4')
                vid_word[vid] = w
                vid_split[vid] = split

Xtr, ytr = [], []
for fp in sorted(glob.glob(os.path.join(LANDMARKS_DIR, '*.npy'))):
    stem = os.path.basename(fp).replace('.npy', '')
    if '__INCLUDE__' in stem:
        vid = stem.split('__')[-1] + '.mp4'
    else:
        vid = stem + '.mp4'
    w = vid_word.get(vid)
    sp = vid_split.get(vid)
    if w is None or sp != 'train':
        continue
    lm = np.load(fp).astype(np.float32)
    if lm.shape[0] == 0:
        continue
    Xtr.append(extract_features(lm, layout='signsage'))
    ytr.append(label_to_idx[w])
Xtr = np.stack(Xtr); ytr = np.array(ytr, dtype=np.int64)
print(f"Export training data: {len(Xtr)} clips, {n_classes} classes")

# ---- Model ----
class TinyCNN1D(nn.Module):
    def __init__(self, n_in=98, n_classes=17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x).squeeze(-1)
        return self.head(x)

torch.manual_seed(42)
np.random.seed(42)
model = TinyCNN1D(n_in=98, n_classes=n_classes)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
N = len(Xtr)
for ep in range(60):
    perm = np.random.permutation(N)
    total = 0.0
    for s in range(0, N, 16):
        ib = perm[s:s+16]
        xb = torch.from_numpy(Xtr[ib]).float()
        yb = torch.from_numpy(ytr[ib]).long()
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(ib)
model.eval()
print(f"Trained export model, final loss={total/N:.4f}")

# ---- Save state dict ----
torch.save(model.state_dict(), os.path.join(EXPORT_DIR, "model_1dcnn_17word.pt"))
print(f"Saved state dict")

# ---- JSON spec ----
spec = {
    "model": "1D-CNN",
    "n_classes": n_classes,
    "labels": labels,
    "feature_spec": {
        "dim": FEAT_DIM,
        "n_frames": N_FRAMES,
        "layout": "signsage",
        "description": "left_hand(21x,y=42) + right_hand(21x,y=42) + 7_pose_joints(x,y=14) = 98 dims/frame",
        "pose_joints": POSE_JOINTS,  # nose, leye, reye, lsh, rsh, lel, rel
        "normalisation": "shoulder-center + shoulder-width per frame",
        "temporal": f"resample to {N_FRAMES} frames (uniform subsample if T>=60, edge-pad if T<60)",
        "coords": "x,y only (z dropped)",
    },
    "extractor": {
        "name": "MediaPipe Tasks Vision",
        "version": "1.2.2",
        "api": "tasks-vision (PoseLandmarker + FaceLandmarker + HandLandmarker)",
        "running_mode": "IMAGE",
        "delegate": "CPU",
        "num_poses": 1,
        "num_faces": 1,
        "num_hands": 2,
        "model_assets": {
            "pose": "pose_landmarker_lite (float16)",
            "face": "face_landmarker (float16)",
            "hand": "hand_landmarker (float16)",
        },
        "min_confidences": {"pose_detection": 0.5, "pose_presence": 0.5, "pose_tracking": 0.5},
        "note": "Frontend uses VIDEO mode + GPU delegate; Python uses IMAGE mode + CPU delegate. Missing landmarks = [0,0,0] zeros.",
    },
    "output_layout": {
        "pose": "0-32 (33 landmarks)",
        "face": "33-500 (468 landmarks)",
        "left_hand": "501-521 (21 landmarks)",
        "right_hand": "522-542 (21 landmarks)",
        "total": 543,
    },
    "dataset": "INCLUDE (Zenodo record 4010759, CC-BY-4.0)",
    "attribution": "INCLUDE, Sridhar et al., ACM MM 2020, AI4Bharat",
    "vocabulary": VOCAB,
    "split": "official INCLUDE train/test",
    "train_clips": int(len(Xtr)),
}
with open(os.path.join(EXPORT_DIR, "spec.json"), "w") as f:
    json.dump(spec, f, indent=2)
print(f"Saved spec.json")

# ---- Golden test vector ----
# Pick one test clip, extract logits
vid_test_word = {}
for path in [CSV_TEST]:
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = norm(row['Word'])
            if w in VOCAB:
                vid = row['Video'].replace('.MOV', '.mp4').replace('.MP4', '.mp4')
                vid_test_word[vid] = w

# Find a test clip in landmarks
golden = None
for fp in sorted(glob.glob(os.path.join(LANDMARKS_DIR, '*.npy'))):
    stem = os.path.basename(fp).replace('.npy', '')
    if '__INCLUDE__' in stem:
        vid = stem.split('__')[-1] + '.mp4'
    else:
        vid = stem + '.mp4'
    if vid in vid_test_word:
        lm = np.load(fp).astype(np.float32)
        if lm.shape[0] > 0:
            golden = (lm, vid_test_word[vid], os.path.basename(fp))
            break

if golden:
    lm, true_word, fname = golden
    feat = extract_features(lm, layout='signsage')  # (60,98)
    with torch.no_grad():
        logits = model(torch.from_numpy(feat[None, ...]).float())
    probs = torch.softmax(logits, dim=-1).numpy()[0]
    pred_idx = int(probs.argmax())
    
    np.save(os.path.join(EXPORT_DIR, "golden_test_vector.npy"), lm)
    golden_info = {
        "landmark_file": fname,
        "true_word": true_word,
        "true_idx": label_to_idx[true_word],
        "pred_word": labels[pred_idx],
        "pred_idx": pred_idx,
        "correct": pred_idx == label_to_idx[true_word],
        "logits": logits.numpy()[0].tolist(),
        "probs": probs.tolist(),
        "landmark_shape": list(lm.shape),
        "feature_shape": list(feat.shape),
    }
    with open(os.path.join(EXPORT_DIR, "golden_test_vector.json"), "w") as f:
        json.dump(golden_info, f, indent=2)
    print(f"Golden test vector: {fname} -> true={true_word} pred={labels[pred_idx]} correct={golden_info['correct']}")
else:
    print("WARNING: No golden test vector found")

print(f"\nExport complete: {EXPORT_DIR}/")
for f in sorted(os.listdir(EXPORT_DIR)):
    print(f"  {f}")