"""Step 6: Cross-extractor test (simplified).

Evaluate 1D-CNN + GRU trained on 17-word INCLUDE data on extracted_poses/*.npy
(Holistic (T,75,3) layout from OpenHands Tasks-API extraction).

Only overlapping words: friend, hello, hospital, market, school, thank_you.
PRIMARY = official-test clips only; train-split reported separately.
Compare with OpenHands ST-GCN (45% top-1, 9/20).
"""
import os, sys, glob, json, csv, re
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/mnt/Drive-D/Codes/signsage/research/track_b")
from features import extract_features

POSES_DIR = "/mnt/Drive-D/Codes/signsage/research/openhands_pilot/extracted_poses"
CSV_TRAIN = "/mnt/Drive-D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/train_include.csv"
CSV_TEST = "/mnt/Drive-D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/test_include.csv"
LANDMARKS_DIR = "/mnt/Drive-D/Codes/signsage/research/track_b/data/include_landmarks"

OVERLAP = ["friend", "hello", "hospital", "market", "school", "thank_you"]
VOCAB = ['hello','thank you','you','friend','hospital','school','market',
         'doctor','medicine','good','bad','morning','night','today',
         'tomorrow','mother','father']
NUM = re.compile(r'^\d+\.\s*')
def norm(w):
    return re.sub(r'\s+', ' ', NUM.sub('', w or '')).strip().lower()

# ---- Video -> (word, split) from CSV ----
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

# ---- Load 17-word training data ----
X17, y17 = [], []
labels17 = sorted(VOCAB)
l2i17 = {l: i for i, l in enumerate(labels17)}
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
    X17.append(extract_features(lm, layout='signsage'))
    y17.append(l2i17[w])
X17 = np.stack(X17); y17 = np.array(y17, dtype=np.int64)
print(f"17-word train data: {len(X17)} clips")

# ---- Load extracted_poses overlap data ----
X_te, y_te, words_te = [], [], []
X_tr, y_tr = [], []
for fp in sorted(glob.glob(os.path.join(POSES_DIR, '*.npy'))):
    basename = os.path.basename(fp).replace('.npy', '')
    parts = basename.split('__')
    mvi = parts[-1] + '.mp4'
    w = vid_word.get(mvi)
    sp = vid_split.get(mvi)
    if w is None or w.replace(' ', '_') not in OVERLAP:
        continue
    lm = np.load(fp).astype(np.float32)
    feat = extract_features(lm, layout='holistic')
    w_key = w.replace(' ', '_')
    if sp == 'test':
        X_te.append(feat); y_te.append(OVERLAP.index(w_key)); words_te.append(w_key)
    else:
        X_tr.append(feat); y_tr.append(OVERLAP.index(w_key))

X_te = np.stack(X_te); y_te = np.array(y_te, dtype=np.int64)
X_tr = np.stack(X_tr); y_tr = np.array(y_tr, dtype=np.int64)
print(f"extracted_poses overlap: train={len(X_tr)} test={len(X_te)}")
print(f"  Test words: {dict(zip(*np.unique(y_te, return_counts=True)))}")

# ---- Models ----
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

class TinyGRU(nn.Module):
    def __init__(self, n_in=98, n_classes=17, hidden=32):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)
    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h.squeeze(0))

def train_quick(model, X, y, epochs=30, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    N = len(X)
    for ep in range(epochs):
        perm = np.random.permutation(N)
        for s in range(0, N, 16):
            ib = perm[s:s+16]
            xb = torch.from_numpy(X[ib]).float()
            yb = torch.from_numpy(y[ib]).long()
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()

def eval_overlap(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
    # Select overlap class logits (columns for the 6 overlap words in 17-class space)
    overlap_cols = [l2i17[w.replace('_', ' ')] for w in OVERLAP]
    logits_ov = logits[:, overlap_cols]
    pred = logits_ov.argmax(1).numpy()
    acc = (pred == y).mean()
    return float(acc)

results = {}
for name, Model in [("1D-CNN", TinyCNN1D), ("GRU", TinyGRU)]:
    print(f"\n=== {name} cross-extractor ===")
    accs = []
    for seed in range(3):  # 3 seeds for speed
        model = Model(n_in=98, n_classes=17)
        train_quick(model, X17, y17, epochs=30, seed=seed)
        acc = eval_overlap(model, X_te, y_te)
        accs.append(acc)
        # Also evaluate on train-split clips (reported separately)
        acc_tr = eval_overlap(model, X_tr, y_tr)
        print(f"  seed {seed}: test_acc={acc:.4f} train_split_acc={acc_tr:.4f}")
    results[name] = {
        'test_acc_mean': float(np.mean(accs)),
        'test_acc_std': float(np.std(accs)),
        'seed_accs': accs,
        'n_test': len(X_te),
        'n_train_split': len(X_tr),
    }
    print(f"  MEAN test_acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f} over {len(X_te)} test clips")

print(f"\n=== Comparison ===")
print(f"OpenHands ST-GCN reference: 45% top-1 (9/20 correct on 20 test clips)")
print(f"Our 1D-CNN:  {results['1D-CNN']['test_acc_mean']:.1%} ({len(X_te)} test clips)")
print(f"Our GRU:     {results['GRU']['test_acc_mean']:.1%} ({len(X_te)} test clips)")

out_path = "/mnt/Drive-D/Codes/signsage/research/track_b/results_cross_extractor.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote {out_path}")