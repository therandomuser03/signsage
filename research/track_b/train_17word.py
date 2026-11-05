"""Step 5: Train 1D-CNN + GRU on INCLUDE 17-word vocabulary.

Data: research/track_b/data/include_landmarks/*.npy (T,543,3) SignSage layout
Features: research/track_b/features.extract_features(layout="signsage") -> (60,98)
Split: OFFICIAL INCLUDE train/test (from train_include.csv / test_include.csv)
Models: 1D-CNN and GRU, 5 seeds each
Reports: mean +/- std test acc/macro-F1, per-class P/R/F1, top-10 confusions,
         chance baseline, accuracy vs max-softmax threshold with coverage.
"""
import os, sys, glob, time, json, csv, re
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

sys.path.insert(0, "/mnt/Drive-D/Codes/signsage/research/track_b")
from features import extract_features

LANDMARKS_DIR = "/mnt/Drive-D/Codes/signsage/research/track_b/data/include_landmarks"
CSV_TRAIN = "/mnt/Drive-D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/train_include.csv"
CSV_TEST = "/mnt/Drive-D/Codes/signsage/research/openhands_pilot/openhands_repo/openhands/datasets/assets/include_metadata/Train_Test_Split/test_include.csv"

VOCAB = ['hello','thank you','you','friend','hospital','school','market',
         'doctor','medicine','good','bad','morning','night','today',
         'tomorrow','mother','father']
NUM = re.compile(r'^\d+\.\s*')
def norm(w):
    return re.sub(r'\s+', ' ', NUM.sub('', w or '')).strip().lower()

# ---- Load official split ----
word_to_split = {}   # normalized word -> list of (video_id, split)
word_to_count = defaultdict(lambda: {'train': 0, 'test': 0})
video_to_split = {}
video_to_word = {}

for path, split in [(CSV_TRAIN, 'train'), (CSV_TEST, 'test')]:
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = norm(row['Word'])
            if w in VOCAB:
                vid = row['Video'].replace('.MOV','.mp4').replace('.MP4','.mp4')
                video_to_split[vid] = split
                video_to_word[vid] = w
                word_to_count[w][split] += 1

# ---- Load all landmark files ----
samples = []  # (word, split, features, filepath)
missing_files = []
for fp in sorted(glob.glob(os.path.join(LANDMARKS_DIR, '*.npy'))):
    basename = os.path.basename(fp)
    # Extract video ID from filename
    # e.g., friend__INCLUDE__00546__MVI_3826.npy  or  MVI_0001.npy
    stem = basename.replace('.npy', '')
    if '__INCLUDE__' in stem:
        vid = stem.split('__')[-1] + '.mp4'
    else:
        vid = stem + '.mp4'
    
    word = video_to_word.get(vid)
    split = video_to_split.get(vid)
    if word is None or split is None:
        missing_files.append(basename)
        continue
    
    lm = np.load(fp).astype(np.float32)  # (T,543,3)
    if lm.shape[0] == 0:
        missing_files.append(f"{basename} (0 frames)")
        continue
    feat = extract_features(lm, layout='signsage')  # (60,98)
    samples.append((word, split, feat, basename))

print(f"Loaded {len(samples)} clips with landmarks")
if missing_files:
    print(f"  WARNING: {len(missing_files)} files not in CSV split: {missing_files[:5]}")

# Per-word counts
for w in VOCAB:
    tr = sum(1 for s, sp, _, _ in samples if s == w and sp == 'train')
    te = sum(1 for s, sp, _, _ in samples if s == w and sp == 'test')
    print(f"  {w:14} train={tr:2d} test={te:2d} total={tr+te:3d}")

# Build label map
labels = sorted(VOCAB)
label_to_idx = {l: i for i, l in enumerate(labels)}
n_classes = len(labels)

# ---- Split by official train/test ----
Xtr, ytr, ftr = [], [], []
Xte, yte, fte = [], [], []
for word, split, feat, fp in samples:
    if split == 'train':
        Xtr.append(feat); ytr.append(label_to_idx[word]); ftr.append(fp)
    else:
        Xte.append(feat); yte.append(label_to_idx[word]); fte.append(fp)

Xtr = np.stack(Xtr); ytr = np.array(ytr, dtype=np.int64)
Xte = np.stack(Xte); yte = np.array(yte, dtype=np.int64)
print(f"\nTrain: {len(Xtr)} clips, Test: {len(Xte)} clips")
print(f"Train class counts: {dict(zip(*np.unique(ytr, return_counts=True)))}")
print(f"Test class counts: {dict(zip(*np.unique(yte, return_counts=True)))}")

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

# ---- Augmentation (train only) ----
def augment(x):
    B = x.shape[0]
    xp = x.reshape(B, 60, 49, 2).copy()
    for b in range(B):
        if np.random.rand() < 0.5:
            xp[b] *= np.random.uniform(0.95, 1.05)
        if np.random.rand() < 0.5:
            xp[b] += np.random.uniform(-0.03, 0.03)
        if np.random.rand() < 0.5:
            ang = np.deg2rad(np.random.uniform(-8, 8))
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            xp[b] = xp[b] @ R.T
        if np.random.rand() < 0.5:
            nw = int(60 * np.random.uniform(0.9, 1.1))
            nw = max(2, min(120, nw))
            idx = np.linspace(0, 59, nw).round().astype(int)
            warped = xp[b][idx]
            idx2 = np.linspace(0, nw - 1, 60).round().astype(int)
            xp[b] = warped[idx2]
    return xp.reshape(B, 60, 98)

def train_model(model, Xtr, ytr, epochs=60, lr=1e-3, batch=16, seed=42, verbose=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    N = len(Xtr)
    for ep in range(epochs):
        perm = np.random.permutation(N)
        total_loss = 0.0
        for s in range(0, N, batch):
            ib = perm[s:s + batch]
            xb = torch.from_numpy(augment(Xtr[ib])).float()
            yb = torch.from_numpy(ytr[ib]).long()
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(ib)
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  loss={total_loss / N:.4f}")
    model.eval()

def evaluate(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
    probs = torch.softmax(logits, dim=-1).numpy()
    pred = logits.argmax(1).numpy()
    return pred, probs, logits.numpy()

def accuracy_metrics(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = cm.trace() / max(1, cm.sum())
    per_class = {}
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        per_class[c] = (p, r, f1)
    macro_f1 = float(np.mean([v[2] for v in per_class.values()]))
    counts = np.bincount(y_true, minlength=n_classes)
    baseline = counts.max() / max(1, len(y_true))
    off = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                off.append((int(cm[i, j]), i, j))
    off.sort(reverse=True)
    return acc, macro_f1, per_class, cm, baseline, off[:10]

def threshold_curve(probs, y_true, thresholds=(0.5, 0.7, 0.9)):
    """For each threshold: accuracy on covered samples, coverage fraction."""
    results = []
    for t in thresholds:
        max_prob = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        covered = max_prob >= t
        if covered.sum() > 0:
            acc = (pred[covered] == y_true[covered]).mean()
        else:
            acc = 0.0
        coverage = covered.mean()
        results.append((t, acc, coverage, int(covered.sum())))
    return results

# ---- Train over 5 seeds ----
results = {}
for name, Model in [("1D-CNN", TinyCNN1D), ("GRU", TinyGRU)]:
    print(f"\n=== {name} (5 seeds) ===")
    seed_results = []
    for seed in range(5):
        model = Model(n_in=98, n_classes=n_classes)
        train_model(model, Xtr, ytr, seed=seed, verbose=(seed == 0))
        pred_tr, _, _ = evaluate(model, Xtr)
        pred_te, probs_te, logits_te = evaluate(model, Xte)
        acc_tr, mf1_tr, pc_tr, cm_tr, base_tr, off_tr = accuracy_metrics(ytr, pred_tr, n_classes)
        acc_te, mf1_te, pc_te, cm_te, base_te, off_te = accuracy_metrics(yte, pred_te, n_classes)
        thresh = threshold_curve(probs_te, yte)
        seed_results.append({
            'seed': seed,
            'train_acc': float(acc_tr), 'train_macro_f1': float(mf1_tr),
            'test_acc': float(acc_te), 'test_macro_f1': float(mf1_te),
            'baseline': float(base_te),
            'per_class': {labels[c]: dict(precision=float(pc_te[c][0]), recall=float(pc_te[c][1]), f1=float(pc_te[c][2])) for c in range(n_classes)},
            'confusions': [[int(v) for v in row] for row in cm_te],
            'top_confusions': [[labels[i], labels[j], cnt] for cnt, i, j in off_te],
            'threshold_curve': [{'threshold': t, 'acc': float(a), 'coverage': float(cov), 'n_covered': n} for t, a, cov, n in thresh],
        })
        print(f"  seed {seed}: train_acc={acc_tr:.4f} train_f1={mf1_tr:.4f} | test_acc={acc_te:.4f} test_f1={mf1_te:.4f}")
    
    # Aggregate
    test_accs = [r['test_acc'] for r in seed_results]
    test_f1s = [r['test_macro_f1'] for r in seed_results]
    train_accs = [r['train_acc'] for r in seed_results]
    train_f1s = [r['train_macro_f1'] for r in seed_results]
    
    results[name] = {
        'n_classes': n_classes,
        'labels': labels,
        'train_acc_mean': float(np.mean(train_accs)), 'train_acc_std': float(np.std(train_accs)),
        'train_macro_f1_mean': float(np.mean(train_f1s)), 'train_macro_f1_std': float(np.std(train_f1s)),
        'test_acc_mean': float(np.mean(test_accs)), 'test_acc_std': float(np.std(test_accs)),
        'test_macro_f1_mean': float(np.mean(test_f1s)), 'test_macro_f1_std': float(np.std(test_f1s)),
        'baseline': float(seed_results[0]['baseline']),
        'seed_results': seed_results,
    }
    
    print(f"\n  MEAN +/- STD over 5 seeds:")
    print(f"    Train acc: {results[name]['train_acc_mean']:.4f} +/- {results[name]['train_acc_std']:.4f}")
    print(f"    Train macro-F1: {results[name]['train_macro_f1_mean']:.4f} +/- {results[name]['train_macro_f1_std']:.4f}")
    print(f"    Test acc:  {results[name]['test_acc_mean']:.4f} +/- {results[name]['test_acc_std']:.4f}")
    print(f"    Test macro-F1:  {results[name]['test_macro_f1_mean']:.4f} +/- {results[name]['test_macro_f1_std']:.4f}")
    print(f"    Chance baseline: {results[name]['baseline']:.4f}")

# Aggregate per-class metrics across seeds (use seed 0 for per-class, or average)
# Average per-class P/R/F1 across seeds
per_class_avg = {}
for c in range(n_classes):
    label = labels[c]
    precs = [r['per_class'][label]['precision'] for r in seed_results[0:] for seed_results in [results['1D-CNN']['seed_results']]]  # placeholder
    # Use the first model's seed results for per-class display
    pass

# Use 1D-CNN seed 0 per-class as representative
rep = results['1D-CNN']['seed_results'][0]
print(f"\n  Per-class P/R/F1 (1D-CNN, seed 0):")
for label in labels:
    pc = rep['per_class'][label]
    print(f"    {label:14} P={pc['precision']:.2f} R={pc['recall']:.2f} F1={pc['f1']:.2f}")

print(f"\n  Top confusions (1D-CNN, seed 0, true->pred):")
for label_i, label_j, cnt in rep['top_confusions']:
    print(f"    {label_i:14} -> {label_j:14}  {cnt}")

print(f"\n  Threshold curve (1D-CNN, seed 0):")
for tc in rep['threshold_curve']:
    print(f"    threshold={tc['threshold']} acc={tc['acc']:.3f} coverage={tc['coverage']:.3f} n={tc['n_covered']}")

# Save results
out_path = "/mnt/Drive-D/Codes/signsage/research/track_b/results_17word.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote {out_path}")