#!/usr/bin/env python3
"""Evaluate all 4 OpenHands INCLUDE checkpoints on the 122-clip 6-word pilot dataset.

Measures:
- Per-word and overall Top-1 and Top-5 accuracy
- Split breakdown (Train vs Test)
- Per-clip latency on CPU (mean, median, p95)
- Peak RAM usage (MB)

Outputs structured results to evaluation_results.json.
ponytail: ceiling=6 words pilot; upgrade path = evaluate on all 263 classes if full dataset is downloaded.
"""
import os, sys, time, glob, json, resource, tracemalloc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import omegaconf

REPO_DIR = os.path.join(os.path.dirname(__file__), "openhands_repo")
sys.path.insert(0, REPO_DIR)

from openhands.models.loader import get_model
from openhands.datasets.pose_transforms import PoseSelect, CenterAndScaleNormalize, PoseUniformSubsampling

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSES_DIR = os.path.join(BASE_DIR, "extracted_poses")
METADATA_DIR = os.path.join(BASE_DIR, "checkpoints/include_metadata/Train_Test_Split")

CHECKPOINTS = {
    "stgcn": {
        "cfg": os.path.join(BASE_DIR, "checkpoints/include_stgcn/include/st_gcn/config.yaml"),
        "ckpt": os.path.join(BASE_DIR, "checkpoints/include_stgcn/include/st_gcn/epoch=215-step=23543.ckpt"),
        "use_uniform_sub": False,
    },
    "lstm": {
        "cfg": os.path.join(BASE_DIR, "checkpoints/include_lstm/include/lstm/config.yaml"),
        "ckpt": os.path.join(BASE_DIR, "checkpoints/include_lstm/include/lstm/epoch=294-step=63719.ckpt"),
        "use_uniform_sub": False,
    },
    "bert": {
        "cfg": os.path.join(BASE_DIR, "checkpoints/include_bert/include/bert/config.yaml"),
        "ckpt": os.path.join(BASE_DIR, "checkpoints/include_bert/include/bert/epoch=328-step=18094.ckpt"),
        "use_uniform_sub": True,
    },
    "slgcn": {
        "cfg": os.path.join(BASE_DIR, "checkpoints/include_slgcn/include/sl_gcn/config.yaml"),
        "ckpt": os.path.join(BASE_DIR, "checkpoints/include_slgcn/include/sl_gcn/epoch=112-step=12203.ckpt"),
        "use_uniform_sub": False,
    },
}

WORD_MAP = {
    "friend": "81.Friend",
    "hello": "48.Hello",
    "hospital": "30.Hospital",
    "market": "32.Market",
    "school": "24.School",
    "thank_you": "55.Thankyou",
}

def load_dataset_metadata():
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train_include.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test_include.csv"))
    
    train_vids = set(train_df["Video"].str.lower().str.replace(".mov", "").str.replace(".mp4", ""))
    test_vids = set(test_df["Video"].str.lower().str.replace(".mov", "").str.replace(".mp4", ""))
    
    glosses = sorted(train_df["Word"].unique())
    gloss_to_idx = {g: i for i, g in enumerate(glosses)}
    
    # Load all extracted clips
    pose_files = sorted(glob.glob(os.path.join(POSES_DIR, "*.npy")))
    samples = []
    for pf in pose_files:
        fname = os.path.basename(pf).replace(".npy", "")
        # fname e.g. friend__INCLUDE__00546__MVI_3826
        parts = fname.split("__")
        w_key = parts[0]
        mvi = parts[-1].lower()
        
        split = "train" if mvi in train_vids else ("test" if mvi in test_vids else "unknown")
        true_label = WORD_MAP.get(w_key, "unknown")
        true_idx = gloss_to_idx.get(true_label, -1)
        
        samples.append({
            "filename": fname,
            "path": pf,
            "word_key": w_key,
            "true_label": true_label,
            "true_idx": true_idx,
            "split": split,
        })
    return samples, glosses, gloss_to_idx

def evaluate_model(model_name, info, samples, glosses):
    print(f"\n==========================================")
    print(f"Evaluating Model: {model_name.upper()}")
    print(f"==========================================")
    
    # Measure memory before
    tracemalloc.start()
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss # KB on Linux
    
    cfg = omegaconf.OmegaConf.load(info["cfg"])
    model = get_model(cfg.model, in_channels=2, num_class=263)
    ckpt = torch.load(info["ckpt"], map_location="cpu", weights_only=False)
    state = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.eval()
    
    pose_select = PoseSelect(preset="mediapipe_holistic_minimal_27")
    center_scale = CenterAndScaleNormalize(reference_points_preset="shoulder_mediapipe_holistic_minimal_27")
    uniform_sub = PoseUniformSubsampling(num_frames=120) if info["use_uniform_sub"] else None
    
    results = []
    latencies = []
    
    # Warmup
    dummy = torch.randn(1, 2, 120 if info["use_uniform_sub"] else 70, 27)
    with torch.no_grad():
        _ = model(dummy)
        
    for s in samples:
        raw_kps = np.load(s["path"]) # (T, 75, 3)
        kps2d = raw_kps[:, :, :2].astype(np.float32)
        
        t0 = time.perf_counter()
        data = {"frames": torch.tensor(kps2d).permute(2, 0, 1)} # (2, T, 75)
        data = pose_select(data) # (2, T, 27)
        if uniform_sub:
            data = uniform_sub(data)
        data = center_scale(data) # (2, T, 27)
        
        inp = data["frames"].unsqueeze(0) # (1, 2, T, 27)
        with torch.no_grad():
            logits = model(inp)
            probs = F.softmax(logits, dim=-1)[0]
            top5_v, top5_i = probs.topk(5)
            
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)
        
        top1_idx = top5_i[0].item()
        top5_indices = top5_i.tolist()
        
        is_top1 = (top1_idx == s["true_idx"])
        is_top5 = (s["true_idx"] in top5_indices)
        
        results.append({
            "filename": s["filename"],
            "word": s["word_key"],
            "split": s["split"],
            "true_idx": s["true_idx"],
            "true_label": s["true_label"],
            "pred_top1_idx": top1_idx,
            "pred_top1_label": glosses[top1_idx],
            "pred_top1_prob": float(top5_v[0].item()),
            "top5_labels": [glosses[i] for i in top5_indices],
            "top5_probs": [float(v) for v in top5_v.tolist()],
            "is_top1": is_top1,
            "is_top5": is_top5,
            "latency_ms": latency_ms,
        })
        
    mem_peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    traced_curr, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_ram_mb = mem_peak_kb / 1024.0 # MB
    
    # Compute metrics
    total_clips = len(results)
    top1_correct = sum(r["is_top1"] for r in results)
    top5_correct = sum(r["is_top5"] for r in results)
    
    acc_top1 = (top1_correct / total_clips) * 100.0
    acc_top5 = (top5_correct / total_clips) * 100.0
    
    # Split metrics
    splits_summary = {}
    for sp in ["train", "test"]:
        sp_res = [r for r in results if r["split"] == sp]
        if sp_res:
            sp_top1 = sum(r["is_top1"] for r in sp_res) / len(sp_res) * 100.0
            sp_top5 = sum(r["is_top5"] for r in sp_res) / len(sp_res) * 100.0
            splits_summary[sp] = {
                "count": len(sp_res),
                "top1_acc": sp_top1,
                "top5_acc": sp_top5,
            }
            
    # Word metrics
    words_summary = {}
    for w in sorted(set(r["word"] for r in results)):
        w_res = [r for r in results if r["word"] == w]
        w_top1 = sum(r["is_top1"] for r in w_res) / len(w_res) * 100.0
        w_top5 = sum(r["is_top5"] for r in w_res) / len(w_res) * 100.0
        words_summary[w] = {
            "count": len(w_res),
            "top1_acc": w_top1,
            "top5_acc": w_top5,
            "correct_top1": sum(r["is_top1"] for r in w_res),
            "correct_top5": sum(r["is_top5"] for r in w_res),
        }
        
    latency_summary = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }
    
    print(f"Overall Accuracy: Top-1 = {acc_top1:.2f}% ({top1_correct}/{total_clips}), Top-5 = {acc_top5:.2f}% ({top5_correct}/{total_clips})")
    print(f"Split breakdown:")
    for sp, met in splits_summary.items():
        print(f"  {sp.upper()} ({met['count']} clips): Top-1 = {met['top1_acc']:.2f}%, Top-5 = {met['top5_acc']:.2f}%")
    print(f"Per-word Top-1 Accuracy:")
    for w, met in words_summary.items():
        print(f"  {w:10s} ({met['count']:2d} clips): Top-1 = {met['top1_acc']:5.1f}% ({met['correct_top1']}/{met['count']}), Top-5 = {met['top5_acc']:5.1f}%")
    print(f"Latency: Mean = {latency_summary['mean_ms']:.2f} ms, Median = {latency_summary['median_ms']:.2f} ms, P95 = {latency_summary['p95_ms']:.2f} ms")
    print(f"Peak RAM (process RSS): {peak_ram_mb:.1f} MB (Traced model peak: {traced_peak/(1024*1024):.1f} MB)")
    
    return {
        "model": model_name,
        "checkpoint_file": os.path.basename(info["ckpt"]),
        "config_file": os.path.basename(info["cfg"]),
        "total_clips": total_clips,
        "top1_accuracy": acc_top1,
        "top5_accuracy": acc_top5,
        "split_summary": splits_summary,
        "word_summary": words_summary,
        "latency_ms": latency_summary,
        "peak_ram_mb": peak_ram_mb,
        "traced_peak_mb": traced_peak / (1024 * 1024),
        "predictions": results,
    }

def main():
    samples, glosses, gloss_to_idx = load_dataset_metadata()
    print(f"Loaded metadata: {len(samples)} extracted clips found.")
    if len(samples) == 0:
        print("ERROR: No extracted clips found in extracted_poses/. Run extract_poses.py first.")
        sys.exit(1)
        
    all_evaluations = {}
    for name, info in CHECKPOINTS.items():
        res = evaluate_model(name, info, samples, glosses)
        all_evaluations[name] = res
        
    out_json = os.path.join(BASE_DIR, "evaluation_results.json")
    with open(out_json, "w") as f:
        json.dump(all_evaluations, f, indent=2)
    print(f"\nAll results saved to {out_json}")

if __name__ == "__main__":
    main()
