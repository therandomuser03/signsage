import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'research/training')
import pandas as pd
from audit_dataset import clean_dataframe, video_split

def test_no_class_coverage_pipeline():
    """Demonstrate that the six-class experiment is not evaluable with clean_dataframe pipeline."""
    df = pd.read_csv('staging_kaggle_results/isign/iSign_v1.1.csv')
    VOCAB = ['you','music','three','one','water','two']
    sub = df[df['text'].astype(str).str.strip().str.lower().isin(VOCAB)].copy()
    
    # Full pipeline: clean_dataframe first (as the training script does)
    cleaned, info = clean_dataframe(sub)
    print(f"Pipeline: {info['usable_rows']} usable rows, {info['unique_videos']} unique videos")
    print(f"Top targets: {info['top_targets']}")
    
    # Check all seeds 1-1000
    val_with_all = 0
    test_with_all = 0
    seed_with_val_all = []
    seed_with_test_all = []
    
    for seed in range(1, 1001):
        train_idx, val_idx, test_idx, split_info = video_split(cleaned, seed=seed)
        val_classes = set(val_idx['text'].astype(str).str.strip().str.lower())
        test_classes = set(test_idx['text'].astype(str).str.strip().str.lower())
        
        if set(VOCAB) <= val_classes:
            val_with_all += 1
            seed_with_val_all.append(seed)
        if set(VOCAB) <= test_classes:
            test_with_all += 1
            seed_with_test_all.append(seed)
    
    print(f"\nSeeds where val has all 6 classes: {val_with_all}/1000")
    print(f"Seeds where test has all 6 classes: {test_with_all}/1000")
    
    # Show worst-case seeds (current default seed=1)
    train_idx, val_idx, test_idx, split_info = video_split(cleaned, seed=1)
    val_classes = set(val_idx['text'].astype(str).str.strip().str.lower())
    test_classes = set(test_idx['text'].astype(str).str.strip().str.lower())
    print(f"\nseed=1 (previous default):")
    print(f"  val missing: {set(VOCAB) - val_classes}")
    print(f"  test missing: {set(VOCAB) - test_classes}")
    
    # Show best seed
    best_seed = min(range(1, 1001), 
                    key=lambda s: len(set(VOCAB) - set(video_split(cleaned, seed=s)[1]['text'].astype(str).str.strip().str.lower())))
    train_idx, val_idx, test_idx, _ = video_split(cleaned, best_seed)
    val_classes = set(val_idx['text'].astype(str).str.strip().str.lower())
    test_classes = set(test_idx['text'].astype(str).str.strip().str.lower())
    print(f"\nbest seed={best_seed}:")
    print(f"  val missing: {set(VOCAB) - val_classes}")
    print(f"  test missing: {set(VOCAB) - test_classes}")

def test_video_never_overlaps():
    """Verify that video IDs never overlap across partitions (invariant)."""
    df = pd.read_csv('staging_kaggle_results/isign/iSign_v1.1.csv')
    VOCAB = ['you','music','three','one','water','two']
    sub = df[df['text'].astype(str).str.strip().str.lower().isin(VOCAB)].copy()
    cleaned, info = clean_dataframe(sub)
    
    # Check seed=1 (previous default)
    train_idx, val_idx, test_idx, _ = video_split(cleaned, seed=1)
    assert set(train_idx["video_id"]).isdisjoint(set(val_idx["video_id"]))
    assert set(train_idx["video_id"]).isdisjoint(set(test_idx["video_id"]))
    assert set(val_idx["video_id"]).isdisjoint(set(test_idx["video_id"]))
    print("\nPASS: Video IDs never overlap across partitions (invariant)")
    
    # Check seed=909
    train_idx, val_idx, test_idx, _ = video_split(cleaned, seed=909)
    assert set(train_idx["video_id"]).isdisjoint(set(val_idx["video_id"]))
    assert set(train_idx["video_id"]).isdisjoint(set(test_idx["video_id"]))
    assert set(val_idx["video_id"]).isdisjoint(set(test_idx["video_id"]))
    print("PASS: Video IDs never overlap with seed=909 (invariant)")

if __name__ == '__main__':
    test_no_class_coverage_pipeline()
    test_video_never_overlaps()
