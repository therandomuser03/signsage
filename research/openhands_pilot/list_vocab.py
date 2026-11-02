#!/usr/bin/env python3
"""Print all 263 INCLUDE class labels in model-sorted order (alphabetical by Word column).

Usage: python list_vocab.py [--split_csv path/to/train_include.csv]

ponytail: ceiling=263 fixed classes; upgrade path = load from checkpoint hyper_parameters if ever changed.
"""
import sys, argparse, pandas as pd

DEFAULT_CSV = "checkpoints/include_metadata/Train_Test_Split/train_include.csv"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_csv", default=DEFAULT_CSV)
    args = p.parse_args()

    df = pd.read_csv(args.split_csv)
    glosses = sorted(df["Word"].unique())
    print(f"Total classes: {len(glosses)}")
    for i, g in enumerate(glosses):
        print(f"{i:3d}  {g}")

if __name__ == "__main__":
    main()
