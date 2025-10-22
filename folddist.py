#!/usr/bin/env python3
"""
verify_folds.py

Verifies train/test splits and prints 3-class window distributions per fold
for the NormWear LBNP dataset, using existing .pkl windows and JSON splits.
"""

import pickle
import json
from pathlib import Path
from collections import Counter

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_DIR = Path("/home/naim/LBNP/level7/pkl_windows_normwear_v3")
PKL_DIR = OUTPUT_DIR / "pkls"
SPLIT_DIR = OUTPUT_DIR / "splits"

CLASS_LABELS = {1: "mild", 2: "moderate", 3: "severe"}

# -----------------------------
# Helpers
# -----------------------------
def summarize_fold(json_path: Path, pkl_dir: Path):
    """
    Reads train_test_split JSON and counts class labels per phase.
    Returns: dict with keys 'train' and 'test', each a Counter of class -> count
    """
    with open(json_path, "r") as f:
        split = json.load(f)
    counts = {"train": Counter(), "test": Counter()}
    subjects = {"train": set(), "test": set()}
    for phase in ["train", "test"]:
        for pkl_name in split[phase]:
            pkl_path = pkl_dir / pkl_name
            with open(pkl_path, "rb") as fh:
                data = pickle.load(fh)
            cls = data["label"][0]["class"]
            counts[phase][cls] += 1
            subjects[phase].add(data["meta"]["subj"])
    return counts, subjects

# -----------------------------
# Main
# -----------------------------
def main():
    fold_summaries = {}
    for fold_idx in [1,2,3]:
        json_path = SPLIT_DIR / f"train_test_split_fold{fold_idx}.json"
        if not json_path.exists():
            print(f"⚠️ JSON split for fold {fold_idx} not found: {json_path}")
            continue
        counts, subjects = summarize_fold(json_path, PKL_DIR)
        fold_summaries[fold_idx] = {"counts": counts, "subjects": subjects}

        print(f"\n📊 Fold {fold_idx} summary:")
        print(f"  Train subjects ({len(subjects['train'])}): {sorted(subjects['train'])}")
        print(f"    Class distribution: " +
              ", ".join([f"{CLASS_LABELS[c]}={counts['train'].get(c,0)}" for c in sorted(CLASS_LABELS)]))
        print(f"  Test subjects ({len(subjects['test'])}): {sorted(subjects['test'])}")
        print(f"    Class distribution: " +
              ", ".join([f"{CLASS_LABELS[c]}={counts['test'].get(c,0)}" for c in sorted(CLASS_LABELS)]))

    # -----------------------------
    # Overall totals
    # -----------------------------
    total_counts = Counter()
    for fs in fold_summaries.values():
        total_counts.update(fs["counts"]["train"])
        total_counts.update(fs["counts"]["test"])

    print("\n🔢 Overall window count by class (all folds):")
    for cls in sorted(CLASS_LABELS):
        print(f"  Class {cls} ({CLASS_LABELS[cls]}): {total_counts.get(cls,0)} windows")


if __name__ == "__main__":
    main()
