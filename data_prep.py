#!/usr/bin/env python3
"""
data_prep.py

Single-threaded data preparation for NormWear-compatible LBNP dataset.

Outputs:
 - per-window .pkl files with keys: uid, data_raw, cwt, sampling_rate, label, meta
 - train_test_split_fold{1..3}.json (only for the requested fold)
 - scaler_fold{1..3}.json (per-fold channel mean/std computed on training windows)
 - metadata_fold{FOLD}.json

Usage:
 python data_prep.py --fold 1
"""

import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, filtfilt, firwin
import torch
import torch.nn.functional as F
import argparse
import time
import random

# --------------------------------------
# CONFIGURATION
# --------------------------------------
CSV_DIR = Path("/home/naim/LBNP/level7/csv_export")
OUTPUT_DIR = Path("/home/naim/LBNP/level7/pkl_windows_normwear_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ["LBNP", "EKG_SD50", "Photopleth", "ABP_rd_ledning", "Nexfin_ABP", "EKG_kopi"]
LABEL_COLUMN = "Nexfin_SV"  # regression target

FS_ORIG = 1000
FS_NEW = 64
WINDOW_SEC = 15
WINDOW_SAMPLES = int(WINDOW_SEC * FS_NEW)
TRAIN_OVERLAP_SEC = 10
TRAIN_STEP = WINDOW_SAMPLES - int(TRAIN_OVERLAP_SEC * FS_NEW)
TEST_STEP = WINDOW_SAMPLES

SV_LAG_SEC = 10
SV_LAG_SAMPLES = int(SV_LAG_SEC * FS_NEW)

CWT_LOW = 1
CWT_HIGH = 65
CWT_STEP = 1
CWT_WLEN = 100

LP_CUTOFF = 25.0
LP_ORDER = 101

#USE_CUDA = True
#DEVICE = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
USE_CUDA = True
if USE_CUDA and torch.cuda.is_available():
    DEVICE = torch.device("cuda:2")  # e.g., GPU #2
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# deterministic subject-wise folds
SUBJECTS = list(range(1, 24))  # 1..23

# split 23 subjects into 3 non-overlapping test sets
random.seed(42)
shuffled_subjects = SUBJECTS.copy()
random.shuffle(shuffled_subjects)
TEST_SPLITS = [
    shuffled_subjects[0:7],
    shuffled_subjects[7:15],
    shuffled_subjects[15:23]
]
FOLDS = {}
for i, test_subs in enumerate(TEST_SPLITS, start=1):
    train_subs = [s for s in SUBJECTS if s not in test_subs]
    FOLDS[i] = {"train": train_subs, "test": test_subs}

# output dirs
PKL_DIR = OUTPUT_DIR / "pkls"
PKL_DIR.mkdir(parents=True, exist_ok=True)
SCALER_DIR = OUTPUT_DIR / "scalers"
SCALER_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR = OUTPUT_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------
# CLI
# --------------------------------------
parser = argparse.ArgumentParser(description="LBNP Data Preparation Script")
parser.add_argument("--fold", type=int, default=1, choices=[1,2,3],
                    help="Select which fold (1,2,3) to generate")
args = parser.parse_args()
FOLD_ID = args.fold

print(f"\n🔹 Preparing data for Fold {FOLD_ID} (train subjects={FOLDS[FOLD_ID]['train']}, test subjects={FOLDS[FOLD_ID]['test']})\n")

# --------------------------------------
# Helpers: signal processing & CWT
# --------------------------------------
def lowpass_filter_array(x, fs=FS_ORIG, cutoff=LP_CUTOFF, order=LP_ORDER):
    if order % 2 == 0:
        order += 1
    nyq = 0.5 * fs
    b = firwin(order, cutoff / nyq, window='hamming')
    try:
        y = filtfilt(b, [1.0], x, axis=-1)
    except Exception:
        y = np.array([np.convolve(ch, b, mode='same') for ch in x])
    return y

def resample_and_align(x, labels, sv, fs_orig=FS_ORIG, fs_new=FS_NEW):
    x_f = lowpass_filter_array(x, fs=fs_orig)
    x_res = resample_poly(x_f, up=fs_new, down=fs_orig, axis=1)
    labels_res = resample_poly(labels.astype(float), up=fs_new, down=fs_orig)
    sv_res = resample_poly(sv.astype(float), up=fs_new, down=fs_orig)
    return x_res, labels_res, sv_res

def interpolate_nans(x_array):
    x_interp = np.copy(x_array)
    for i in range(x_array.shape[0]):
        ch = x_array[i]
        nans = np.isnan(ch)
        if np.any(nans):
            not_nan = np.where(~nans)[0]
            if len(not_nan) == 0:
                x_interp[i] = 0.0
            else:
                x_interp[i, nans] = np.interp(np.where(nans)[0], not_nan, ch[not_nan])
    return x_interp

def ricker_wavelet(points, scale, device=torch.device('cpu')):
    A = 2 / (torch.sqrt(3*scale) * torch.pi**0.25)
    wsq = scale**2
    vec = torch.arange(0, points, device=device) - (points-1)/2
    xsq = vec**2
    mod = (1 - xsq / wsq)
    gauss = torch.exp(-xsq / (2*wsq))
    wavelet = A * mod * gauss
    return wavelet

def cwt_ricker_torch(x, lowest_scale=CWT_LOW, largest_scale=CWT_HIGH, step=CWT_STEP, wavelet_len=CWT_WLEN, device=torch.device('cpu')):
    batch_size, seq_len = x.shape
    scales = torch.arange(lowest_scale, largest_scale + step, step, device=device, dtype=torch.float32)
    wlen = min(10*largest_scale, seq_len)
    wavelets = torch.stack([ricker_wavelet(wlen, s, device=device) for s in scales]).view(len(scales),1,wlen)
    x_in = x.unsqueeze(1)
    pad = wlen // 2
    cwt_out = F.conv1d(x_in, weight=wavelets, padding=pad)
    return cwt_out

def cwt_wrap_torch(x, lowest_scale=CWT_LOW, largest_scale=CWT_HIGH, step=CWT_STEP, wavelet_len=CWT_WLEN, device=torch.device('cpu')):
    d1 = x[:,1:] - x[:,:-1]
    d2 = d1[:,1:] - d1[:,:-1]
    x_stack = torch.stack([x[:,2:], d1[:,1:], d2]).permute(1,0,2).float()
    bn, n_, new_L = x_stack.shape
    x_for = x_stack.reshape(bn*n_, new_L).to(device)
    cwt_res = cwt_ricker_torch(x_for, lowest_scale, largest_scale, step=step, wavelet_len=wavelet_len, device=device)
    _, n_scales, seq_len = cwt_res.shape
    out = cwt_res.reshape(bn, n_, n_scales, seq_len).permute(0,1,3,2)
    return out

def level_to_class(level):
    try:
        lvl = int(level)
    except Exception:
        lvl = int(round(float(level)))
    if lvl <= 2:
        return 1
    elif lvl <= 4:
        return 2
    else:
        return 3

# --------------------------------------
# Process CSV
# --------------------------------------
def process_single_csv(csv_path, channels, output_dir, fold_id, fold_cfg, device=torch.device('cpu')):
    df = pd.read_csv(csv_path)
    stem = csv_path.stem
    try:
        parts = stem.split("_")
        subj = int(parts[1])
        trial = int(parts[3])
    except Exception:
        nums = [int(s) for s in stem.split("_") if s.isdigit()]
        subj = nums[0] if len(nums)>0 else -1
        trial = nums[1] if len(nums)>1 else -1

    if subj not in fold_cfg["train"] and subj not in fold_cfg["test"]:
        return []

    phase = "train" if subj in fold_cfg["train"] else "test"

    # channel arrays
    missing = [ch for ch in channels if ch not in df.columns]
    if missing:
        print(f"⚠️ WARNING: Missing channels in {csv_path.name}: {missing}. Filling zeros.")
    arrays = [df[ch].to_numpy(dtype=float) if ch in df.columns else np.zeros(len(df),dtype=float) for ch in channels]
    x = np.stack(arrays, axis=0)

    labels = df["Labels"].to_numpy() if "Labels" in df.columns else np.zeros(len(df),dtype=int)
    sv = df[LABEL_COLUMN].to_numpy() if LABEL_COLUMN in df.columns else np.zeros(len(df),dtype=float)

    x_res, labels_res, sv_res = resample_and_align(x, labels, sv)
    x_res = interpolate_nans(x_res)
    sv_res = np.nan_to_num(sv_res, nan=np.nanmean(sv_res) if np.any(~np.isnan(sv_res)) else 0.0)
    labels_res = np.nan_to_num(labels_res, nan=1).astype(int)

    # SV lag
    if SV_LAG_SAMPLES > 0:
        sv_shifted = np.roll(sv_res, -SV_LAG_SAMPLES)
        if SV_LAG_SAMPLES < len(sv_shifted):
            sv_shifted[-SV_LAG_SAMPLES:] = np.nan
            last_valid = np.nanmean(sv_res) if np.any(~np.isnan(sv_res)) else 0.0
            sv_shifted = np.nan_to_num(sv_shifted, nan=last_valid)
        else:
            sv_shifted = np.nan_to_num(sv_shifted, nan=np.nanmean(sv_res) if np.any(~np.isnan(sv_res)) else 0.0)
    else:
        sv_shifted = sv_res

    n_samples = x_res.shape[1]
    saved = []
    start = 0
    win_idx = 0
    step = TRAIN_STEP if phase=="train" else TEST_STEP

    while start + WINDOW_SAMPLES <= n_samples:
        end = start + WINDOW_SAMPLES
        x_win_raw = x_res[:,start:end]
        label_win_mode = int(np.bincount(labels_res[start:end].astype(int)).argmax())
        class_label = level_to_class(label_win_mode)
        sv_mean = float(np.mean(sv_shifted[start:end]))

        cwt_list = []
        for ch_idx in range(x_win_raw.shape[0]):
            series = torch.tensor(x_win_raw[ch_idx,:], dtype=torch.float32, device=device).unsqueeze(0)
            cwt_ch = cwt_wrap_torch(series, device=device)
            cwt_list.append(cwt_ch[0].cpu().numpy())
        cwt_stack = np.stack(cwt_list, axis=0)

        uid = f"Subj_{subj}_Ser_{trial}_win_{win_idx:04d}"
        out_obj = {
            "uid": uid,
            "data_raw": x_win_raw.astype(np.float16),
            "cwt": cwt_stack.astype(np.float16),
            "sampling_rate": FS_NEW,
            "label": [{"class": int(class_label)}, {"reg": float(sv_mean)}],
            "meta": {"subj": int(subj), "trial": int(trial), "start_sample_resampled": int(start),
                     "fold": int(fold_id), "phase": phase}
        }

        out_path = PKL_DIR / f"{uid}.pkl"
        with open(out_path,"wb") as fh:
            pickle.dump(out_obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

        saved.append((subj, out_path.name, fold_id, phase))
        win_idx += 1
        start += step

    print(f"[Fold {fold_id} | {phase}] {csv_path.name} -> saved {win_idx} windows (Subj {subj} Trial {trial})")
    return saved

# --------------------------------------
# Main orchestration
# --------------------------------------
def main():
    t0 = time.time()
    csv_files = sorted(list(CSV_DIR.glob("Subj_*_Ser_*_app.csv")))
    print(f"Found {len(csv_files)} CSV files in {CSV_DIR}")

    fold_cfg = FOLDS[FOLD_ID]
    allowed_subs = set(fold_cfg["train"] + fold_cfg["test"])
    all_saved = []

    for csv_file in csv_files:
        try:
            subj_id = int(csv_file.stem.split("_")[1])
        except Exception:
            nums = [int(s) for s in csv_file.stem.split("_") if s.isdigit()]
            subj_id = nums[0] if nums else None
        if subj_id is None or subj_id not in allowed_subs:
            continue
        saved = process_single_csv(csv_file, CHANNELS, PKL_DIR, FOLD_ID, fold_cfg, DEVICE)
        all_saved.extend(saved)

    # train/test file lists
    fold_file_map = {"train": [], "test": []}
    for subj, fname, fold_id, phase in all_saved:
        if fold_id == FOLD_ID:
            fold_file_map[phase].append(fname)

    split_json = {"fold": FOLD_ID, "train": fold_file_map["train"], "test": fold_file_map["test"]}
    split_path = SPLIT_DIR / f"train_test_split_fold{FOLD_ID}.json"
    with open(split_path,"w") as fh:
        json.dump(split_json, fh, indent=2)
    print(f"Saved split JSON: {split_path.name} (train={len(fold_file_map['train'])}, test={len(fold_file_map['test'])})")

    # scaler computation
    train_files = fold_file_map["train"]
    if len(train_files) == 0:
        print(f"⚠️ No training windows for fold {FOLD_ID}, skipping scaler computation.")
    else:
        channel_sums = None
        channel_sumsq = None
        count = 0
        for fname in train_files:
            pkl_path = PKL_DIR / fname
            with open(pkl_path,"rb") as fh:
                obj = pickle.load(fh)
            data_raw = np.array(obj["data_raw"], dtype=np.float64)
            if channel_sums is None:
                channel_sums = np.zeros(data_raw.shape[0],dtype=np.float64)
                channel_sumsq = np.zeros(data_raw.shape[0],dtype=np.float64)
            channel_sums += data_raw.mean(axis=1)
            channel_sumsq += (data_raw.std(axis=1)**2)
            count += 1
        means = (channel_sums / float(count)).tolist()
        stds = (np.sqrt(channel_sumsq / float(count))).tolist()
        scaler = {"mean": means, "std": stds}
        sfile = SCALER_DIR / f"scaler_fold{FOLD_ID}.json"
        with open(sfile,"w") as fh:
            json.dump(scaler, fh, indent=2)
        print(f"Saved scaler for fold {FOLD_ID} -> {sfile.name} (channels={len(means)})")

    # metadata
    metadata = {
        "fs_new": FS_NEW,
        "window_sec": WINDOW_SEC,
        "window_samples": WINDOW_SAMPLES,
        "train_overlap_sec": TRAIN_OVERLAP_SEC,
        "train_step_samples": TRAIN_STEP,
        "test_step_samples": TEST_STEP,
        "sv_lag_sec": SV_LAG_SEC,
        "channels": CHANNELS,
        "cwt": {"lowest": CWT_LOW, "largest": CWT_HIGH, "step": CWT_STEP, "wavelet_len": CWT_WLEN},
        "fold": FOLD_ID,
        "n_pkl": len(all_saved),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    meta_path = OUTPUT_DIR / f"metadata_fold{FOLD_ID}.json"
    with open(meta_path,"w") as fh:
        json.dump(metadata, fh, indent=2)

    t1 = time.time()
    print(f"\n✅ Done for Fold {FOLD_ID}. Saved {len(all_saved)} windows total.")
    print(f"Metadata -> {meta_path.name}")
    print(f"Elapsed: {t1-t0:.1f}s\n")

if __name__ == "__main__":
    main()
