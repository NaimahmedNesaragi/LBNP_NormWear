# train.py
"""
Training script for NormWear LBNP fine-tuning.

Features:
- Uses precomputed CWT from .pkl via LBNPDataset -> x: [B, nvar, 3, L, F]
- Phase 1: train classification & regression heads only, with validation & best-heads checkpoint
- Phase 2: fine-tune top encoder blocks (8..11), starting from best Phase 1 checkpoint
- Regression target scaled using training-set mean/std (computed per fold)
- Mixed precision (torch.cuda.amp), gradient accumulation support
- Metrics: accuracy, macro F1 (from confusion matrix), RMSE (unscaled)
- Checkpointing and TensorBoard logging
- CSV logging

Usage:
    python train.py --fold 1
"""

import json
import csv
import math
import argparse
from pathlib import Path
import time
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import your dataset and model
from dataset import LBNPDataset
from model import NormWearWithHeads

# ----------------------
# CONFIG
# ----------------------
PROJECT_ROOT = Path("/home/naim/LBNP/level7")
PKL_DIR = PROJECT_ROOT / "pkl_windows_normwear_v3" / "pkls"
SPLIT_DIR = PROJECT_ROOT / "pkl_windows_normwear_v3" / "splits"
OUT_DIR = PROJECT_ROOT / "training_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIATES = ["Photopleth", "EKG_kopi", "ABP_rd_ledning", "Nexfin_ABP"]
NUM_CLASSES = 3

BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True

LR_HEADS = 1e-3
LR_FINETUNE = 3e-5
NUM_EPOCHS_HEADS = 10
NUM_EPOCHS_FINETUNE = 24

CLASS_WEIGHT = 1.0
REG_WEIGHT = 0.5

ACCUM_STEPS = 2
MIXED_PRECISION = True

USE_CUDA = True
DEVICE = torch.device("cuda:2") if USE_CUDA and torch.cuda.is_available() else torch.device("cpu")
print("Using device:", DEVICE)

FINETUNE_BLOCK_NAMES = [
    "normwear.encoder_blocks.6",
    "normwear.encoder_blocks.7",
    "normwear.encoder_blocks.8",
    "normwear.encoder_blocks.9",
    "normwear.encoder_blocks.10",
    "normwear.encoder_blocks.11"
]

BEST_METRIC = "val_acc"
SAVE_EVERY = 1
EARLY_STOPPING_PATIENCE = 8

# ----------------------
# Helper functions
# ----------------------
def compute_regression_scaler_from_train(pkls_dir, train_files):
    regs = []
    for fname in train_files:
        p = pkls_dir / fname
        try:
            with open(p, "rb") as fh:
                obj = json.load(fh) if p.suffix == ".json" else None
        except Exception:
            obj = None
        if obj is None:
            import pickle
            with open(p, "rb") as fh:
                obj = pickle.load(fh)
        reg = float(obj["label"][1]["reg"])
        regs.append(reg)
    regs = np.array(regs, dtype=np.float64)
    mean = float(np.mean(regs))
    std = float(np.std(regs)) if np.std(regs) > 0 else 1.0
    return mean, std

def compute_metrics_from_predictions(all_targets, all_preds, num_classes):
    N = len(all_targets)
    if N == 0:
        return 0.0, 0.0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_targets, all_preds):
        cm[int(t), int(p)] += 1
    acc = np.trace(cm) / float(np.sum(cm))
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    return float(acc), float(macro_f1)

def rmse_unscaled(preds_scaled, targets_scaled, reg_mean, reg_std):
    preds = preds_scaled * reg_std + reg_mean
    targets = targets_scaled * reg_std + reg_mean
    mse = np.mean((preds - targets) ** 2) if len(preds) > 0 else 0.0
    return float(np.sqrt(mse))

# ----------------------
# Training loop
# ----------------------
def train_one_fold(fold_id, args):
    run_dir = OUT_DIR / f"fold_{fold_id}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_csv = run_dir / "training_log.csv"

    split_path = SPLIT_DIR / f"train_test_split_fold{fold_id}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split JSON not found: {split_path}")
    with open(split_path, "r") as fh:
        split = json.load(fh)
    train_files = split["train"]
    test_files = split["test"]

    print(f"Fold {fold_id}: train windows={len(train_files)}, test windows={len(test_files)}")

    reg_mean, reg_std = compute_regression_scaler_from_train(PKL_DIR, train_files)
    with open(run_dir / f"reg_scaler_fold{fold_id}.json", "w") as fh:
        json.dump({"reg_mean": reg_mean, "reg_std": reg_std}, fh, indent=2)
    print(f"Regression scaler: mean={reg_mean:.4f}, std={reg_std:.4f}")

    train_ds = LBNPDataset(PKL_DIR, train_files, variates=VARIATES)
    val_ds = LBNPDataset(PKL_DIR, test_files, variates=VARIATES)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)

    model = NormWearWithHeads(pretrained=True, num_classes=NUM_CLASSES, device=DEVICE).to(DEVICE)
    criterion_class = nn.CrossEntropyLoss().to(DEVICE)
    criterion_reg = nn.MSELoss().to(DEVICE)
    scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION)

    # Logging
    with open(logs_csv, "w", newline="") as fh:
        csv.writer(fh).writerow([
            "epoch", "phase",
            "train_loss", "train_class_loss", "train_reg_loss", "train_acc", "train_f1", "train_rmse",
            "val_loss", "val_class_loss", "val_reg_loss", "val_acc", "val_f1", "val_rmse"
        ])
    tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    # ---------------- Phase 1: Heads ----------------
    for name, param in model.named_parameters():
        if ("class_head" not in name) and ("reg_head" not in name):
            param.requires_grad = False
    optimizer = torch.optim.AdamW([
        {"params": model.class_head.parameters(), "lr": LR_HEADS},
        {"params": model.reg_head.parameters(), "lr": LR_HEADS}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, NUM_EPOCHS_HEADS))

    best_val_acc = -1.0
    best_epoch = -1
    early_stop_counter = 0
    print("=== Phase 1: training heads only ===")
    for epoch in range(1, NUM_EPOCHS_HEADS + 1):
        model.train()
        running_loss = running_class_loss = running_reg_loss = 0.0
        preds_all = []
        targets_all = []
        preds_reg_all = []
        targets_reg_all = []
        optimizer.zero_grad()

        for step, (x, y_class, y_reg) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y_class = y_class.to(DEVICE, non_blocking=True)
            y_reg = y_reg.to(DEVICE, non_blocking=True)
            y_reg_scaled = (y_reg - reg_mean) / (reg_std + 1e-9)

            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                out_class, out_reg = model(x)
                loss_class = criterion_class(out_class, y_class)
                loss_reg = criterion_reg(out_reg, y_reg_scaled)
                loss = CLASS_WEIGHT * loss_class + REG_WEIGHT * loss_reg
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            running_loss += float(loss.item() * ACCUM_STEPS)
            running_class_loss += float(loss_class.item())
            running_reg_loss += float(loss_reg.item())
            preds_all.extend(torch.argmax(out_class.detach(), dim=1).cpu().numpy().tolist())
            targets_all.extend(y_class.cpu().numpy().tolist())
            preds_reg_all.extend(out_reg.detach().cpu().numpy().tolist())
            targets_reg_all.extend(y_reg_scaled.detach().cpu().numpy().tolist())

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()
        train_acc, train_f1 = compute_metrics_from_predictions(np.array(targets_all), np.array(preds_all), NUM_CLASSES)
        train_rmse = rmse_unscaled(np.array(preds_reg_all), np.array(targets_reg_all), reg_mean, reg_std)
        avg_loss = running_loss / max(1, len(train_loader))
        avg_class_loss = running_class_loss / max(1, len(train_loader))
        avg_reg_loss = running_reg_loss / max(1, len(train_loader))

        # print(f"[Fold {fold_id} | Heads] Epoch {epoch}/{NUM_EPOCHS_HEADS} "
        #       f"loss={avg_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} rmse_unscaled={train_rmse:.4f}")
        



        # --- Phase 1 Validation ---
        model.eval()
        val_preds = []
        val_targets = []
        val_preds_reg = []
        val_targets_reg = []
        val_loss = val_class_loss = val_reg_loss = 0.0
        with torch.no_grad():
            for x, y_class, y_reg in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                y_class = y_class.to(DEVICE, non_blocking=True)
                y_reg = y_reg.to(DEVICE, non_blocking=True)
                y_reg_scaled = (y_reg - reg_mean) / (reg_std + 1e-9)

                with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                    out_class, out_reg = model(x)
                    loss_class = criterion_class(out_class, y_class)
                    loss_reg = criterion_reg(out_reg, y_reg_scaled)
                    loss = CLASS_WEIGHT * loss_class + REG_WEIGHT * loss_reg

                val_loss += float(loss.item())
                val_class_loss += float(loss_class.item())
                val_reg_loss += float(loss_reg.item())

                val_preds.extend(torch.argmax(out_class, dim=1).cpu().numpy().tolist())
                val_targets.extend(y_class.cpu().numpy().tolist())
                val_preds_reg.extend(out_reg.detach().cpu().numpy().tolist())
                val_targets_reg.extend(y_reg_scaled.detach().cpu().numpy().tolist())

        val_acc, val_f1 = compute_metrics_from_predictions(np.array(val_targets), np.array(val_preds), NUM_CLASSES)
        val_rmse = rmse_unscaled(np.array(val_preds_reg), np.array(val_targets_reg), reg_mean, reg_std)
        val_avg_loss = val_loss / max(1, len(val_loader))
        val_avg_class_loss = val_class_loss / max(1, len(val_loader))
        val_avg_reg_loss = val_reg_loss / max(1, len(val_loader))

        print(f"[Fold {fold_id} | Heads] Epoch {epoch}/{NUM_EPOCHS_HEADS} "
                f"train_loss={avg_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} train_rmse={train_rmse:.4f} "
                f"val_loss={val_avg_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_rmse={val_rmse:.4f}")

        tb_writer.add_scalar("Heads/train_loss", avg_loss, epoch)
        tb_writer.add_scalar("Heads/train_acc", train_acc, epoch)
        tb_writer.add_scalar("Heads/train_f1", train_f1, epoch)
        tb_writer.add_scalar("Heads/train_rmse_unscaled", train_rmse, epoch)
        tb_writer.add_scalar("Heads/val_loss", val_avg_loss, epoch)
        tb_writer.add_scalar("Heads/val_acc", val_acc, epoch)
        tb_writer.add_scalar("Heads/val_f1", val_f1, epoch)
        tb_writer.add_scalar("Heads/val_rmse_unscaled", val_rmse, epoch)

        with open(logs_csv, "a", newline="") as fh:
            csv.writer(fh).writerow([
                epoch, "heads",
                avg_loss, avg_class_loss, avg_reg_loss, train_acc, train_f1, train_rmse,
                val_avg_loss, val_avg_class_loss, val_avg_reg_loss, val_acc, val_f1, val_rmse
            ])

        # Save best heads checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), ckpt_dir / f"best_heads_fold{fold_id}.pth")
        else:
            early_stop_counter += 1

    # ---------------- Phase 2: Fine-tune ----------------
    print("=== Phase 2: fine-tune top encoder blocks ===")
    # Load best Phase 1 checkpoint
    heads_ckpt = ckpt_dir / f"best_heads_fold{fold_id}.pth"
    if heads_ckpt.exists():
        model.load_state_dict(torch.load(heads_ckpt, map_location=DEVICE))

    # Unfreeze top encoder blocks
    for name, param in model.named_parameters():
        if any(b in name for b in FINETUNE_BLOCK_NAMES):
            param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params after unfreeze: {sum([p.numel() for p in trainable_params])}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LR_FINETUNE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_acc = -1.0
    early_stop_counter = 0

    for epoch in range(1, NUM_EPOCHS_FINETUNE + 1):
        model.train()
        running_loss = running_class_loss = running_reg_loss = 0.0
        preds_all = []
        targets_all = []
        preds_reg_all = []
        targets_reg_all = []
        optimizer.zero_grad()

        for step, (x, y_class, y_reg) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y_class = y_class.to(DEVICE, non_blocking=True)
            y_reg = y_reg.to(DEVICE, non_blocking=True)
            y_reg_scaled = (y_reg - reg_mean) / (reg_std + 1e-9)

            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                out_class, out_reg = model(x)
                loss_class = criterion_class(out_class, y_class)
                loss_reg = criterion_reg(out_reg, y_reg_scaled)
                loss = CLASS_WEIGHT * loss_class + REG_WEIGHT * loss_reg
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            running_loss += float(loss.item() * ACCUM_STEPS)
            running_class_loss += float(loss_class.item())
            running_reg_loss += float(loss_reg.item())

            preds_all.extend(torch.argmax(out_class.detach(), dim=1).cpu().numpy().tolist())
            targets_all.extend(y_class.cpu().numpy().tolist())
            preds_reg_all.extend(out_reg.detach().cpu().numpy().tolist())
            targets_reg_all.extend(y_reg_scaled.detach().cpu().numpy().tolist())

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        train_acc, train_f1 = compute_metrics_from_predictions(np.array(targets_all), np.array(preds_all), NUM_CLASSES)
        train_rmse = rmse_unscaled(np.array(preds_reg_all), np.array(targets_reg_all), reg_mean, reg_std)
        avg_loss = running_loss / max(1, len(train_loader))
        avg_class_loss = running_class_loss / max(1, len(train_loader))
        avg_reg_loss = running_reg_loss / max(1, len(train_loader))

        #print(f"[Fold {fold_id} | FineTune] Epoch {epoch}/{NUM_EPOCHS_FINETUNE} "
              #f"loss={avg_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} rmse_unscaled={train_rmse:.4f}")




        # --- Validation at epoch end ---
        model.eval()
        val_preds = []
        val_targets = []
        val_preds_reg = []
        val_targets_reg = []
        val_loss = val_class_loss = val_reg_loss = 0.0
        with torch.no_grad():
            for x, y_class, y_reg in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                y_class = y_class.to(DEVICE, non_blocking=True)
                y_reg = y_reg.to(DEVICE, non_blocking=True)
                y_reg_scaled = (y_reg - reg_mean) / (reg_std + 1e-9)

                with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                    out_class, out_reg = model(x)
                    loss_class = criterion_class(out_class, y_class)
                    loss_reg = criterion_reg(out_reg, y_reg_scaled)
                    loss = CLASS_WEIGHT * loss_class + REG_WEIGHT * loss_reg

                val_loss += float(loss.item())
                val_class_loss += float(loss_class.item())
                val_reg_loss += float(loss_reg.item())

                val_preds.extend(torch.argmax(out_class, dim=1).cpu().numpy().tolist())
                val_targets.extend(y_class.cpu().numpy().tolist())
                val_preds_reg.extend(out_reg.detach().cpu().numpy().tolist())
                val_targets_reg.extend(y_reg_scaled.detach().cpu().numpy().tolist())

        val_acc, val_f1 = compute_metrics_from_predictions(np.array(val_targets), np.array(val_preds), NUM_CLASSES)
        val_rmse = rmse_unscaled(np.array(val_preds_reg), np.array(val_targets_reg), reg_mean, reg_std)
        val_avg_loss = val_loss / max(1, len(val_loader))
        val_avg_class_loss = val_class_loss / max(1, len(val_loader))
        val_avg_reg_loss = val_reg_loss / max(1, len(val_loader))

        print(f"[Fold {fold_id} | FineTune] Epoch {epoch}/{NUM_EPOCHS_FINETUNE} "
                f"train_loss={avg_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} train_rmse={train_rmse:.4f} "
                f"val_loss={val_avg_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_rmse={val_rmse:.4f}")

        tb_writer.add_scalar("FineTune/train_loss", avg_loss, epoch)
        tb_writer.add_scalar("FineTune/train_acc", train_acc, epoch)
        tb_writer.add_scalar("FineTune/train_f1", train_f1, epoch)
        tb_writer.add_scalar("FineTune/train_rmse_unscaled", train_rmse, epoch)
        tb_writer.add_scalar("FineTune/val_loss", val_avg_loss, epoch)
        tb_writer.add_scalar("FineTune/val_acc", val_acc, epoch)
        tb_writer.add_scalar("FineTune/val_f1", val_f1, epoch)
        tb_writer.add_scalar("FineTune/val_rmse_unscaled", val_rmse, epoch)

        with open(logs_csv, "a", newline="") as fh:
            csv.writer(fh).writerow([
                epoch, "finetune",
                avg_loss, avg_class_loss, avg_reg_loss, train_acc, train_f1, train_rmse,
                val_avg_loss, val_avg_class_loss, val_avg_reg_loss, val_acc, val_f1, val_rmse
            ])

        # Checkpointing best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / f"best_finetune_fold{fold_id}.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        scheduler.step(val_avg_loss)

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Fold {fold_id} training complete. Best val_acc={best_val_acc:.4f}")
    tb_writer.close()

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold ID to train")
    args = parser.parse_args()
    train_one_fold(args.fold, args)
