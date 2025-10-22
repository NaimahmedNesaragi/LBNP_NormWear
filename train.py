# train.py
"""
Training script for NormWear LBNP fine-tuning using configs.py and utils.py
"""

import argparse
import time
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import LBNPDataset
from model import NormWearWithHeads
from configs import Config
from utils import (
    compute_regression_scaler_from_train,
    compute_metrics_from_predictions,
    rmse_unscaled
)


def train_one_fold(fold_id: int):
    cfg = Config()

    # --- Directories ---
    run_dir = cfg.OUT_DIR / f"fold_{fold_id}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_csv = run_dir / "training_log.csv"

    # --- Load train/test split ---
    split_path = cfg.SPLIT_DIR / f"train_test_split_fold{fold_id}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Split JSON not found: {split_path}")
    with open(split_path, "r") as f:
        split = json.load(f)
    train_files, test_files = split["train"], split["test"]
    print(f"Fold {fold_id}: train windows={len(train_files)}, test windows={len(test_files)}")

    # --- Compute regression scaler ---
    reg_mean, reg_std = compute_regression_scaler_from_train(cfg.PKL_DIR, train_files)
    with open(run_dir / f"reg_scaler_fold{fold_id}.json", "w") as f:
        json.dump({"reg_mean": reg_mean, "reg_std": reg_std}, f)
    print(f"Regression scaler: mean={reg_mean:.4f}, std={reg_std:.4f}")

    # --- Dataset & DataLoader ---
    train_ds = LBNPDataset(cfg.PKL_DIR, train_files, variates=cfg.VARIATES)
    val_ds = LBNPDataset(cfg.PKL_DIR, test_files, variates=cfg.VARIATES)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, drop_last=False)

    # --- Model, criterion, scaler ---
    model = NormWearWithHeads(pretrained=True, num_classes=cfg.NUM_CLASSES, device=cfg.DEVICE).to(cfg.DEVICE)
    criterion_class = nn.CrossEntropyLoss().to(cfg.DEVICE)
    criterion_reg = nn.MSELoss().to(cfg.DEVICE)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.MIXED_PRECISION)

    # --- Logging setup ---
    with open(logs_csv, "w", newline="") as fh:
        csv.writer(fh).writerow([
            "epoch", "phase",
            "train_loss", "train_class_loss", "train_reg_loss", "train_acc", "train_f1", "train_rmse",
            "val_loss", "val_class_loss", "val_acc", "val_f1", "val_rmse"
        ])
    tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    # ================= Phase 1: Train heads only =================
    print("=== Phase 1: training heads only ===")
    for name, param in model.named_parameters():
        if ("class_head" not in name) and ("reg_head" not in name):
            param.requires_grad = False

    optimizer = torch.optim.AdamW([
        {"params": model.class_head.parameters(), "lr": cfg.LR_HEADS},
        {"params": model.reg_head.parameters(), "lr": cfg.LR_HEADS}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.NUM_EPOCHS_HEADS))

    best_val_acc = -1.0
    early_stop_counter = 0

    for epoch in range(1, cfg.NUM_EPOCHS_HEADS + 1):
        model.train()
        running_loss = running_class_loss = running_reg_loss = 0.0
        preds_all, targets_all = [], []
        preds_reg_all, targets_reg_all = [], []
        optimizer.zero_grad()

        for step, (x, y_class, y_reg) in enumerate(train_loader):
            x = x.to(cfg.DEVICE, non_blocking=True)
            y_class = y_class.to(cfg.DEVICE, non_blocking=True)
            y_reg = y_reg.to(cfg.DEVICE, non_blocking=True)
            y_reg_scaled = (y_reg - reg_mean) / (reg_std + 1e-9)

            with torch.cuda.amp.autocast(enabled=cfg.MIXED_PRECISION):
                out_class, out_reg = model(x)
                loss_class = criterion_class(out_class, y_class)
                loss_reg = criterion_reg(out_reg, y_reg_scaled)
                loss = cfg.CLASS_WEIGHT * loss_class + cfg.REG_WEIGHT * loss_reg
                loss = loss / cfg.ACCUM_STEPS

            scaler.scale(loss).backward()
            running_loss += float(loss.item() * cfg.ACCUM_STEPS)
            running_class_loss += float(loss_class.item())
            running_reg_loss += float(loss_reg.item())

            preds_all.extend(torch.argmax(out_class.detach(), dim=1).cpu().numpy().tolist())
            targets_all.extend(y_class.cpu().numpy().tolist())
            preds_reg_all.extend(out_reg.detach().cpu().numpy().tolist())
            targets_reg_all.extend(y_reg_scaled.detach().cpu().numpy().tolist())

            if (step + 1) % cfg.ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # --- Train metrics ---
        train_acc, train_f1 = compute_metrics_from_predictions(targets_all, preds_all, cfg.NUM_CLASSES)
        train_rmse = rmse_unscaled(preds_reg_all, targets_reg_all, reg_mean, reg_std)
        avg_loss = running_loss / max(1, len(train_loader))
        avg_class_loss = running_class_loss / max(1, len(train_loader))
        avg_reg_loss = running_reg_loss / max(1, len(train_loader))

        # --- Validation ---
        model.eval()
        val_preds, val_targets, val_preds_reg, val_targets_reg = [], [], [], []
        val_loss = val_class_loss = val_reg_loss = 0.0
        with torch.no_grad():
            for x, y_class, y_reg in val_loader:
                x = x.to(cfg.DEVICE, non_blocking=True)
                y_class = y_class.to(cfg.DEVICE, non_blocking=True)
                y_reg = y_reg.to(cfg.DEVICE, non_blocking=True)
                y_reg_scaled = (y_reg - reg_mean) / (reg_std + 1e-9)

                with torch.cuda.amp.autocast(enabled=cfg.MIXED_PRECISION):
                    out_class, out_reg = model(x)
                    loss_class = criterion_class(out_class, y_class)
                    loss_reg = criterion_reg(out_reg, y_reg_scaled)
                    loss = cfg.CLASS_WEIGHT * loss_class + cfg.REG_WEIGHT * loss_reg

                val_loss += float(loss.item())
                val_class_loss += float(loss_class.item())
                val_reg_loss += float(loss_reg.item())

                val_preds.extend(torch.argmax(out_class, dim=1).cpu().numpy().tolist())
                val_targets.extend(y_class.cpu().numpy().tolist())
                val_preds_reg.extend(out_reg.detach().cpu().numpy().tolist())
                val_targets_reg.extend(y_reg_scaled.detach().cpu().numpy().tolist())

        val_acc, val_f1 = compute_metrics_from_predictions(val_targets, val_preds, cfg.NUM_CLASSES)
        val_rmse = rmse_unscaled(val_preds_reg, val_targets_reg, reg_mean, reg_std)
        val_avg_loss = val_loss / max(1, len(val_loader))
        val_avg_class_loss = val_class_loss / max(1, len(val_loader))
        val_avg_reg_loss = val_reg_loss / max(1, len(val_loader))

        print(f"[Fold {fold_id} | Heads] Epoch {epoch}/{cfg.NUM_EPOCHS_HEADS} "
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
                val_avg_loss, val_avg_class_loss, val_acc, val_f1, val_rmse
            ])

        # Save best heads checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / f"best_heads_fold{fold_id}.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping heads at epoch {epoch}")
            break

    # ================= Phase 2: Fine-tune top encoder blocks =================
    print("=== Phase 2: fine-tune top encoder blocks ===")
    heads_ckpt = ckpt_dir / f"best_heads_fold{fold_id}.pth"
    if heads_ckpt.exists():
        model.load_state_dict(torch.load(heads_ckpt, map_location=cfg.DEVICE))
        print(f"✅ Loaded best heads checkpoint from {heads_ckpt}")
    else:
        raise FileNotFoundError(f"Cannot find heads checkpoint: {heads_ckpt}")

    # Unfreeze top encoder blocks
    for name, param in model.named_parameters():
        param.requires_grad = any(block in name for block in cfg.FINETUNE_BLOCK_NAMES)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No parameters are trainable in Phase 2. Check FINETUNE_BLOCK_NAMES.")
    print(f"Trainable params after unfreeze: {sum(p.numel() for p in trainable_params)}")

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.LR_FINETUNE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    best_val_acc = -1.0
    early_stop_counter = 0

    for epoch in range(1, cfg.NUM_EPOCHS_FINETUNE + 1):
        model.train()
        running_loss = running_class_loss = running_reg_loss = 0.0
        preds_all, targets_all = [], []
        preds_reg_all, targets_reg_all = [], []
        optimizer.zero_grad()

        for step, (x, y_class, y_reg) in enumerate(train_loader):
            x = x.to(cfg.DEVICE, non_blocking=True)
            y_class = y_class.to(cfg.DEVICE, non_blocking=True)
            y_reg = y_reg.to(cfg.DEVICE, non_blocking=True)
            y_reg_scaled = (y_reg - reg_mean) / (reg_std + 1e-9)

            with torch.cuda.amp.autocast(enabled=cfg.MIXED_PRECISION):
                out_class, out_reg = model(x)
                loss_class = criterion_class(out_class, y_class)
                loss_reg = criterion_reg(out_reg, y_reg_scaled)
                loss = cfg.CLASS_WEIGHT * loss_class + cfg.REG_WEIGHT * loss_reg
                loss = loss / cfg.ACCUM_STEPS

            scaler.scale(loss).backward()
            running_loss += float(loss.item() * cfg.ACCUM_STEPS)
            running_class_loss += float(loss_class.item())
            running_reg_loss += float(loss_reg.item())

            preds_all.extend(torch.argmax(out_class.detach(), dim=1).cpu().numpy().tolist())
            targets_all.extend(y_class.cpu().numpy().tolist())
            preds_reg_all.extend(out_reg.detach().cpu().numpy().tolist())
            targets_reg_all.extend(y_reg_scaled.detach().cpu().numpy().tolist())

            if (step + 1) % cfg.ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # --- Train metrics ---
        train_acc, train_f1 = compute_metrics_from_predictions(targets_all, preds_all, cfg.NUM_CLASSES)
        train_rmse = rmse_unscaled(preds_reg_all, targets_reg_all, reg_mean, reg_std)
        avg_loss = running_loss / max(1, len(train_loader))
        avg_class_loss = running_class_loss / max(1, len(train_loader))
        avg_reg_loss = running_reg_loss / max(1, len(train_loader))

        # --- Validation ---
        model.eval()
        val_preds, val_targets, val_preds_reg, val_targets_reg = [], [], [], []
        val_loss = val_class_loss = val_reg_loss = 0.0
        with torch.no_grad():
            for x, y_class, y_reg in val_loader:
                x = x.to(cfg.DEVICE, non_blocking=True)
                y_class = y_class.to(cfg.DEVICE, non_blocking=True)
                y_reg = y_reg.to(cfg.DEVICE, non_blocking=True)
                y_reg_scaled = (y_reg - reg_mean) / (reg_std + 1e-9)

                with torch.cuda.amp.autocast(enabled=cfg.MIXED_PRECISION):
                    out_class, out_reg = model(x)
                    loss_class = criterion_class(out_class, y_class)
                    loss_reg = criterion_reg(out_reg, y_reg_scaled)
                    loss = cfg.CLASS_WEIGHT * loss_class + cfg.REG_WEIGHT * loss_reg

                val_loss += float(loss.item())
                val_class_loss += float(loss_class.item())
                val_reg_loss += float(loss_reg.item())

                val_preds.extend(torch.argmax(out_class, dim=1).cpu().numpy().tolist())
                val_targets.extend(y_class.cpu().numpy().tolist())
                val_preds_reg.extend(out_reg.detach().cpu().numpy().tolist())
                val_targets_reg.extend(y_reg_scaled.detach().cpu().numpy().tolist())

        val_acc, val_f1 = compute_metrics_from_predictions(val_targets, val_preds, cfg.NUM_CLASSES)
        val_rmse = rmse_unscaled(val_preds_reg, val_targets_reg, reg_mean, reg_std)
        val_avg_loss = val_loss / max(1, len(val_loader))
        val_avg_class_loss = val_class_loss / max(1, len(val_loader))
        val_avg_reg_loss = val_reg_loss / max(1, len(val_loader))

        print(f"[Fold {fold_id} | FineTune] Epoch {epoch}/{cfg.NUM_EPOCHS_FINETUNE} "
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
                val_avg_loss, val_avg_class_loss, val_acc, val_f1, val_rmse
            ])

        # --- Save best checkpoint & early stopping ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / f"best_finetune_fold{fold_id}.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        scheduler.step(val_avg_loss)

        if early_stop_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping fine-tune at epoch {epoch}")
            break

    print(f"Fold {fold_id} fine-tuning complete. Best val_acc={best_val_acc:.4f}")

    tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold ID to train")
    args = parser.parse_args()
    train_one_fold(args.fold)
