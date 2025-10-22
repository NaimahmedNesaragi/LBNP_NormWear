# sanity_check_train.py
"""
Quick sanity check for train.py loops.
Runs one batch through train and validation to verify data, model, and losses.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LBNPDataset
from model import NormWearWithHeads
from configs import Config
from utils import compute_regression_scaler_from_train, compute_metrics_from_predictions, rmse_unscaled

def sanity_check(fold_id=3):
    cfg = Config()

    # Load split
    split_path = cfg.SPLIT_DIR / f"train_test_split_fold{fold_id}.json"
    split = cfg.load_json(split_path)
    train_files, test_files = split["train"], split["test"]

    # Regression scaler
    reg_mean, reg_std = compute_regression_scaler_from_train(cfg.PKL_DIR, train_files)
    print(f"Regression scaler: mean={reg_mean:.4f}, std={reg_std:.4f}")

    # Datasets
    train_ds = LBNPDataset(cfg.PKL_DIR, train_files, variates=cfg.VARIATES)
    val_ds = LBNPDataset(cfg.PKL_DIR, test_files, variates=cfg.VARIATES)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    device = cfg.DEVICE
    model = NormWearWithHeads(pretrained=True, num_classes=cfg.NUM_CLASSES, device=device).to(device)
    criterion_class = nn.CrossEntropyLoss().to(device)
    criterion_reg = nn.MSELoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.MIXED_PRECISION)

    # Grab one batch
    x, y_class, y_reg = next(iter(train_loader))
    x, y_class, y_reg = x.to(device), y_class.to(device), y_reg.to(device)
    y_reg_scaled = (y_reg - reg_mean) / (reg_std + 1e-9)

    # Forward pass
    with torch.cuda.amp.autocast(enabled=cfg.MIXED_PRECISION):
        out_class, out_reg = model(x)
        loss_class = criterion_class(out_class, y_class)
        loss_reg = criterion_reg(out_reg, y_reg_scaled)
        loss = cfg.CLASS_WEIGHT * loss_class + cfg.REG_WEIGHT * loss_reg

    print("Train batch forward pass successful")
    print(f"Class loss: {loss_class.item():.4f}, Reg loss: {loss_reg.item():.4f}, Total loss: {loss.item():.4f}")

    # Compute metrics
    preds_class = torch.argmax(out_class, dim=1).cpu().numpy()
    preds_reg = out_reg.detach().cpu().numpy()
    targets_class = y_class.cpu().numpy()
    targets_reg = y_reg_scaled.cpu().numpy()

    acc, f1 = compute_metrics_from_predictions(targets_class, preds_class, cfg.NUM_CLASSES)
    rmse = rmse_unscaled(preds_reg, targets_reg, reg_mean, reg_std)

    print(f"Train batch metrics -> Accuracy: {acc:.4f}, Macro F1: {f1:.4f}, RMSE: {rmse:.4f}")

    # Validation batch
    x_val, y_class_val, y_reg_val = next(iter(val_loader))
    x_val, y_class_val, y_reg_val = x_val.to(device), y_class_val.to(device), y_reg_val.to(device)
    y_reg_val_scaled = (y_reg_val - reg_mean) / (reg_std + 1e-9)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=cfg.MIXED_PRECISION):
            out_class_val, out_reg_val = model(x_val)
            loss_class_val = criterion_class(out_class_val, y_class_val)
            loss_reg_val = criterion_reg(out_reg_val, y_reg_val_scaled)
            loss_val = cfg.CLASS_WEIGHT * loss_class_val + cfg.REG_WEIGHT * loss_reg_val

    print("Validation batch forward pass successful")
    print(f"Class loss: {loss_class_val.item():.4f}, Reg loss: {loss_reg_val.item():.4f}, Total loss: {loss_val.item():.4f}")

    preds_class_val = torch.argmax(out_class_val, dim=1).cpu().numpy()
    preds_reg_val = out_reg_val.detach().cpu().numpy()
    targets_class_val = y_class_val.cpu().numpy()
    targets_reg_val = y_reg_val_scaled.cpu().numpy()

    acc_val, f1_val = compute_metrics_from_predictions(targets_class_val, preds_class_val, cfg.NUM_CLASSES)
    rmse_val = rmse_unscaled(preds_reg_val, targets_reg_val, reg_mean, reg_std)

    print(f"Validation batch metrics -> Accuracy: {acc_val:.4f}, Macro F1: {f1_val:.4f}, RMSE: {rmse_val:.4f}")


if __name__ == "__main__":
    sanity_check(fold_id=1)
