# utils.py
import pickle
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple

# ----------------------
# Regression scaler
# ----------------------
def compute_regression_scaler_from_train(pkls_dir: Path, file_list: List[str]) -> Tuple[float, float]:
    """
    Compute mean and std of regression targets from training set.
    Supports .pkl or .json files.
    """
    regs = []
    for fname in file_list:
        p = pkls_dir / fname
        obj = None
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix == ".pkl":
            with open(p, "rb") as fh:
                obj = pickle.load(fh)
        elif p.suffix == ".json":
            with open(p, "r") as fh:
                obj = json.load(fh)
        else:
            raise ValueError(f"Unsupported file format: {p}")
        reg = float(obj["label"][1]["reg"])
        regs.append(reg)
    regs = np.array(regs, dtype=np.float64)
    mean = float(np.mean(regs))
    std = float(np.std(regs)) if np.std(regs) > 0 else 1.0
    return mean, std

# ----------------------
# Classification metrics
# ----------------------
def compute_metrics_from_predictions(targets: List[int], preds: List[int], num_classes: int) -> Tuple[float, float]:
    """
    Compute accuracy and macro-F1 from lists of targets and predictions.
    """
    targets = np.array(targets, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    acc = np.trace(cm) / float(np.sum(cm)) if np.sum(cm) > 0 else 0.0
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
    return float(acc), macro_f1

# ----------------------
# Regression RMSE (unscaled)
# ----------------------
def rmse_unscaled(preds_scaled: List[float], targets_scaled: List[float], mean: float, std: float) -> float:
    """
    Compute unscaled RMSE for regression.
    """
    preds = np.array(preds_scaled) * std + mean
    targets = np.array(targets_scaled) * std + mean
    return float(np.sqrt(np.mean((preds - targets) ** 2))) if len(preds) > 0 else 0.0

# ----------------------
# Checkpoint helpers
# ----------------------
def save_checkpoint(model: torch.nn.Module, path: Path):
    """
    Save model state dict to path.
    """
    torch.save(model.state_dict(), path)

def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> torch.nn.Module:
    """
    Load model state dict from path onto device.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model.load_state_dict(torch.load(path, map_location=device))
    return model
