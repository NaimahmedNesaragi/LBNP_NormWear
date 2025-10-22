# configs.py
from pathlib import Path
import torch
import json

class Config:
    # ----------------------
    # Paths
    # ----------------------
    PROJECT_ROOT = Path("/home/naim/LBNP/level7")
    PKL_DIR = PROJECT_ROOT / "pkl_windows_normwear_v3" / "pkls"
    SPLIT_DIR = PROJECT_ROOT / "pkl_windows_normwear_v3" / "splits"
    OUT_DIR = PROJECT_ROOT / "training_runs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # NormWear repository and checkpoint
    NORWEAR_REPO = PROJECT_ROOT / "NormWear"
    PRETRAINED_CHECKPOINT = PROJECT_ROOT / "normwear_last_checkpoint-15470-correct.pth"

    # ----------------------
    # Model / Dataset
    # ----------------------
    VARIATES = ["Photopleth", "EKG_kopi", "ABP_rd_ledning", "Nexfin_ABP"]
    ALL_VARIATES = ["LBNP", "EKG_SD50", "Photopleth", "ABP_rd_ledning", "Nexfin_ABP", "EKG_kopi"]
    NUM_CLASSES = 3
    FEAT_DIM = 768  # CLS token dimension from NormWear

    # ----------------------
    # Training
    # ----------------------
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

    FINETUNE_BLOCK_NAMES = [
        "normwear.encoder_blocks.6",
        "normwear.encoder_blocks.7",
        "normwear.encoder_blocks.8",
        "normwear.encoder_blocks.9",
        "normwear.encoder_blocks.10",
        "normwear.encoder_blocks.11"
    ]

    EARLY_STOPPING_PATIENCE = 8
    SAVE_EVERY = 1  # epochs
    BEST_METRIC = "val_acc"

    # ----------------------
    # Utilities
    # ----------------------
    @staticmethod
    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_json(data, path):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
