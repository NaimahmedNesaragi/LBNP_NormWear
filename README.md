# LBNP_NormWear
This repository provides the **NormWear-based LBNP analysis pipeline**, supporting classification and regression of physiological signals. It includes **preprocessing, dataset preparation, model training, and evaluation**, with both **head-only training** and **fine-tuning** phases.
## 🔹 Project Structure
LBNP_NormWear/
├── dataset.py # LBNPDataset class for loading .pkl windows
├── model.py # NormWearWithHeads model definition
├── configs.py # Centralized configuration file
├── utils.py # Helper functions for metrics, scaler, checkpoints
├── train.py # Training script (heads + finetune)
├── sanity.py # Quick test script to verify pipeline
├── pkl_windows_normwear_v3/
│ ├── pkls/ # Preprocessed windowed data (.pkl)
│ └── splits/ # Train/test split JSON files
└── training_runs/ # Automatically created output folder for logs & checkpoints

## 🔹 Installation
1. Clone the repo:
```bash
git clone https://github.com/NaimahmedNesaragi/LBNP_NormWear.git
cd LBNP_NormWear

2. Create a Python environment:
conda create -n normwear_gpu python=3.10
conda activate normwear_gpu

3. Install dependencies:
pip install torch torchvision torchaudio
pip install timm numpy pandas scikit-learn tensorboard

4. Ensure NormWear pretrained weights are available at:
/home/naim/LBNP/level7/normwear_last_checkpoint-15470-correct.pth

Configuration (configs.py)
Centralized settings include:
Paths for data (PKL_DIR, SPLIT_DIR) and output (OUT_DIR)
Model parameters (VARIATES, NUM_CLASSES, FEAT_DIM)
Training parameters (BATCH_SIZE, LR_HEADS, LR_FINETUNE, NUM_EPOCHS_*, ACCUM_STEPS, MIXED_PRECISION)
Finetune blocks (FINETUNE_BLOCK_NAMES) and early stopping (EARLY_STOPPING_PATIENCE)

. Train a fold
Train a single fold with head-only and fine-tune phases:
python train.py --fold 3
Logs saved to training_runs/fold_<fold>_<timestamp>/
TensorBoard logs in tb/ folder inside the run directory
Best checkpoints:
best_heads_fold<fold>.pth
best_finetune_fold<fold>.pth

Modify configs

You can change:
BATCH_SIZE, LR_HEADS, LR_FINETUNE
NUM_EPOCHS_HEADS, NUM_EPOCHS_FINETUNE
VARIATES for input channels
FINETUNE_BLOCK_NAMES for selective fine-tuning

Pipeline Overview

Data Preparation
Windowed .pkl files for each subject/trial
Train/test splits stored as JSON (SPLIT_DIR)

Regression Scaler
Computes mean & std of training set regression targets
Used to scale regression targets during training

Model
NormWear backbone
Two heads:
Classification head → predict discrete LBNP stages
Regression head → predict continuous SV/Nexfin values

Training Phases
Phase 1: Train heads only
Backbone frozen
Only classification & regression heads optimized
Phase 2: Fine-tune top encoder blocks
Selective unfreezing of last N transformer blocks
Lower learning rate (LR_FINETUNE)
Early stopping based on validation accuracy

Metrics
Classification: Accuracy, Macro F1
Regression: RMSE (unscaled)
TensorBoard logging for visualization

