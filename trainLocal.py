# train.py
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from dataset1 import LBNPDataset
from model import NormWearWithHeads

# ---------------- CONFIG ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
pkl_dir = Path("/home/naim/LBNP/level7/pkl_windows_normwear")
split_dir = Path("/home/naim/LBNP/level7/pkl_windows_normwear")
fold_ids = [1, 2, 3]  # Which folds to run

batch_size = 16
lr_heads = 1e-3
lr_finetune = 1e-4
num_epochs_heads = 5
num_epochs_full = 10
classification_weight = 1.0
regression_weight = 0.5
num_classes = 3

criterion_class = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()

# ---------------- TRAIN FUNCTION ----------------
def train_one_fold(fold_id):
    print(f"\n===== Starting Fold {fold_id} =====")

    # --- Load train/test split JSON ---
    split_path = split_dir / f"train_test_split_fold{fold_id}.json"
    with open(split_path, "r") as f:
        split = json.load(f)

    train_files = split["train"]
    test_files = split["test"]
    print(f" Train files: {len(train_files)} | Test files: {len(test_files)}")

    # --- Data ---
    train_dataset = LBNPDataset(pkl_dir, train_files)
    test_dataset = LBNPDataset(pkl_dir, test_files)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model ---
    model = NormWearWithHeads(pretrained=True, num_classes=num_classes, device=device).to(device)

    # ---------------- Phase 1: Train heads only ----------------
    for name, param in model.named_parameters():
        if "class_head" not in name and "reg_head" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW([
        {"params": model.class_head.parameters(), "lr": lr_heads},
        {"params": model.reg_head.parameters(), "lr": lr_heads}
    ])

    print("Training heads only...")
    for epoch in range(num_epochs_heads):
        model.train()
        total_loss = 0
        for x, y_class, y_reg in train_loader:
            x, y_class, y_reg = x.to(device), y_class.to(device), y_reg.to(device)

            # 🔹 Convert one-hot to class indices if needed
            if y_class.dim() > 1 and y_class.size(1) > 1:
                y_class = torch.argmax(y_class, dim=1)

            optimizer.zero_grad()
            out_class, out_reg = model(x)
            print("out_class.shape:", out_class.shape)  # should be [batch, num_classes]
            print("y_class.shape:", y_class.shape)      # should be [batch]

            loss_class = criterion_class(out_class, y_class)
            loss_reg = criterion_reg(out_reg, y_reg)
            loss = classification_weight * loss_class + regression_weight * loss_reg

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Fold {fold_id} | Heads] Epoch {epoch+1}/{num_epochs_heads} | Loss: {total_loss/len(train_loader):.4f}")

    # ---------------- Phase 2: Fine-tune top encoder layers ----------------
    print(" Unfreezing top encoder layers for fine-tuning...")
    for name, param in model.named_parameters():
        if any(b in name for b in [
            "normwear.encoder_blocks.8",
            "normwear.encoder_blocks.9",
            "normwear.encoder_blocks.10",
            "normwear.encoder_blocks.11"
        ]):
            param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": lr_finetune}
    ])

    for epoch in range(num_epochs_full):
        model.train()
        total_loss = 0
        for x, y_class, y_reg in train_loader:
            x, y_class, y_reg = x.to(device), y_class.to(device), y_reg.to(device)

            # 🔹 Convert one-hot to class indices if needed
            if y_class.dim() > 1 and y_class.size(1) > 1:
                y_class = torch.argmax(y_class, dim=1)

            optimizer.zero_grad()
            out_class, out_reg = model(x)
            loss_class = criterion_class(out_class, y_class)
            loss_reg = criterion_reg(out_reg, y_reg)
            loss = classification_weight * loss_class + regression_weight * loss_reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Fold {fold_id} | Fine-tune] Epoch {epoch+1}/{num_epochs_full} | Loss: {total_loss/len(train_loader):.4f}")

    # ---------------- Evaluation ----------------
    model.eval()
    total_correct, total_samples, mse = 0, 0, 0
    with torch.no_grad():
        for x, y_class, y_reg in test_loader:
            x, y_class, y_reg = x.to(device), y_class.to(device), y_reg.to(device)

            # 🔹 Convert one-hot to class indices if needed
            if y_class.dim() > 1 and y_class.size(1) > 1:
                y_class = torch.argmax(y_class, dim=1)

            out_class, out_reg = model(x)
            preds = torch.argmax(out_class, dim=1)
            total_correct += (preds == y_class).sum().item()
            total_samples += y_class.size(0)
            mse += torch.mean((out_reg - y_reg)**2).item() * y_class.size(0)

    acc = total_correct / total_samples
    mse /= total_samples
    print(f" Fold {fold_id} | Test Acc: {acc:.3f} | Test MSE: {mse:.4f}")

    # Save per-fold results
    torch.save(model.state_dict(), split_dir / f"normwear_finetuned_fold{fold_id}.pth")


# ---------------- RUN ALL FOLDS ----------------
for fold_id in fold_ids:
    train_one_fold(fold_id)

print("\n 3-Fold training completed successfully!")
