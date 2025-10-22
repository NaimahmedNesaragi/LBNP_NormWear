# model.py
"""
NormWear Model Wrapper for LBNP Dataset

This module wraps the pretrained NormWear model and adds:
- Flexible selection of nvar physiological variates
- CLS-based embedding extraction
- Dual heads: classification (3-class) and regression (SV)
- Supports precomputed CWT input

Input assumptions:
- Precomputed CWT per variate: [batch, nvar, in_chans=3, L, F]
  where in_chans=3 corresponds to [raw, d1, d2] signals from cwt_wrap
- Variable number of variates (nvar ≤ 6)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add NormWear repo to sys.path
normwear_path = Path("/home/naim/LBNP/level7/NormWear")
sys.path.insert(0, str(normwear_path))

# Import NormWear module
from modules.normwear import NormWear

class NormWearWithHeads(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, device=None, feat_dim=768):
        super().__init__()

        self.device = device if device is not None else torch.device("cpu")

        # NormWear: in_chans=3 is fixed for pretrained model
        # Each variate gets 3 channels: raw, d1, d2
        self.normwear = NormWear(in_chans=3).to(self.device)

        # Load pretrained weights
        checkpoint_path = "/home/naim/LBNP/level7/normwear_last_checkpoint-15470-correct.pth"
        if pretrained and checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "model" in ckpt:
                self.normwear.load_state_dict(ckpt["model"], strict=False)
            else:
                self.normwear.load_state_dict(ckpt, strict=False)
            print(f"✅ Loaded pretrained weights from {checkpoint_path}")

        # Feature dimension from NormWear embeddings (CLS token)
        self.feat_dim = feat_dim

        # Dual heads
        self.class_head = nn.Linear(self.feat_dim, num_classes).to(self.device)
        self.reg_head = nn.Linear(self.feat_dim, 1).to(self.device)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: input tensor
               Either:
               - precomputed CWT: [B, nvar, 3, L, F]
               - raw signal: [B, nvar, L] (will compute CWT internally)
        Returns:
            class_logits: [B, num_classes]
            reg_output: [B]
        """
        x = x.to(self.device)

        # --- Step 1: Convert raw signals to CWT if necessary ---
        if x.dim() == 3:
            # [B, nvar, L] -> compute CWT per variate
            B, nvar, L = x.shape
            cwt_list = []
            for i in range(nvar):
                # x[:, i, :] -> [B, L]
                # Each variate produces CWT: [B, in_chans=3, L, F]
                cwt_i = cwt_wrap(x[:, i, :], lowest_scale=1, largest_scale=65, step=1, wavelet_len=100)
                cwt_i = cwt_i.to(self.device)
                cwt_list.append(cwt_i)
            # Stack along nvar → [B, nvar, 3, L, F]
            x = torch.stack(cwt_list, dim=1)

        # --- Step 2: NormWear embedding ---
        # Input x: [B, nvar, in_chans=3, L, F]
        features = self.normwear.get_signal_embedding(x)
        # features: [B, nvar, P, E] where:
        # B = batch size
        # nvar = number of selected variates (1–6)
        # P = sequence of patch embeddings or CLS+patch tokens
        # E = feature dimension (feat_dim)

        # --- Step 3: CLS token extraction ---
        # Take CLS token per variate (first token, index 0)
        # cls_emb: [B, nvar, E]
        cls_emb = features[:, :, 0, :]

        # --- Step 4: Aggregate across variates ---
        # Option A: mean over nvar → [B, E]
        feat = cls_emb.mean(dim=1)
        # Option B (alternative): flatten variates → [B, nvar*E]
        # feat = cls_emb.reshape(B, -1)

        # --- Step 5: Apply dual heads ---
        class_logits = self.class_head(feat)        # [B, num_classes]
        reg_output = self.reg_head(feat).squeeze(1) # [B]

        return class_logits, reg_output


# --- Quick test ---
if __name__ == "__main__":
    B = 2       # batch size
    nvar = 4    # example: 4 selected variates
    L = 960     # sequence length
    F = 64      # number of frequency bins in CWT
    in_chans = 3

    # Dummy precomputed CWT input: [B, nvar, 3, L, F]
    x_dummy = torch.randn(B, nvar, in_chans, L, F)

    model = NormWearWithHeads(pretrained=False, num_classes=3)
    class_logits, reg_output = model(x_dummy)

    print("Input shape:", x_dummy.shape)
    print("Class logits shape:", class_logits.shape)
    print("Regression output shape:", reg_output.shape)
