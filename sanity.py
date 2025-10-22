import torch
from model import NormWearWithHeads

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simulate a batch of raw signals: [batch, nvar, L]
batch_size = 2
nvar = 3        # Number of channels
L = 5000        # Example signal length
x = torch.randn(batch_size, nvar, L)

# Initialize model
model = NormWearWithHeads(pretrained=False, num_classes=3, device=device).to(device)

# Forward pass
with torch.no_grad():
    out_class, out_reg = model(x)

print("Output shapes:")
print("Class logits:", out_class.shape)  # Expected: [batch_size, num_classes]
print("Regression output:", out_reg.shape)  # Expected: [batch_size]
