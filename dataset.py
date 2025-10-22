# dataset.py
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class LBNPDataset(Dataset):
    """
    PyTorch Dataset for LBNP windows with precomputed CWT scalograms.

    Returns:
        x: torch.FloatTensor [nvar, in_chans=3, L, F]
        y_class: torch.LongTensor (0-indexed class)
        y_reg: torch.FloatTensor (SV regression target)
    """

    def __init__(self, pkl_dir, file_list, variates=None, transform=None):
        """
        Args:
            pkl_dir (str or Path): directory containing .pkl windows
            file_list (list[str]): list of filenames to use
            variates (list[str], optional): subset of channels/variates to select (e.g., ["PPG","EKG_kopi","ABP","Nexfin_ABP"])
            transform (callable, optional): optional transform applied to x
        """
        self.pkl_dir = Path(pkl_dir)
        self.file_list = file_list
        self.transform = transform

        # Full list of channels in .pkl data_prep
        self.all_variates = ["LBNP", "EKG_SD50", "Photopleth", "ABP_rd_ledning", "Nexfin_ABP", "EKG_kopi"]

        # Select only requested variates or all by default
        self.variates = variates or self.all_variates
        for var in self.variates:
            if var not in self.all_variates:
                raise ValueError(f"Requested variate '{var}' not in .pkl channels: {self.all_variates}")

        # Precompute indices for efficiency
        self.var_idx = [self.all_variates.index(v) for v in self.variates]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load window .pkl
        fname = self.file_list[idx]
        with open(self.pkl_dir / fname, "rb") as fh:
            d = pickle.load(fh)

        # --- Load precomputed CWT ---
        # Shape in .pkl: [nvar=6, in_chans=3, L, F]
        cwt_all = np.array(d["cwt"], dtype=np.float32)  

        # Select requested variates -> [nvar_selected, 3, L, F]
        x = cwt_all[self.var_idx, :, :, :]

        # Convert to torch tensor
        x = torch.tensor(x, dtype=torch.float32)

        # Classification target (3-class LBNP)
        y_class = d["label"][0]["class"] - 1  # 0-indexed
        y_class = torch.tensor(y_class, dtype=torch.long)

        # Regression target (SV mean)
        y_reg = torch.tensor(d["label"][1]["reg"], dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        return x, y_class, y_reg

# --- Optional quick test ---
if __name__ == "__main__":
    # Example usage
    pkl_dir = "/home/naim/LBNP/level7/pkl_windows_normwear_v3/pkls"
    file_list = [f.name for f in Path(pkl_dir).glob("Subj_1*.pkl")]

    # Select 4 variates as discussed
    variates = ["Photopleth", "EKG_kopi", "ABP_rd_ledning", "Nexfin_ABP"]
    dataset = LBNPDataset(pkl_dir, file_list, variates=variates)

    x, y_class, y_reg = dataset[0]
    print("x.shape:", x.shape)  # expected: [4, 3, L, F]
    print("y_class:", y_class)
    print("y_reg:", y_reg)
