# dataset.py
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json

class LBNPDataset(Dataset):
    def __init__(self, pkl_dir, split_json, phase="train", channels=None,
                 use_scaled=True, use_cwt=False, transform=None):
        """
        Args:
            pkl_dir (str or Path): directory containing .pkl windows
            split_json (str or Path): JSON file with train/test splits from data_prep.py
            phase (str): "train" or "test"
            channels (list[str], optional): list of channels to select
            use_scaled (bool): if True, use precomputed scaled data; else raw
            use_cwt (bool): if True, return precomputed CWT
            transform (callable, optional): optional transform applied to x
        """
        self.pkl_dir = Path(pkl_dir)
        self.phase = phase
        self.use_scaled = use_scaled
        self.use_cwt = use_cwt
        self.transform = transform

        # Load split JSON
        with open(split_json, "r") as fh:
            split_data = json.load(fh)
        self.file_list = split_data[phase]

        # Infer available channels from first file
        sample_path = self.pkl_dir / self.file_list[0]
        with open(sample_path, "rb") as fh:
            sample = pickle.load(fh)
        available_channels = sample["data_raw"].shape[0]
        self.all_channels = channels or list(range(available_channels))
        if isinstance(self.all_channels[0], str):
            raise ValueError("Channel selection by index is recommended (integers)")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        pkl_path = self.pkl_dir / fname
        with open(pkl_path, "rb") as fh:
            d = pickle.load(fh)

        # Select data
        key = "data_scaled" if self.use_scaled else "data_raw"
        x = d[key][self.all_channels, :].astype(np.float32)  # [C, T]

        # Optionally append CWT as additional input
        if self.use_cwt:
            cwt = d.get("cwt")
            if cwt is not None:
                # select same channels
                cwt_sel = cwt[self.all_channels, :, :].astype(np.float32)  # [C, F, T]
                # return as a dict for architecture to handle separately
                x_out = {"signal": torch.tensor(x), "cwt": torch.tensor(cwt_sel)}
            else:
                x_out = torch.tensor(x)
        else:
            x_out = torch.tensor(x)

        # Targets
        y_class = d["label"][0]["class"] - 1  # 0-indexed
        y_class = torch.tensor(y_class, dtype=torch.long)
        y_sv = torch.tensor(d["label"][1]["reg"], dtype=torch.float32)

        if self.transform:
            if self.use_cwt:
                x_out = self.transform(x_out)
            else:
                x_out = self.transform(x_out)

        return x_out, y_class, y_sv

# -------------------------
# Optional quick test
# -------------------------
if __name__ == "__main__":
    pkl_dir = "/home/naim/LBNP/level7/pkl_windows_normwear_v3/pkls"
    split_json = "/home/naim/LBNP/level7/pkl_windows_normwear_v3/splits/train_test_split_fold1.json"
    channels = [0,1,2]  # select first 3 channels from data_prep
    dataset = LBNPDataset(pkl_dir, split_json, phase="train", channels=channels,
                           use_scaled=True, use_cwt=True)
    x, y_class, y_sv = dataset[0]
    print("x (signal) shape:", x["signal"].shape)
    print("x (cwt) shape:", x["cwt"].shape)
    print("y_class:", y_class)
    print("y_sv:", y_sv)
