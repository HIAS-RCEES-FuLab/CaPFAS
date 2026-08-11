import os
import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

# -----------------------------
# Parameter
# -----------------------------
# Set the dataset directory path
base_dir = r"D:\Dataset"
all_files = [
    f for f in os.listdir(base_dir)
    if f.endswith(".csv") and "NEG" in f and (
        "standard" in f.lower()
    )
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_msms_bin = 9500
max_nl_bin = 5000
precursor_dim = 3
latent_dim = 128
model_path = "Demo_model.pth"
cf2_mass = 49.9968
ppm_tol = 5

# -----------------------------
# Dataset
# -----------------------------
class MultiModalDataset(Dataset):
    def __init__(self, file_list):
        self.msms_data, self.nl_data, self.precursor_data = [], [], []
        self.labels, self.files, self.msms_lens, self.losses = [], [], [], []

        for f in file_list:
            df = pd.read_csv(os.path.join(base_dir, f))
            fname_lower = f.lower()
            if "nonpfas" in fname_lower:
                label = 0
            elif "pfas" in fname_lower and "nonpfas" not in fname_lower:
                label = 1
            else:
                continue

            for _, row in df.iterrows():
                try:
                    # MS/MS features
                    msms_x = np.zeros(max_msms_bin, dtype=np.float32)
                    feats = []
                    if "Mass_spectral_features" in df.columns and pd.notna(row["Mass_spectral_features"]):
                        feats = ast.literal_eval(row["Mass_spectral_features"])
                        for bin_idx, mz, inten, kmd in feats:
                            if bin_idx < max_msms_bin:
                                msms_x[int(bin_idx)] = inten

                    # Mass difference
                    nl_x = np.zeros(max_nl_bin, dtype=np.float32)
                    if "Neutral_losses_binned" in df.columns and pd.notna(row["Neutral_losses_binned"]):
                        nl_list = ast.literal_eval(row["Neutral_losses_binned"])
                        for bin_idx, _ in nl_list:
                            if 0 <= bin_idx < max_nl_bin:
                                nl_x[int(bin_idx)] = 1.0

                    # Precursor
                    precursor_x = np.zeros(precursor_dim, dtype=np.float32)
                    if "Exact_mass" in df.columns and "KMD" in df.columns:
                        exact_mass_norm = row["Exact_mass"] / 1000
                        kmd = row["KMD"]
                        kmd_ratio = kmd / exact_mass_norm if exact_mass_norm != 0 else 0
                        precursor_x[:] = np.array([exact_mass_norm, kmd, kmd_ratio], dtype=np.float32)

                    # save data
                    self.msms_data.append(msms_x)
                    self.nl_data.append(nl_x)
                    self.precursor_data.append(precursor_x)
                    self.labels.append(label)
                    self.files.append(f)
                    self.msms_lens.append(len(feats))
                    self.losses.append(row.get("Neutral_losses", "[]"))
                    self.raw_rows.append(row.to_dict())

                except:
                    continue

        # Convert
        self.msms_data = torch.tensor(np.array(self.msms_data), dtype=torch.float32)
        self.nl_data = torch.tensor(np.array(self.nl_data), dtype=torch.float32)
        self.precursor_data = torch.tensor(np.array(self.precursor_data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.msms_data[idx],
            self.nl_data[idx],
            self.precursor_data[idx],
            self.labels[idx],
            self.files[idx],
            self.msms_lens[idx],
            self.losses[idx],
        )

# -----------------------------
# Model
# -----------------------------
class SubNet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = h_dim
        layers.extend([nn.Linear(prev_dim, latent_dim), nn.ReLU()])
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class MultiModalNet(nn.Module):
    def __init__(self, msms_dim, nl_dim, precursor_dim, latent_dim):
        super().__init__()
        self.msms_net = SubNet(msms_dim, latent_dim, [1024, 512])
        self.nl_net = SubNet(nl_dim, latent_dim, [512, 256])
        self.precursor_net = SubNet(precursor_dim, latent_dim, [32, 64])
        self.fusion = nn.Sequential(nn.Linear(latent_dim*3, 64), nn.BatchNorm1d(64),
                                    nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))
    def forward(self, msms_x, nl_x, precursor_x):
        msms_latent = self.msms_net(msms_x)
        nl_latent = self.nl_net(nl_x)
        precursor_latent = self.precursor_net(precursor_x)
        fused = torch.cat([msms_latent, nl_latent, precursor_latent], dim=1)
        return self.fusion(fused)

# -----------------------------
# Get dataset
# -----------------------------
standard_files = [f for f in all_files if "standard" in f.lower()]
test_dataset = MultiModalDataset(standard_files)

# -----------------------------
# Load model
# -----------------------------
model = MultiModalNet(max_msms_bin, max_nl_bin, precursor_dim, latent_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -----------------------------
# Testing
# -----------------------------
records, y_true_list, y_score_list = [], [], []

with torch.no_grad():
    for i in range(len(test_dataset)):
        msms_x, nl_x, precursor_x, y_true, fname, msms_len, losses = test_dataset[i]
        msms_x, nl_x, precursor_x = msms_x.unsqueeze(0).to(device), nl_x.unsqueeze(0).to(device), precursor_x.unsqueeze(0).to(device)
        logits = model(msms_x, nl_x, precursor_x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        y_pred = np.argmax(probs)
        records.append({"file": fname, "true_label": int(y_true.item()), "pred_label": y_pred,
                        "msms_len": msms_len})
        y_true_list.append(int(y_true.item()))
        y_score_list.append(probs[1])
results_df = pd.DataFrame(records)
results_df["correct"] = results_df["true_label"] == results_df["pred_label"]

# -----------------------------
# Accuracy
# -----------------------------
for f in results_df["file"].unique():
    file_df = results_df[results_df["file"] == f]
    total_samples = len(file_df)
    acc = file_df["correct"].mean() if total_samples > 0 else 0.0
    print(f"\nFile: {f}\nTotal Samples: {total_samples}\nAccuracy: {acc:.4f}")
overall_acc = results_df["correct"].mean()
print(f"\nFinal Test Set Overall Accuracy: {overall_acc:.4f}")
