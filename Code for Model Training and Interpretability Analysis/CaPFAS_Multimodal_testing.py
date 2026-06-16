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

# Setting
base_dir = r"D:\MSMS\MSMS_H+H-\Train_data"
all_files = [
    f for f in os.listdir(base_dir)
    if f.endswith(".csv") and "NEG" in f and (
        "standard" in f.lower() or "nist" in f.lower() or "gnps" in f.lower() or "mona" in f.lower()
    )
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_msms_bin = 9500
max_nl_bin = 5000
precursor_dim = 3
latent_dim = 128
model_path = "multimodal_model.pth"

cf2_mass = 49.9968
ppm_tol = 5

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

                    # Neutral losses
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

        self.msms_data = torch.tensor(np.array(self.msms_data), dtype=torch.float32)
        self.nl_data = torch.tensor(np.array(self.nl_data), dtype=torch.float32)
        self.precursor_data = torch.tensor(np.array(self.precursor_data), dtype=torch.float32)

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

def has_cf2_loss(loss_list, target_mass=cf2_mass, ppm=ppm_tol):
    try:
        losses = ast.literal_eval(loss_list)
        tol = target_mass * ppm * 1e-6
        return any(abs(l - target_mass) <= tol for l in losses)
    except:
        return False

standard_files = [f for f in all_files if "standard" in f.lower()]
nonstandard_files = [f for f in all_files if "standard" not in f.lower()]

dataset_nonstandard = MultiModalDataset(nonstandard_files)
dataset_standard = MultiModalDataset(standard_files)

_, temp_idx = train_test_split(np.arange(len(dataset_nonstandard)), test_size=0.3,
                               random_state=42, stratify=dataset_nonstandard.labels)
_, test_idx = train_test_split(temp_idx, test_size=1/3,
                               random_state=42, stratify=dataset_nonstandard.labels[temp_idx])
test_dataset_split = Subset(dataset_nonstandard, test_idx)
test_dataset = ConcatDataset([test_dataset_split, dataset_standard])

model = MultiModalNet(max_msms_bin, max_nl_bin, precursor_dim, latent_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

records, y_true_list, y_score_list = [], [], []

with torch.no_grad():
    for i in range(len(test_dataset)):
        msms_x, nl_x, precursor_x, y_true, fname, msms_len, losses = test_dataset[i]
        msms_x, nl_x, precursor_x = msms_x.unsqueeze(0).to(device), nl_x.unsqueeze(0).to(device), precursor_x.unsqueeze(0).to(device)
        logits = model(msms_x, nl_x, precursor_x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        y_pred = np.argmax(probs)
        cf2_flag = has_cf2_loss(losses)

        records.append({"file": fname, "true_label": int(y_true.item()), "pred_label": y_pred,
                        "has_cf2": cf2_flag, "msms_len": msms_len})
        y_true_list.append(int(y_true.item()))
        y_score_list.append(probs[1])

results_df = pd.DataFrame(records)
results_df["correct"] = results_df["true_label"] == results_df["pred_label"]

for f in results_df["file"].unique():
    file_df = results_df[results_df["file"] == f]
    total_samples = len(file_df)
    acc = file_df["correct"].mean() if total_samples > 0 else 0.0
    correct_cf2_present = file_df[(file_df["correct"]) & (file_df["has_cf2"])].shape[0]
    correct_cf2_absent = file_df[(file_df["correct"]) & (~file_df["has_cf2"])].shape[0]
    incorrect_cf2_present = file_df[(~file_df["correct"]) & (file_df["has_cf2"])].shape[0]
    incorrect_cf2_absent = file_df[(~file_df["correct"]) & (~file_df["has_cf2"])].shape[0]

    print(f"\nFile: {f}\nTotal Samples: {total_samples}\nAccuracy: {acc:.4f}")
    print(f"Correct Predictions - CF2 Present: {correct_cf2_present}")
    print(f"Correct Predictions - CF2 Absent: {correct_cf2_absent}")
    print(f"Incorrect Predictions - CF2 Present: {incorrect_cf2_present}")
    print(f"Incorrect Predictions - CF2 Absent: {incorrect_cf2_absent}")

    if "standard" in f.lower():
        error_msms = file_df[file_df["correct"] == False]["msms_len"].values
        print(f"Standard file wrong samples MS/MS feature lengths: {error_msms}")

overall_acc = results_df["correct"].mean()
roc_auc = roc_auc_score(y_true_list, y_score_list)
pr_auc = average_precision_score(y_true_list, y_score_list)

print(f"\nFinal Test Set Overall Accuracy: {overall_acc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

summary = pd.crosstab(index=results_df["has_cf2"], columns=results_df["correct"],
                      rownames=["CF2 Present"], colnames=["Model Correct"])
print("\nPrediction Summary by CF2 presence and correctness:")
print(summary)
