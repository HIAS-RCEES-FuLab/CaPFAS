import os
import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, ConcatDataset
from sklearn.model_selection import train_test_split

# Setting
base_dir = r"D:\MSMS\MSMS_H+H-\Train_data"
all_files = [
    f for f in os.listdir(base_dir)
    if f.endswith(".csv") and "NEG" in f and (
        "standard" in f.lower() or "nist" in f.lower() or "gnps" in f.lower() or "mona" in f.lower()
    )
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_bin_index = 5000
model_path = "neutral_losses_mlp_model.pth"

cf2_mass = 49.9968
ppm_tol = 5

class NeutralLossDataset(Dataset):
    def __init__(self, file_list):
        self.data = []
        self.labels = []
        self.files = []
        self.losses = []

        for f in file_list:
            fpath = os.path.join(base_dir, f)
            df = pd.read_csv(fpath)

            fname_lower = f.lower()
            if "nonpfas" in fname_lower:
                label = 0
            elif "pfas" in fname_lower and "nonpfas" not in fname_lower:
                label = 1
            else:
                continue

            if "Neutral_losses_binned" not in df.columns:
                continue

            for i, nl_str in enumerate(df["Neutral_losses_binned"].dropna()):
                try:
                    nl_list = ast.literal_eval(nl_str)  # [(bin_idx, mz), ...]
                    x = np.zeros(max_bin_index, dtype=np.float32)
                    for bin_idx, _ in nl_list:
                        if 0 <= bin_idx < max_bin_index:
                            x[int(bin_idx)] = 1.0
                    self.data.append(x)
                    self.labels.append(label)
                    self.files.append(f)
                    self.losses.append(nl_str)
                except:
                    continue

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.files[idx], self.losses[idx]

class NeutralLossMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.model(x)

def has_cf2_loss(loss_list, target_mass=cf2_mass, ppm=ppm_tol):
    try:
        nl_list = ast.literal_eval(loss_list)
        for _, mz in nl_list:
            tol = target_mass * ppm * 1e-6
            if abs(mz - target_mass) <= tol:
                return True
        return False
    except:
        return False

standard_files = [f for f in all_files if "standard" in f.lower()]
nonstandard_files = [f for f in all_files if "standard" not in f.lower()]

dataset_nonstandard = NeutralLossDataset(nonstandard_files)
dataset_standard = NeutralLossDataset(standard_files)

train_idx, temp_idx = train_test_split(
    np.arange(len(dataset_nonstandard)),
    test_size=0.3,
    random_state=42,
    stratify=dataset_nonstandard.labels
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=1/3,
    random_state=42,
    stratify=dataset_nonstandard.labels[temp_idx]
)

test_dataset_split = Subset(dataset_nonstandard, test_idx)
test_dataset = ConcatDataset([test_dataset_split, dataset_standard])

model = NeutralLossMLP(max_bin_index).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

records = []
with torch.no_grad():
    for i in range(len(test_dataset)):
        x, y_true, fname, losses = test_dataset[i]
        x = x.unsqueeze(0).to(device)

        logits = model(x)
        y_pred = torch.argmax(logits, dim=1).item()
        cf2_flag = has_cf2_loss(losses)

        records.append({
            "file": fname,
            "true_label": int(y_true.item()),
            "pred_label": y_pred,
            "has_cf2": cf2_flag
        })

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

    print(f"\nFile: {f}")
    print(f"Total Samples: {total_samples}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Correct Predictions - CF2 Present: {correct_cf2_present}")
    print(f"Correct Predictions - CF2 Absent: {correct_cf2_absent}")
    print(f"Incorrect Predictions - CF2 Present: {incorrect_cf2_present}")
    print(f"Incorrect Predictions - CF2 Absent: {incorrect_cf2_absent}")

overall_acc = results_df["correct"].mean()
print(f"\nFinal Test Set Overall Accuracy (Split + Standards): {overall_acc:.4f}")

summary = pd.crosstab(
    index=results_df["has_cf2"],
    columns=results_df["correct"],
    rownames=["CF2 Present"],
    colnames=["Model Correct"]
)
print("\nPrediction Summary by CF2 presence and correctness:")
print(summary)
