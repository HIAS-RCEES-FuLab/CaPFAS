import os
import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split

# Setting
base_dir = r"D:\MSMS\MSMS_H+H-\Train_data"
all_files = [
    f for f in os.listdir(base_dir)
    if f.endswith(".csv") and "NEG" in f and (
        "nist" in f.lower() or "gnps" in f.lower() or "mona" in f.lower() or "standard" in f.lower()
    )
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "precursor_mlp_model.pth"

cf2_mass = 49.9968
ppm_tol = 5

class MSMSNumericDataset(Dataset):
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

            if not {'Exact_mass', 'KMD'}.issubset(df.columns):
                continue

            for i, row in df.iterrows():
                try:
                    exact_mass_norm = row['Exact_mass'] / 1000
                    kmd = row['KMD']
                    kmd_ratio = kmd / exact_mass_norm if exact_mass_norm != 0 else 0
                    features = np.array([exact_mass_norm, kmd, kmd_ratio], dtype=np.float32)
                    self.data.append(features)
                    self.labels.append(label)
                    self.files.append(f)
                    self.losses.append(df.iloc[i]["Neutral_losses"] if "Neutral_losses" in df.columns else "[]")
                except:
                    continue

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.files[idx], self.losses[idx]

class NumericMLP(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.model(x)

def has_cf2_loss(loss_list, target_mass=cf2_mass, ppm=ppm_tol):
    try:
        losses = ast.literal_eval(loss_list)
        for l in losses:
            tol = target_mass * ppm * 1e-6
            if abs(l - target_mass) <= tol:
                return True
        return False
    except:
        return False

standard_files = [f for f in all_files if "standard" in f.lower()]
nonstandard_files = [f for f in all_files if "standard" not in f.lower()]

dataset_nonstandard = MSMSNumericDataset(nonstandard_files)
dataset_standard = MSMSNumericDataset(standard_files)

_, temp_idx = train_test_split(
    np.arange(len(dataset_nonstandard)),
    test_size=0.3,
    random_state=42,
    stratify=dataset_nonstandard.labels
)
_, test_idx = train_test_split(
    temp_idx,
    test_size=1/3,
    random_state=42,
    stratify=dataset_nonstandard.labels[temp_idx]
)
test_dataset_split = Subset(dataset_nonstandard, test_idx)

test_dataset = ConcatDataset([test_dataset_split, dataset_standard])

model = NumericMLP(input_dim=3).to(device)
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

    if "standard" in f.lower():
        print(f"Standard file - total wrong samples: {len(file_df[file_df['correct']==False])}")

overall_acc = results_df["correct"].mean()
print(f"\nFinal Test Set Overall Accuracy: {overall_acc:.4f}")

summary = pd.crosstab(
    index=results_df["has_cf2"],
    columns=results_df["correct"],
    rownames=["CF2 Present"],
    colnames=["Model Correct"]
)
print("\nPrediction Summary by CF2 presence and correctness:")
print(summary)
