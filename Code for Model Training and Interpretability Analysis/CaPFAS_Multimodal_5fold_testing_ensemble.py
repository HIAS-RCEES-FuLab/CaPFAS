import os
import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

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
epochs = 50
model_prefix = "multimodal_model_fold"

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
                    msms_x = np.zeros(max_msms_bin, dtype=np.float32)
                    feats = []
                    if "Mass_spectral_features" in df.columns and pd.notna(row["Mass_spectral_features"]):
                        feats = ast.literal_eval(row["Mass_spectral_features"])
                        for bin_idx, mz, inten, kmd in feats:
                            if bin_idx < max_msms_bin:
                                msms_x[int(bin_idx)] = inten

                    nl_x = np.zeros(max_nl_bin, dtype=np.float32)
                    if "Neutral_losses_binned" in df.columns and pd.notna(row["Neutral_losses_binned"]):
                        nl_list = ast.literal_eval(row["Neutral_losses_binned"])
                        for bin_idx, _ in nl_list:
                            if 0 <= bin_idx < max_nl_bin:
                                nl_x[int(bin_idx)] = 1.0

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
                except:
                    continue

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
            self.losses[idx]
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
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
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

_, test_idx_nonstandard = train_test_split(
    np.arange(len(dataset_nonstandard)),
    test_size=0.1,
    random_state=42,
    stratify=dataset_nonstandard.labels
)
test_dataset_nonstandard = Subset(dataset_nonstandard, test_idx_nonstandard)

test_dataset = ConcatDataset([test_dataset_nonstandard, dataset_standard])

n_folds = 5
fold_probs_list = []

for fold in range(1, n_folds + 1):
    model_path = f"{model_prefix}{fold}.pth"
    model = MultiModalNet(max_msms_bin, max_nl_bin, precursor_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    fold_probs = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            msms_x, nl_x, precursor_x, y_true, fname, msms_len, losses = test_dataset[i]
            msms_x = msms_x.unsqueeze(0).to(device)
            nl_x = nl_x.unsqueeze(0).to(device)
            precursor_x = precursor_x.unsqueeze(0).to(device)
            logits = model(msms_x, nl_x, precursor_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # shape (2,)
            fold_probs.append(probs)
    fold_probs_list.append(np.array(fold_probs))  # shape (n_samples, 2)

fold_probs_array = np.stack(fold_probs_list, axis=0)
ensemble_probs = np.mean(fold_probs_array, axis=0)  # shape (n_samples, 2)
ensemble_pred_labels = np.argmax(ensemble_probs, axis=1)

y_true_list = []
fname_list = []
msms_len_list = []
losses_list = []

for i in range(len(test_dataset)):
    _, _, _, y_true, fname, msms_len, losses = test_dataset[i]
    y_true_list.append(int(y_true.item()))
    fname_list.append(fname)
    msms_len_list.append(msms_len)
    losses_list.append(losses)

y_true_array = np.array(y_true_list)  # shape (n_samples,)

from sklearn.metrics import accuracy_score

comparison_records = []

for k in range(fold_probs_array.shape[0]):
    probs = fold_probs_array[k]            # shape (n_samples,2) for fold k
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true_array, preds)
    try:
        roc_auc = roc_auc_score(y_true_array, probs[:, 1])
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true_array, probs[:, 1])
    except ValueError:
        pr_auc = float("nan")

    comparison_records.append({
        "model": f"Fold_{k+1}",
        "accuracy": acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    })

# Ensemble (mean) metrics
ensemble_preds = np.argmax(ensemble_probs, axis=1)
ensemble_acc = accuracy_score(y_true_array, ensemble_preds)
try:
    ensemble_roc_auc = roc_auc_score(y_true_array, ensemble_probs[:, 1])
except ValueError:
    ensemble_roc_auc = float("nan")
try:
    ensemble_pr_auc = average_precision_score(y_true_array, ensemble_probs[:, 1])
except ValueError:
    ensemble_pr_auc = float("nan")

comparison_records.append({
    "model": "Ensemble_Mean",
    "accuracy": ensemble_acc,
    "roc_auc": ensemble_roc_auc,
    "pr_auc": ensemble_pr_auc
})

comparison_df = pd.DataFrame(comparison_records)
print("\nOverall Performance Comparison (Merged Test Set):")
print(comparison_df.to_string(index=False, float_format='{:0.4f}'.format))

comparison_df.to_csv("fold_vs_ensemble_overall_metrics.csv", index=False)

records = []
for i in range(len(test_dataset)):
    probs = ensemble_probs[i]
    y_pred = int(np.argmax(probs))
    cf2_flag = has_cf2_loss(losses_list[i])

    records.append({
        "file": fname_list[i],
        "true_label": int(y_true_array[i]),
        "pred_label": int(y_pred),
        "has_cf2": cf2_flag,
        "msms_len": msms_len_list[i],
        "score": float(probs[1])
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

    print(f"\nFile: {f}\nTotal Samples: {total_samples}\nAccuracy: {acc:.4f}")
    print(f"Correct Predictions - CF2 Present: {correct_cf2_present}")
    print(f"Correct Predictions - CF2 Absent: {correct_cf2_absent}")
    print(f"Incorrect Predictions - CF2 Present: {incorrect_cf2_present}")
    print(f"Incorrect Predictions - CF2 Absent: {incorrect_cf2_absent}")

overall_acc = results_df["correct"].mean()
roc_auc = ensemble_roc_auc
pr_auc = ensemble_pr_auc

print(f"\nFinal Test Set Overall Accuracy (Ensemble): {overall_acc:.4f}")
print(f"ROC-AUC (Ensemble): {roc_auc:.4f}")
print(f"PR-AUC  (Ensemble): {pr_auc:.4f}")
summary = pd.crosstab(index=results_df["has_cf2"], columns=results_df["correct"],
                      rownames=["CF2 Present"], colnames=["Model Correct"])
print("\nPrediction Summary by CF2 presence and correctness:")
print(summary)

# ROC
fpr, tpr, _ = roc_curve(y_true_array, results_df["score"].values)

plt.figure(figsize=(8,6), facecolor='none')
plt.plot(fpr, tpr, color='#87CEFA', lw=4)  # 恢复淡蓝色曲线
plt.xlim(0,1)
plt.ylim(0,1)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
plt.tight_layout()
plt.savefig("roc_curve_ensemble.png", dpi=300, transparent=True)
plt.show()

# PR
precision, recall, _ = precision_recall_curve(y_true_array, results_df["score"].values)
pos_ratio = np.mean(y_true_array)

plt.figure(figsize=(8,6), facecolor='none')
plt.plot(recall, precision, color='#FFB6C1', lw=4)  # 恢复淡红色曲线
plt.plot([0,1], [pos_ratio, pos_ratio], color='gray', lw=3, linestyle='--')
plt.xlim(0,1)
plt.ylim(0,1)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
plt.tight_layout()
plt.savefig("pr_curve_ensemble.png", dpi=300, transparent=True)
plt.show()
