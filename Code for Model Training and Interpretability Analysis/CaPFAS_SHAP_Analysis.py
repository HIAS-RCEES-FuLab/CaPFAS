from sklearn.model_selection import train_test_split
import pandas as pd
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, ConcatDataset
import shap
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.special import expit  # sigmoid

# parameter setting
base_dir = r"D:\MSMS\MSMS_H+H-\Train_data"
shap_save_path = r"D:\MSMS\SHAP_results"
os.makedirs(shap_save_path, exist_ok=True)

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
model_prefix = "multimodal_model_fold"

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
            nn.Linear(latent_dim * 3, 64),
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

model_path = f"{model_prefix}1.pth"
explainer_model = MultiModalNet(max_msms_bin, max_nl_bin, precursor_dim, latent_dim).to(device)
explainer_model.load_state_dict(torch.load(model_path, map_location=device))
explainer_model.eval()

def sample_dataset(dataset, labels, n_samples):
    sampled = []
    for label in [0, 1]:
        idx_label = np.where(labels == label)[0]
        chosen_idx = np.random.choice(idx_label, size=min(n_samples, len(idx_label)), replace=False)
        sampled.extend([dataset[i] for i in chosen_idx])
    return sampled

all_labels = np.array([test_dataset[i][3].item() for i in range(len(test_dataset))])

print(len(test_dataset))

background_samples = sample_dataset(test_dataset, all_labels, 50)
test_samples = sample_dataset(test_dataset, all_labels, 1500)

background_msms = torch.stack([s[0] for s in background_samples]).to(device)
background_nl = torch.stack([s[1] for s in background_samples]).to(device)
background_precursor = torch.stack([s[2] for s in background_samples]).to(device)

test_msms = torch.stack([s[0] for s in test_samples]).to(device)
test_nl = torch.stack([s[1] for s in test_samples]).to(device)
test_precursor = torch.stack([s[2] for s in test_samples]).to(device)

# SHAP
background = [background_msms, background_nl, background_precursor]
test_input = [test_msms, test_nl, test_precursor]

explainer = shap.GradientExplainer(explainer_model, background)
shap_values = explainer.shap_values(test_input)  # list [num_classes, num_samples, num_features]

shap_values_pos = np.array(shap_values[1])  # shape = [num_samples, num_features]

# feature_names
num_features = shap_values_pos.shape[1]

feature_names = []
# MSMS bins
for i in range(min(max_msms_bin, num_features)):
    feature_names.append(f"bin_{i}")
# NL bins
for i in range(min(max_nl_bin, num_features - len(feature_names))):
    feature_names.append(f"nl_{i}")
# Precursor features
for name in ["Exact_mass_norm", "KMD", "KMD_ratio"]:
    if len(feature_names) < num_features:
        feature_names.append(name)

assert len(feature_names) == num_features, "feature_names 与 shap_values_pos 长度不一致！"

shap_pos_list = []
for modality_idx, modality_shap in enumerate(shap_values):
    if modality_shap.ndim == 3 and modality_shap.shape[2] == 2:
        shap_pos_list.append(modality_shap[:, :, 1])  # shape -> [num_samples, num_features]
    else:
        shap_pos_list.append(np.array(modality_shap))

shap_values_pos = np.concatenate(shap_pos_list, axis=1)  # shape -> [num_samples, total_features]
num_samples, num_features = shap_values_pos.shape
print(f"Total features after concatenation: {num_features}")

feature_names = (
    [f"bin_{i}" for i in range(max_msms_bin)] +
    [f"nl_{i}" for i in range(max_nl_bin)] +
    ["Exact_mass_norm", "KMD", "KMD_ratio"]
)
print(f"Generated {len(feature_names)} feature names, first 5: {feature_names[:5]}")

feature_importance_mean = np.mean(np.abs(shap_values_pos), axis=0)
print(f"feature_importance_mean shape: {feature_importance_mean.shape}, first 5 values: {feature_importance_mean[:5]}")

# save pkl
shap_file = os.path.join(shap_save_path, "shap_feature_importance_pos.pkl")
os.makedirs(shap_save_path, exist_ok=True)
pickle.dump({
    "feature_importance_mean": feature_importance_mean,
    "feature_names": feature_names,
    "shap_values_pos": shap_values_pos
}, open(shap_file, "wb"))
print(f"Saved positive sample SHAP feature importance to pickle: {shap_file}")

# save csv
csv_file = os.path.join(shap_save_path, "shap_feature_importance_pos.csv")
df_shap = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importance_mean
})
df_shap.to_csv(csv_file, index=False)
print(f"Saved SHAP feature importance to CSV: {csv_file}")
print(f"CSV preview:\n{df_shap.head()}")

# output Top N features
top_N = 20
top_idx = np.argsort(-feature_importance_mean)[:top_N]
print("Top features contributing to PFAS prediction:")
for i in top_idx:
    print(f"{feature_names[i]}: {feature_importance_mean[i]:.4f}")


