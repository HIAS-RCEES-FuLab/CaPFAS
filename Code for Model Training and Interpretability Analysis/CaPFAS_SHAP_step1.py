import os
import ast
import gc
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import shap

# =========================
# Parameter
# =========================
model_root = r"models"
output_root = r"SHAP/SHAP_result"
os.makedirs(output_root, exist_ok=True)
dataset_name = "MS2_dataset_split_SMILES_type5_12_to_1"
ratio = 12
bin_size = 0.01
ion_modes = ["P", "N"]
model_filename = "MS2_dataset_split_SMILES_type5_12_to_1_bin0p01_P_N_best_pr_auc_after3.pth"
msms_min_mz, msms_max_mz = 50, 1000
nl_min_mz, nl_max_mz = 12, 1000
precursor_dim = 3
latent_dim = 64
shap_background_size = 10
shap_sample_size = 10
feature_type = "Combined"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = dataset_name
model_path = os.path.join(model_root, model_filename)

# =========================
# Bin setting
# =========================
def get_bin_parameters(bin_size):
    return int(msms_min_mz / bin_size), int(msms_max_mz / bin_size), int(nl_min_mz / bin_size), int(nl_max_mz / bin_size)

# =========================
# get file
# =========================
def get_files(split, ion_modes, base_dir):
    file_list = []
    for mode in ion_modes:
        split_dir = os.path.join(base_dir, split, mode)
        for root, dirs, files in os.walk(split_dir):
            for file in files:
                if file.lower().endswith(".csv"):
                    file_list.append((os.path.join(root, file), mode))
    return file_list

# =========================
# Feature processing
# =========================
def process_msms_features(feature_string, bin_size, max_msms_bin):
    x = np.zeros(max_msms_bin, dtype=np.float32)
    if pd.isna(feature_string):
        return x
    try:
        feats = ast.literal_eval(feature_string)
    except Exception:
        return x
    for bin_idx, inten in feats:
        try:
            mz, inten = int(bin_idx) * 0.01, float(inten)
            if msms_min_mz <= mz < msms_max_mz:
                new_bin = int(round(mz / bin_size))
                if 0 <= new_bin < max_msms_bin and inten > x[new_bin]:
                    x[new_bin] = inten
        except Exception:
            continue
    return x

def process_nl_features(feature_string, bin_size, max_nl_bin):
    x = np.zeros(max_nl_bin, dtype=np.float32)
    if pd.isna(feature_string):
        return x
    try:
        nl_list = ast.literal_eval(feature_string)
    except Exception:
        return x
    for bin_idx in nl_list:
        try:
            mz = int(bin_idx) * 0.01
            if nl_min_mz <= mz < nl_max_mz:
                new_bin = int(round(mz / bin_size))
                if 0 <= new_bin < max_nl_bin:
                    x[new_bin] = 1.0
        except Exception:
            continue
    return x

# =========================
# Dataset
# =========================
class MultiModalDataset(Dataset):
    def __init__(self, file_list, bin_size, max_msms_bin, max_nl_bin):
        self.data, self.labels = [], []
        self.canonical_smiles, self.row_indices, self.ion_modes = [], [], []
        failed = 0
        for fpath, mode in file_list:
            print(fpath)
            df = pd.read_csv(fpath)
            ion_feature = 1.0 if mode == "P" else 0.0
            for row_idx, (_, row) in enumerate(df.iterrows()):
                try:
                    label = int(row["CF2"])
                    smiles = row["Canonical_SMILES"]
                    if label not in [0, 1] or pd.isna(smiles) or str(smiles).strip() == "":
                        failed += 1
                        continue
                    msms_x = process_msms_features(row["Feature_Mass_spectral"], bin_size, max_msms_bin) if "Feature_Mass_spectral" in df.columns else np.zeros(max_msms_bin, dtype=np.float32)
                    mass_diff_x = process_nl_features(row["Feature_Mass_difference"], bin_size, max_nl_bin) if "Feature_Mass_difference" in df.columns else np.zeros(max_nl_bin, dtype=np.float32)
                    neutral_loss_x = process_nl_features(row["Feature_Neutral_loss"], bin_size, max_nl_bin) if "Feature_Neutral_loss" in df.columns else np.zeros(max_nl_bin, dtype=np.float32)
                    if feature_type == "Mass_difference":
                        nl_x = mass_diff_x
                    elif feature_type == "Neutral_loss":
                        nl_x = neutral_loss_x
                    elif feature_type == "Combined":
                        nl_x = np.maximum(mass_diff_x, neutral_loss_x)
                    else:
                        raise ValueError(f"Unknown feature_type: {feature_type}")
                    if all(x in df.columns for x in ["Feature_Exact_mass", "Feature_KMD_norm", "Feature_KMD"]):
                        precursor_x = np.array([float(row["Feature_Exact_mass"]) / 1000.0, float(row["Feature_KMD_norm"]), ion_feature], dtype=np.float32)
                    else:
                        precursor_x = np.zeros(precursor_dim, dtype=np.float32)
                    self.data.append((msms_x, nl_x, precursor_x))
                    self.labels.append(label)
                    self.canonical_smiles.append(str(smiles))
                    self.row_indices.append(row_idx)
                    self.ion_modes.append(mode)
                except Exception:
                    failed += 1
                    continue
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)
        print(f"Loaded samples: {len(self.labels)}")
        if len(self.labels) > 0:
            print(f"PFAS: {(self.labels == 1).sum().item()}, Non-PFAS: {(self.labels == 0).sum().item()}")
        print(f"Failed samples: {failed}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# =========================
# Network
# =========================
class SubNet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super().__init__()
        layers, prev_dim = [nn.Dropout(0.2)], input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(0.5)])
            prev_dim = h_dim
        layers.extend([nn.Linear(prev_dim, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU(), nn.Dropout(0.3)])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MultiModalNet(nn.Module):
    def __init__(self, msms_dim, nl_dim, precursor_dim, latent_dim):
        super().__init__()
        self.msms_net = SubNet(msms_dim, latent_dim, [128])
        self.nl_net = SubNet(nl_dim, latent_dim, [128])
        self.precursor_net = SubNet(precursor_dim, latent_dim, [32])
        self.fusion = nn.Sequential(nn.Linear(latent_dim * 3, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2))

    def forward(self, x):
        msms_x, nl_x, precursor_x = x
        fused = torch.cat([self.msms_net(msms_x), self.nl_net(nl_x), self.precursor_net(precursor_x)], dim=1)
        return self.fusion(fused)

# =========================
# Load data and mode
# =========================
_, max_msms_bin, _, max_nl_bin = get_bin_parameters(bin_size)
print("=" * 80)
print(f"Device: {device}")
print(f"Dataset: {dataset_name}")
print(f"Ratio: {ratio}:1 | Bin: {bin_size} | Ion: {'_'.join(ion_modes)}")
print(f"Model: {model_filename}")
print(f"MS2 dimension: {max_msms_bin}")
print(f"NL dimension: {max_nl_bin}")
print("=" * 80)

eval_files = []
for split in ["val", "test"]:
    files = get_files(split, ion_modes, base_dir)
    print(f"{split} files: {len(files)}")
    eval_files.extend(files)

if not eval_files:
    raise RuntimeError("not find val/test CSV file。")

test_dataset = MultiModalDataset(eval_files, bin_size, max_msms_bin, max_nl_bin)
if len(test_dataset) == 0:
    raise RuntimeError("test data set is null。")

model = MultiModalNet(max_msms_bin, max_nl_bin, precursor_dim, latent_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =========================
# SHAP Wrapper
# =========================
class SHAPModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, msms_x, nl_x, precursor_x):
        return self.model([msms_x, nl_x, precursor_x])

# =========================
# P/N PFAS SHAP
# =========================
all_labels = test_dataset.labels.numpy()
all_modes = np.array(test_dataset.ion_modes)

for mode in ion_modes:
    print("\n" + "=" * 80)
    print(f"Begin {mode} mode PFAS SHAP")
    print("=" * 80)

    mode_idx = np.where((all_labels == 1) & (all_modes == mode))[0]

    msms_all = np.stack([test_dataset.data[i][0] for i in mode_idx])
    nl_all = np.stack([test_dataset.data[i][1] for i in mode_idx])
    precursor_all = np.stack([test_dataset.data[i][2] for i in mode_idx])

    print(f"{mode} PFAS samples: {len(mode_idx)}")
    print(f"MS2 shape: {msms_all.shape}")
    print(f"NL shape: {nl_all.shape}")
    print(f"MS1 shape: {precursor_all.shape}")

    rng = np.random.default_rng(42)
    n_total = len(mode_idx)
    background_idx = rng.choice(n_total, min(shap_background_size, n_total), replace=False)
    remaining_idx = np.setdiff1d(np.arange(n_total), background_idx)
    sample_idx = rng.choice(remaining_idx, min(shap_sample_size, len(remaining_idx)), replace=False)

    background = [
        torch.tensor(msms_all[background_idx], dtype=torch.float32, device=device),
        torch.tensor(nl_all[background_idx], dtype=torch.float32, device=device),
        torch.tensor(precursor_all[background_idx], dtype=torch.float32, device=device)
    ]

    samples = [
        torch.tensor(msms_all[sample_idx], dtype=torch.float32, device=device),
        torch.tensor(nl_all[sample_idx], dtype=torch.float32, device=device),
        torch.tensor(precursor_all[sample_idx], dtype=torch.float32, device=device)
    ]

    print(f"SHAP background: {len(background_idx)}")
    print(f"SHAP samples: {len(sample_idx)}")
    print(f"Begin SHAP...")
    shap_start_time = time.time()

    shap_model = SHAPModelWrapper(model).to(device)
    shap_model.eval()
    explainer = shap.GradientExplainer(shap_model, background)
    shap_values = explainer.shap_values(samples)

    shap_time = time.time() - shap_start_time
    print(f"SHAP finish: {shap_time:.2f} s ({shap_time / 60:.2f} min)")

    # =========================
    # 提取class 1 = PFAS
    # =========================
    msms_shap_raw = np.asarray(shap_values[0])
    nl_shap_raw = np.asarray(shap_values[1])
    precursor_shap_raw = np.asarray(shap_values[2])

    print(f"MS2 raw SHAP shape: {msms_shap_raw.shape}")
    print(f"NL raw SHAP shape: {nl_shap_raw.shape}")
    print(f"MS1 raw SHAP shape: {precursor_shap_raw.shape}")

    if msms_shap_raw.ndim == 3 and msms_shap_raw.shape[-1] == 2:
        msms_shap = msms_shap_raw[:, :, 1]
    elif msms_shap_raw.ndim == 2:
        msms_shap = msms_shap_raw

    if nl_shap_raw.ndim == 3 and nl_shap_raw.shape[-1] == 2:
        nl_shap = nl_shap_raw[:, :, 1]
    elif nl_shap_raw.ndim == 2:
        nl_shap = nl_shap_raw

    if precursor_shap_raw.ndim == 3 and precursor_shap_raw.shape[-1] == 2:
        precursor_shap = precursor_shap_raw[:, :, 1]
    elif precursor_shap_raw.ndim == 2:
        precursor_shap = precursor_shap_raw

    msms_shap = np.asarray(msms_shap, dtype=np.float64)
    nl_shap = np.asarray(nl_shap, dtype=np.float64)
    precursor_shap = np.asarray(precursor_shap, dtype=np.float64)

    print(f"MS2 processed SHAP shape: {msms_shap.shape}")
    print(f"NL processed SHAP shape: {nl_shap.shape}")
    print(f"MS1 processed SHAP shape: {precursor_shap.shape}")

    mode_output = os.path.join(output_root, mode)
    os.makedirs(mode_output, exist_ok=True)

    # =========================
    # MS1
    # =========================
    ms1_names = ["Exact mass", "KMD", "Ion mode"]
    ms1_mean_shap = np.asarray(np.mean(np.abs(precursor_shap), axis=0)).reshape(-1)

    print("precursor_shap type:", type(precursor_shap))
    print("precursor_shap shape:", precursor_shap.shape)
    print("precursor_shap ndim:", precursor_shap.ndim)
    print("ms1_mean_shap shape:", ms1_mean_shap.shape)

    ms1_importance = pd.DataFrame({
        "Feature": ms1_names,
        "Mean_abs_SHAP": ms1_mean_shap
    })

    ms1_importance["Relative_importance"] = ms1_importance["Mean_abs_SHAP"] / ms1_mean_shap.sum()

    ms1_importance.to_csv(
        os.path.join(mode_output, f"PFAS_SHAP_{mode}_MS1_all_features.csv"),
        index=False
    )

    # =========================
    # MS2 fragment sequence
    # =========================
    ms2_mean_shap = np.asarray(np.mean(np.abs(msms_shap), axis=0)).reshape(-1)
    ms2_nonzero_idx = np.where(ms2_mean_shap > 0)[0]
    ms2_nonzero_idx = ms2_nonzero_idx[np.argsort(ms2_mean_shap[ms2_nonzero_idx])[::-1]]

    ms2_results = pd.DataFrame({
        "Bin": ms2_nonzero_idx,
        "m/z range": [f"{idx * bin_size:.2f}-{(idx + 1) * bin_size:.2f}" for idx in ms2_nonzero_idx],
        "Mean_abs_SHAP": ms2_mean_shap[ms2_nonzero_idx]
    })

    ms2_results["Relative_importance"] = ms2_results["Mean_abs_SHAP"] / ms2_mean_shap.sum()

    ms2_results.to_csv(
        os.path.join(mode_output, f"PFAS_SHAP_{mode}_MS2_fragment_sequence_all_nonzero.csv"),
        index=False
    )

    # =========================
    # Mass difference + neutral loss
    # =========================
    nl_mean_shap = np.asarray(np.mean(np.abs(nl_shap), axis=0)).reshape(-1)
    nl_nonzero_idx = np.where(nl_mean_shap > 0)[0]
    nl_nonzero_idx = nl_nonzero_idx[np.argsort(nl_mean_shap[nl_nonzero_idx])[::-1]]

    nl_results = pd.DataFrame({
        "Bin": nl_nonzero_idx,
        "Mass difference / neutral loss range": [f"{idx * bin_size:.2f}-{(idx + 1) * bin_size:.2f}" for idx in nl_nonzero_idx],
        "Mean_abs_SHAP": nl_mean_shap[nl_nonzero_idx]
    })

    nl_results["Relative_importance"] = nl_results["Mean_abs_SHAP"] / nl_mean_shap.sum()

    nl_results.to_csv(
        os.path.join(mode_output, f"PFAS_SHAP_{mode}_mass_difference_neutral_loss_all_nonzero.csv"),
        index=False
    )

    del background, samples, shap_values
    del msms_shap_raw, nl_shap_raw, precursor_shap_raw
    del msms_shap, nl_shap, precursor_shap
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n" + "=" * 80)
print("P/N mode PFAS-only SHAP analysis finished")
print(f"Results: {output_root}")
print("=" * 80)

del model, test_dataset
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()