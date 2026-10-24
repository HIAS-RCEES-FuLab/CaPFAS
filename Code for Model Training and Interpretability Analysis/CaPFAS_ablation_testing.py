import os
import ast
import time
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score

# =========================
# Parameter
# =========================
base_dirs = [r"MS2_dataset_split_SMILES_type5_12_to_1"]
ion_mode_sets = [["P", "N"]]
bin_sizes = [0.01]
base_bin_size = 0.01
msms_min_mz, msms_max_mz = 50, 1000
nl_min_mz, nl_max_mz = 12, 1000
precursor_dim = 3
latent_dim = 64
batch_size = 128
feature_type = "Combined"
ablation_configs = [
    ("MS2", ["MS2"]),
    ("NL", ["NL"]),
    ("Precursor", ["Precursor"]),
    ("MS2_NL", ["MS2", "NL"]),
    ("MS2_Precursor", ["MS2", "Precursor"]),
    ("NL_Precursor", ["NL", "Precursor"])
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = r"saved_models_dropout_input_ablation"
test_root = r"MS2_dataset_split_SMILES_type5_12_to_1\test"
output_dir = os.path.join(model_dir, "test_results_pr_auc_after3")
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print(f"Device: {device}")
print(f"Test directory: {test_root}")
print(f"Model directory: {model_dir}")
print(f"Output directory: {output_dir}")
print("=" * 70)

# =========================
# Bin
# =========================
def get_bin_parameters(bin_size):
    min_msms_bin = int(msms_min_mz / bin_size)
    max_msms_bin = int(msms_max_mz / bin_size)
    min_nl_bin = int(nl_min_mz / bin_size)
    max_nl_bin = int(nl_max_mz / bin_size)
    return min_msms_bin, max_msms_bin, min_nl_bin, max_nl_bin

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
# Process feature
# =========================
def process_msms_features(feature_string, bin_size, max_msms_bin):
    msms_x = np.zeros(max_msms_bin, dtype=np.float32)
    if pd.isna(feature_string):
        return msms_x
    try:
        feats = ast.literal_eval(feature_string)
    except Exception:
        return msms_x
    for bin_idx, inten in feats:
        try:
            bin_idx = int(bin_idx)
            inten = float(inten)
            mz = bin_idx * base_bin_size
            if mz < msms_min_mz or mz >= msms_max_mz:
                continue
            new_bin = int(np.floor(mz / bin_size))
            if 0 <= new_bin < max_msms_bin and inten > msms_x[new_bin]:
                msms_x[new_bin] = inten
        except Exception:
            continue
    return msms_x

def process_nl_features(feature_string, bin_size, max_nl_bin):
    nl_x = np.zeros(max_nl_bin, dtype=np.float32)
    if pd.isna(feature_string):
        return nl_x
    try:
        nl_list = ast.literal_eval(feature_string)
    except Exception:
        return nl_x
    for bin_idx in nl_list:
        try:
            bin_idx = int(bin_idx)
            mz = bin_idx * base_bin_size
            if mz < nl_min_mz or mz >= nl_max_mz:
                continue
            new_bin = int(np.floor(mz / bin_size))
            if 0 <= new_bin < max_nl_bin:
                nl_x[new_bin] = 1.0
        except Exception:
            continue
    return nl_x

# =========================
# Dataset
# =========================
class MultiModalDataset(Dataset):
    def __init__(self, file_list, bin_size, max_msms_bin, max_nl_bin):
        self.data = []
        self.labels = []
        self.metadata = []
        failed = 0
        for fpath, mode in file_list:
            print(f"Read: {fpath}")
            try:
                df = pd.read_csv(fpath)
            except Exception as e:
                print(f"Read fail: {fpath}")
                print(e)
                continue
            ion_feature = 1.0 if mode == "P" else 0.0
            for row_idx, row in df.iterrows():
                try:
                    label = int(row["CF2"])
                    if label not in [0, 1]:
                        failed += 1
                        continue
                    if "Feature_Mass_spectral" in df.columns:
                        msms_x = process_msms_features(row["Feature_Mass_spectral"], bin_size, max_msms_bin)
                    else:
                        msms_x = np.zeros(max_msms_bin, dtype=np.float32)
                    if "Feature_Mass_difference" in df.columns:
                        mass_diff_x = process_nl_features(row["Feature_Mass_difference"], bin_size, max_nl_bin)
                    else:
                        mass_diff_x = np.zeros(max_nl_bin, dtype=np.float32)
                    if "Feature_Neutral_loss" in df.columns:
                        neutral_loss_x = process_nl_features(row["Feature_Neutral_loss"], bin_size, max_nl_bin)
                    else:
                        neutral_loss_x = np.zeros(max_nl_bin, dtype=np.float32)
                    if feature_type == "Mass_difference":
                        nl_x = mass_diff_x
                    elif feature_type == "Neutral_loss":
                        nl_x = neutral_loss_x
                    elif feature_type == "Combined":
                        nl_x = np.maximum(mass_diff_x, neutral_loss_x)
                    else:
                        raise ValueError(f"Unknown feature_type: {feature_type}")
                    precursor_x = np.zeros(precursor_dim, dtype=np.float32)
                    required_cols = ["Feature_Exact_mass", "Feature_KMD", "Feature_KMD_norm", "Feature_Mass_scale"]
                    if all(x in df.columns for x in required_cols):
                        exact_mass = float(row["Feature_Exact_mass"])
                        kmd_ratio = float(row["Feature_KMD_norm"])
                        precursor_x[:] = np.array([exact_mass / 1000.0, kmd_ratio, ion_feature], dtype=np.float32)
                    self.data.append((msms_x, nl_x, precursor_x))
                    self.labels.append(label)
                    self.metadata.append({"Source_File": fpath, "Mode": mode, "Row_Index": row_idx})
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
# Model
# =========================
class SubNet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super().__init__()
        layers = [nn.Dropout(0.2)]
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(0.5)])
            prev_dim = h_dim
        layers.extend([nn.Linear(prev_dim, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU(), nn.Dropout(0.3)])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MultiModalNet(nn.Module):
    def __init__(self, msms_dim, nl_dim, precursor_dim, latent_dim, modalities):
        super().__init__()
        self.modalities = modalities
        if "MS2" in modalities:
            self.msms_net = SubNet(msms_dim, latent_dim, [128])
        if "NL" in modalities:
            self.nl_net = SubNet(nl_dim, latent_dim, [128])
        if "Precursor" in modalities:
            self.precursor_net = SubNet(precursor_dim, latent_dim, [32])
        fusion_dim = latent_dim * len(modalities)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        msms_x, nl_x, precursor_x = x
        latent_features = []
        if "MS2" in self.modalities:
            latent_features.append(self.msms_net(msms_x))
        if "NL" in self.modalities:
            latent_features.append(self.nl_net(nl_x))
        if "Precursor" in self.modalities:
            latent_features.append(self.precursor_net(precursor_x))
        fused = torch.cat(latent_features, dim=1)
        return self.fusion(fused)

# =========================
# Test
# =========================
def test_model(model, test_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            (msms_x, nl_x, precursor_x), y = batch
            msms_x = msms_x.to(device)
            nl_x = nl_x.to(device)
            precursor_x = precursor_x.to(device)
            logits = model((msms_x, nl_x, precursor_x))
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    all_labels = np.asarray(all_labels)
    all_preds = np.asarray(all_preds)
    all_probs = np.asarray(all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    return accuracy, roc_auc, pr_auc, tn, fp, fn, tp, all_preds, all_probs

# =========================
# Main
# =========================
overall_start_time = time.time()
all_results = []

for base_dir in base_dirs:
    for bin_size in bin_sizes:
        for ion_modes in ion_mode_sets:
            db_name = os.path.basename(os.path.normpath(base_dir))
            ion_name = "_".join(ion_modes)
            bin_name = str(bin_size).replace(".", "p")
            _, max_msms_bin, _, max_nl_bin = get_bin_parameters(bin_size)
            test_files = get_files("test", ion_modes, base_dir)

            print("\n" + "=" * 70)
            print(f"Load test data: {db_name}")
            print(f"Bin size: {bin_size}")
            print(f"Ion modes: {ion_modes}")
            print(f"Test files: {len(test_files)}")
            print("=" * 70)
            test_dataset = MultiModalDataset(test_files, bin_size, max_msms_bin, max_nl_bin)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            metadata_df = pd.DataFrame(test_dataset.metadata)
            y_test = test_dataset.labels.numpy()
            for ablation_name, modalities in ablation_configs:
                experiment_name = f"{db_name}_{ablation_name}_bin{bin_name}_{ion_name}"
                model_file = os.path.join(model_dir, f"{experiment_name}_best_pr_auc_after3.pth")
                print("\n" + "-" * 70)
                print(f"Testing: {experiment_name}")
                print(f"Features: {'+'.join(modalities)}")
                print(f"Model: {model_file}")
                model = MultiModalNet(max_msms_bin, max_nl_bin, precursor_dim, latent_dim, modalities).to(device)
                try:
                    state_dict = torch.load(model_file, map_location=device)
                    model.load_state_dict(state_dict)
                    accuracy, roc_auc, pr_auc, tn, fp, fn, tp, preds, probs = test_model(model, test_loader)
                    print(f"Accuracy: {accuracy:.4f}")
                    print(f"ROC-AUC: {roc_auc:.4f}")
                    print(f"PR-AUC: {pr_auc:.4f}")
                    print(f"TP={tp}, FN={fn}, TN={tn}, FP={fp}")
                    result = {
                        "Dataset": db_name,
                        "Ablation": ablation_name,
                        "Features": "+".join(modalities),
                        "Bin_Size": bin_size,
                        "Ion_Modes": ion_name,
                        "Model": os.path.basename(model_file),
                        "Accuracy": accuracy,
                        "ROC_AUC": roc_auc,
                        "PR_AUC": pr_auc,
                        "TP": tp,
                        "FN": fn,
                        "TN": tn,
                        "FP": fp
                    }
                    all_results.append(result)
                    prediction_df = metadata_df.copy()
                    prediction_df["True_Label"] = y_test
                    prediction_df["Prediction"] = preds
                    prediction_df["Probability"] = probs
                    prediction_file = os.path.join(output_dir, f"{experiment_name}_best_pr_auc_after3_test_predictions.csv")
                    prediction_df.to_csv(prediction_file, index=False)
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

summary_df = pd.DataFrame(all_results)
summary_file = os.path.join(output_dir, "all_test_results_best_pr_auc_after3.csv")
summary_df.to_csv(summary_file, index=False)

total_time = time.time() - overall_start_time
print("\n" + "=" * 70)
print("All test finish")
print(f"Model number: {len(all_results)}")
print(f"Result summary: {summary_file}")
print(f"Total Runtime: {total_time:.2f} s ({total_time / 60:.2f} min)")
print("=" * 70)