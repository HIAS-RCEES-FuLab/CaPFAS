import os
import ast
import time
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Parameter
# =========================
standard_path = r"standard_PFAS_NEG.csv"
model_root = r"CV5_models"
output_path = r"standard_prediction_results.csv"

folds = [1, 2, 3, 4, 5]
ratio = 12
ion_name = "P_N"
metric = "pr_auc"
model_version = "after3"
bin_sizes = {"0p01": 0.01, "0p1": 0.1, "1": 1.0}

base_bin_size = 0.01
msms_min_mz, msms_max_mz = 50, 1000
nl_min_mz, nl_max_mz = 12, 1000
precursor_dim = 3
latent_dim = 64
batch_size = 128
feature_type = "Combined"
ion_feature = 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print(f"Device: {device}")
print(f"Standard sample: {standard_path}")
print(f"Output: {output_path}")
print("=" * 80)

# =========================
# Bin
# =========================
def get_bin_parameters(bin_size):
    return int(msms_min_mz / bin_size), int(msms_max_mz / bin_size), int(nl_min_mz / bin_size), int(nl_max_mz / bin_size)

# =========================
# MS/MS feature
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
            mz = int(bin_idx) * base_bin_size
            inten = float(inten)
            if mz < msms_min_mz or mz >= msms_max_mz:
                continue
            new_bin = int(np.floor(mz / bin_size))
            if 0 <= new_bin < max_msms_bin and inten > x[new_bin]:
                x[new_bin] = inten
        except Exception:
            continue
    return x

# =========================
# Neutral loss / mass difference
# =========================
def process_nl_features(feature_string, bin_size, max_nl_bin):
    x = np.zeros(max_nl_bin, dtype=np.float32)
    if pd.isna(feature_string):
        return x
    try:
        feats = ast.literal_eval(feature_string)
    except Exception:
        return x
    for bin_idx in feats:
        try:
            mz = int(bin_idx) * base_bin_size
            if mz < nl_min_mz or mz >= nl_max_mz:
                continue
            new_bin = int(np.floor(mz / bin_size))
            if 0 <= new_bin < max_nl_bin:
                x[new_bin] = 1.0
        except Exception:
            continue
    return x

# =========================
# Dataset
# =========================
class StandardPFASDataset(Dataset):
    def __init__(self, df, bin_size, max_msms_bin, max_nl_bin):
        self.data = []
        for _, row in df.iterrows():
            msms_x = process_msms_features(row["Feature_Mass_spectral"], bin_size, max_msms_bin)
            mass_diff_x = process_nl_features(row["Feature_Mass_difference"], bin_size, max_nl_bin)
            neutral_loss_x = process_nl_features(row["Feature_Neutral_loss"], bin_size, max_nl_bin)
            if feature_type == "Mass_difference":
                nl_x = mass_diff_x
            elif feature_type == "Neutral_loss":
                nl_x = neutral_loss_x
            else:
                nl_x = np.maximum(mass_diff_x, neutral_loss_x)
            precursor_x = np.array([float(row["Feature_Exact_mass"]) / 1000.0, float(row["Feature_KMD_norm"]), ion_feature], dtype=np.float32)
            self.data.append((msms_x, nl_x, precursor_x))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
    def __init__(self, msms_dim, nl_dim, precursor_dim, latent_dim):
        super().__init__()
        self.msms_net = SubNet(msms_dim, latent_dim, [128])
        self.nl_net = SubNet(nl_dim, latent_dim, [128])
        self.precursor_net = SubNet(precursor_dim, latent_dim, [32])
        self.fusion = nn.Sequential(nn.Linear(latent_dim * 3, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2))

    def forward(self, x):
        msms_x, nl_x, precursor_x = x
        msms_latent = self.msms_net(msms_x)
        nl_latent = self.nl_net(nl_x)
        precursor_latent = self.precursor_net(precursor_x)
        return self.fusion(torch.cat([msms_latent, nl_latent, precursor_latent], dim=1))

# =========================
# Model name
# =========================
def get_model_filename(fold, bin_name):
    return f"MS2_dataset_split_SMILES_type5_{ratio}_to_1_CV5_V2_fold{fold}_bin{bin_name}_{ion_name}_best_{metric}_after3.pth"

# =========================
# Model predict
# =========================
def predict_model(df, fold, bin_name, bin_size):
    model_filename = get_model_filename(fold, bin_name)
    model_path = os.path.join(model_root, model_filename)
    print(f"\nFold {fold} | Bin {bin_name} | {model_filename}")

    _, max_msms_bin, _, max_nl_bin = get_bin_parameters(bin_size)
    dataset = StandardPFASDataset(df, bin_size, max_msms_bin, max_nl_bin)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = MultiModalNet(max_msms_bin, max_nl_bin, precursor_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        for msms_x, nl_x, precursor_x in loader:
            msms_x = msms_x.to(device)
            nl_x = nl_x.to(device)
            precursor_x = precursor_x.to(device)
            logits = model((msms_x, nl_x, precursor_x))
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    probs = np.asarray(probs, dtype=np.float64)

    del model, dataset, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs

# =========================
# Main
# =========================
if __name__ == "__main__":
    start_time = time.time()
    try:
        df = pd.read_csv(standard_path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(standard_path, encoding="gb18030", low_memory=False)

    prediction_columns = []
    probability_matrix = []

    total = len(folds) * len(bin_sizes)
    current = 0

    for fold in folds:
        for bin_name, bin_size in bin_sizes.items():
            current += 1
            print("\n" + "=" * 80)
            print(f"Model {current}/{total}: Fold {fold}, Bin {bin_name}")
            print("=" * 80)

            probs = predict_model(df, fold, bin_name, bin_size)

            column_name = f"fold{fold}_bin{bin_name}"
            prediction_columns.append(column_name)
            probability_matrix.append(probs)

    for column_name, probs in zip(prediction_columns, probability_matrix):
        df[column_name] = probs

    probability_matrix = np.vstack(probability_matrix)
    ensemble_probability = np.mean(probability_matrix, axis=0)

    df["ensemble_probability"] = ensemble_probability
    df["label"] = (ensemble_probability >= 0.5).astype(int)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Prediction Completed")
    print(f"Number of samples: {len(df)}")
    print(f"Number of models: {len(prediction_columns)}")
    print(f"Ensemble accuracy: {(ensemble_probability >= 0.5).mean():.4f}")
    print(f"Output file: {output_path}")
    print(f"Total processing time: {(time.time() - start_time) / 60:.2f} min")
    print("=" * 80)
