import os
import ast
import time
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score

# =========================
# Parameter
# =========================
model_root = r"CV5_models"
folds = [1, 2, 3, 4, 5]
dataset_ratios = [12]
ion_configurations = [["P", "N"]]
model_metrics = ["acc", "pr_auc", "val_loss"]
model_versions = ["original", "after3"]
bin_sizes = {"0p01": 0.01, "0p1": 0.1, "1": 1.0}
base_bin_size = 0.01
msms_min_mz, msms_max_mz = 50, 1000
nl_min_mz, nl_max_mz = 12, 1000
precursor_dim = 3
latent_dim = 64
batch_size = 128
feature_type = "Combined"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result_dir = r"CV5_test_results"
os.makedirs(result_dir, exist_ok=True)

print("=" * 80)
print(f"Device: {device}")
print(f"Dataset ratios: {dataset_ratios}")
print(f"Folds: {folds}")
print(f"Model metrics: {model_metrics}")
print(f"Model versions: {model_versions}")
print(f"Ion configurations: {['_'.join(x) for x in ion_configurations]}")
print(f"Bin sizes: {list(bin_sizes.keys())}")
print("=" * 80)

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
# MS/MS feature
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
            mz = int(bin_idx) * base_bin_size
            inten = float(inten)
            if mz < msms_min_mz or mz >= msms_max_mz:
                continue
            new_bin = int(np.floor(mz / bin_size))
            if 0 <= new_bin < max_msms_bin and inten > msms_x[new_bin]:
                msms_x[new_bin] = inten
        except Exception:
            continue
    return msms_x

# =========================
# Neutral loss / mass difference
# =========================
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
            mz = int(bin_idx) * base_bin_size
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
        self.canonical_smiles = []
        self.row_indices = []
        self.ion_modes = []
        failed = 0

        for fpath, mode in file_list:
            print(fpath)
            df = pd.read_csv(fpath)

            ion_feature = 1.0 if mode == "P" else 0.0

            for row_idx, (_, row) in enumerate(df.iterrows()):
                try:
                    label = int(row["CF2"])
                    if label not in [0, 1]:
                        failed += 1
                        continue

                    canonical_smiles = row["Canonical_SMILES"]
                    if pd.isna(canonical_smiles) or str(canonical_smiles).strip() == "":
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
                    required_cols = ["Feature_Exact_mass", "Feature_KMD_norm", "Feature_KMD"]

                    if all(x in df.columns for x in required_cols):
                        exact_mass = float(row["Feature_Exact_mass"])
                        kmd_ratio = float(row["Feature_KMD_norm"])
                        precursor_x[:] = np.array([exact_mass / 1000.0, kmd_ratio, ion_feature], dtype=np.float32)

                    self.data.append((msms_x, nl_x, precursor_x))
                    self.labels.append(label)
                    self.canonical_smiles.append(str(canonical_smiles))
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
# SubNet
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

# =========================
# MultiModalNet
# =========================
class MultiModalNet(nn.Module):
    def __init__(self, msms_dim, nl_dim, precursor_dim, latent_dim):
        super().__init__()
        self.msms_net = SubNet(msms_dim, latent_dim, hidden_dims=[128])
        self.nl_net = SubNet(nl_dim, latent_dim, hidden_dims=[128])
        self.precursor_net = SubNet(precursor_dim, latent_dim, hidden_dims=[32])
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        msms_x, nl_x, precursor_x = x
        msms_latent = self.msms_net(msms_x)
        nl_latent = self.nl_net(nl_x)
        precursor_latent = self.precursor_net(precursor_x)
        fused = torch.cat([msms_latent, nl_latent, precursor_latent], dim=1)
        return self.fusion(fused)

# =========================
# Get model name
# =========================
def get_model_filename(fold, ratio, bin_name, metric, ion_name, model_version):
    if model_version == "original":
        return f"model_best_{metric}_MS2_dataset_split_SMILES_type5_{ratio}_to_1_CV5_fold{fold}_bin{bin_name}_{ion_name}.pth"
    elif model_version == "after3":
        after3_metric = "loss" if metric == "val_loss" else metric
        return f"MS2_dataset_split_SMILES_type5_{ratio}_to_1_CV5_fold{fold}_bin{bin_name}_{ion_name}_best_{after3_metric}_after3.pth"
    raise ValueError(f"Unknown model_version: {model_version}")

# =========================
# Test
# =========================
def test_one_experiment(base_dir, fold, ratio, bin_size, bin_name, metric, ion_modes, model_root, model_version):
    db_name = os.path.basename(os.path.normpath(base_dir))
    ion_name = "_".join(ion_modes)

    model_filename = get_model_filename(fold, ratio, bin_name, metric, ion_name, model_version)
    model_path = os.path.join(model_root, model_filename)

    print("\n" + "=" * 80)
    print(f"Fold: {fold}")
    print(f"Dataset: {db_name}")
    print(f"Ratio: {ratio}:1 | Bin: {bin_size} | Metric: {metric} | Version: {model_version} | Ion: {ion_name}")
    print(f"Model: {model_path}")
    print("=" * 80)

    _, max_msms_bin, _, max_nl_bin = get_bin_parameters(bin_size)
    print(f"MS2 dimension: {max_msms_bin}")
    print(f"NL dimension: {max_nl_bin}")

    # =========================
    # Test data
    # =========================
    test_files = get_files("test", ion_modes, base_dir)
    print(f"Test CSV: {len(test_files)}")
    if not test_files:
        return None

    test_dataset = MultiModalDataset(test_files, bin_size, max_msms_bin, max_nl_bin)
    if len(test_dataset) == 0:
        return None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # =========================
    # Model
    # =========================
    model = MultiModalNet(max_msms_bin, max_nl_bin, precursor_dim, latent_dim).to(device)
    print("Load model...")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Load model success")

    # =========================
    # Prediction
    # =========================
    all_preds = []
    all_labels = []
    all_probs = []
    all_smiles = []
    all_row_indices = []
    all_ion_modes = []
    sample_start_idx = 0
    test_start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            (msms_x, nl_x, precursor_x), y_batch = batch
            msms_x = msms_x.to(device)
            nl_x = nl_x.to(device)
            precursor_x = precursor_x.to(device)

            logits = model((msms_x, nl_x, precursor_x))
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            batch_size_actual = len(y_batch)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_smiles.extend(test_dataset.canonical_smiles[sample_start_idx:sample_start_idx + batch_size_actual])
            all_row_indices.extend(test_dataset.row_indices[sample_start_idx:sample_start_idx + batch_size_actual])
            all_ion_modes.extend(test_dataset.ion_modes[sample_start_idx:sample_start_idx + batch_size_actual])
            sample_start_idx += batch_size_actual

    test_time = time.time() - test_start_time
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # =========================
    # Metrics
    # =========================
    test_acc = accuracy_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()

    if len(np.unique(all_labels)) == 2:
        test_roc_auc = roc_auc_score(all_labels, all_probs)
        test_pr_auc = average_precision_score(all_labels, all_probs)
    else:
        test_roc_auc = np.nan
        test_pr_auc = np.nan

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    # =========================
    # Print results
    # =========================
    print("\n" + "-" * 80)
    print(f"Fold: {fold}")
    print(f"Test samples: {len(all_labels)}")
    print(f"PFAS: {np.sum(all_labels == 1)}")
    print(f"Non-PFAS: {np.sum(all_labels == 0)}")
    print(f"Accuracy:    {test_acc:.6f}")
    print(f"ROC-AUC:     {test_roc_auc:.6f}")
    print(f"PR-AUC:      {test_pr_auc:.6f}")
    print(f"Precision:   {precision:.6f}")
    print(f"Recall:      {recall:.6f}")
    print(f"Specificity: {specificity:.6f}")
    print(f"F1-score:    {f1:.6f}")
    print(f"TP = {tp}, FN = {fn}, TN = {tn}, FP = {fp}")
    print(f"Test time: {test_time:.2f} s ({test_time / 60:.2f} min)")
    print("-" * 80)

    # =========================
    # Summary result
    # =========================
    result = {
        "fold": fold,
        "model_version": model_version,
        "ion_mode": ion_name,
        "ratio": f"{ratio}:1",
        "ratio_number": ratio,
        "bin_size": bin_size,
        "bin_name": bin_name,
        "metric": metric,
        "dataset": db_name,
        "model": model_filename,
        "test_samples": len(all_labels),
        "PFAS": int(np.sum(all_labels == 1)),
        "Non_PFAS": int(np.sum(all_labels == 0)),
        "Accuracy": test_acc,
        "ROC_AUC": test_roc_auc,
        "PR_AUC": test_pr_auc,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "TP": int(tp),
        "FN": int(fn),
        "TN": int(tn),
        "FP": int(fp),
        "Test_time_s": test_time
    }

    # =========================
    # Prediction file
    # =========================
    prediction_filename = f"{db_name}_fold{fold}_bin{bin_name}_{ion_name}_{metric}_{model_version}_predictions.csv"
    prediction_path = os.path.join(result_dir, prediction_filename)

    pd.DataFrame({
        "fold": fold,
        "Canonical_SMILES": all_smiles,
        "ion_mode": all_ion_modes,
        "row_index": all_row_indices,
        "label": all_labels,
        "probability": all_probs,
        "prediction": all_preds
    }).to_csv(prediction_path, index=False)

    print(f"Result save: {prediction_path}")

    # =========================
    # 清理
    # =========================
    del model, test_dataset, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

# =========================
# Main
# =========================
if __name__ == "__main__":
    start_time = time.time()
    all_results = []

    total_experiments = len(folds) * len(dataset_ratios) * len(model_metrics) * len(bin_sizes) * len(model_versions) * len(ion_configurations)
    current_experiment = 0

    print("\n" + "=" * 80)
    print(f"Total experiment number: {total_experiments}")
    print(f"Count: {len(folds)} folds × {len(bin_sizes)} bins × {len(model_metrics)} metrics × {len(model_versions)} versions")
    print("=" * 80)

    for fold in folds:
        for ratio in dataset_ratios:
            base_dir = f"MS2_dataset_split_SMILES_type5_{ratio}_to_1"

            for bin_name, bin_size in bin_sizes.items():
                for metric in model_metrics:
                    for model_version in model_versions:
                        for ion_modes in ion_configurations:
                            current_experiment += 1

                            print("\n" + "#" * 80)
                            print(f"Experiment {current_experiment}/{total_experiments}")
                            print(f"Fold = {fold}")
                            print(f"Ratio = {ratio}:1")
                            print(f"Bin = {bin_name}")
                            print(f"Metric = {metric}")
                            print(f"Version = {model_version}")
                            print(f"Ion = {'_'.join(ion_modes)}")
                            print("#" * 80)

                            result = test_one_experiment(
                                base_dir=base_dir,
                                fold=fold,
                                ratio=ratio,
                                bin_size=bin_size,
                                bin_name=bin_name,
                                metric=metric,
                                ion_modes=ion_modes,
                                model_root=model_root,
                                model_version=model_version
                            )

                            if result is not None:
                                all_results.append(result)

    # =========================
    # Save all result
    # =========================
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary_path = os.path.join(result_dir, "all_test_results_summary.csv")
        results_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 80)
        print(f"Finish experiment: {len(all_results)}/{total_experiments}")
        print("\nResult summary:")
        print(results_df[[
            "fold", "model_version", "ion_mode", "ratio", "bin_size", "metric",
            "Accuracy", "ROC_AUC", "PR_AUC", "Precision", "Recall", "Specificity", "F1"
        ]].to_string(index=False))

        # =========================
        # 5-fold
        # =========================
        mean_std_df = results_df.groupby([
            "model_version", "ion_mode", "ratio_number", "bin_size", "bin_name", "metric"
        ])[[
            "Accuracy", "ROC_AUC", "PR_AUC", "Precision", "Recall", "Specificity", "F1"
        ]].agg(["mean", "std"]).reset_index()

        mean_std_path = os.path.join(result_dir, "all_test_results_5fold_mean_std.csv")
        mean_std_df.to_csv(mean_std_path, index=False)

        print("\n" + "=" * 80)
        print(f"5-fold average ± SD result save: {mean_std_path}")
        print("=" * 80)
    else:
        print("\nno success result。")

    # =========================
    # Total time
    # =========================
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("All finish")
    print(f"Total time: {total_time:.2f} s ({total_time / 60:.2f} min)")
    print("=" * 80)