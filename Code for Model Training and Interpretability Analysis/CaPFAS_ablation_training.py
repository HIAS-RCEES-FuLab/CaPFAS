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
base_dirs = [
    r"MS2_dataset_split_SMILES_type5_12_to_1",
]
ion_mode_sets = [
    ["P", "N"],
]
bin_sizes = [0.01]

base_bin_size = 0.01
msms_min_mz, msms_max_mz = 50, 1000
nl_min_mz, nl_max_mz = 12, 1000
precursor_dim = 3
latent_dim = 64
batch_size = 128
epochs = 50
feature_type = "Combined"

# Ablation setting
ablation_configs = [
    ("MS2", ["MS2"]),
    ("NL", ["NL"]),
    ("Precursor", ["Precursor"]),
    ("MS2_NL", ["MS2", "NL"]),
    ("MS2_Precursor", ["MS2", "Precursor"]),
    ("NL_Precursor", ["NL", "Precursor"])
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = r"saved_models_change_precursor_dropout_input_ablation"
os.makedirs(model_dir, exist_ok=True)

print("=" * 70)
print(f"Device: {device}")
print(f"Datasets: {len(base_dirs)}")
print(f"Bin sizes: {bin_sizes}")
print(f"Ion mode sets: {ion_mode_sets}")
print(f"Total experiments: {len(base_dirs) * len(bin_sizes) * len(ion_mode_sets)}")
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
# Feature process
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
        failed = 0

        for fpath, mode in file_list:
            print(fpath)
            df = pd.read_csv(fpath)

            ion_feature = 1.0 if mode == "P" else 0.0

            for _, row in df.iterrows():
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
                    required_cols = [
                        "Feature_Exact_mass",
                        "Feature_KMD",
                        "Feature_KMD_norm",
                        "Feature_Mass_scale"
                    ]

                    if all(x in df.columns for x in required_cols):
                        exact_mass = float(row["Feature_Exact_mass"])
                        exact_mass_norm = exact_mass / 1000.0
                        kmd = float(row["Feature_KMD"])
                        kmd_ratio = float(row["Feature_KMD_norm"])
                        mass_scale = float(row["Feature_Mass_scale"])
                        precursor_x[:] = np.array([
                            exact_mass_norm,
                            #kmd,
                            kmd_ratio,
                            #mass_scale,
                            ion_feature
                        ], dtype=np.float32)

                    self.data.append((msms_x, nl_x, precursor_x))
                    self.labels.append(label)

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
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        layers.append(nn.BatchNorm1d(latent_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MultiModalNet(nn.Module):
    def __init__(self, msms_dim, nl_dim, precursor_dim, latent_dim, modalities):
        super().__init__()
        self.modalities = modalities
        if "MS2" in modalities:
            self.msms_net = SubNet(msms_dim, latent_dim, hidden_dims=[128])
        if "NL" in modalities:
            self.nl_net = SubNet(nl_dim, latent_dim, hidden_dims=[128])
        if "Precursor" in modalities:
            self.precursor_net = SubNet(precursor_dim, latent_dim, hidden_dims=[32])
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
# Focal Loss
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

def train_one_experiment(base_dir, bin_size, ion_modes, ablation_name, modalities):
    db_name = os.path.basename(os.path.normpath(base_dir))
    ion_name = "_".join(ion_modes)
    bin_name = str(bin_size).replace(".", "p")
    experiment_name = f"{db_name}_{ablation_name}_bin{bin_name}_{ion_name}"
    min_msms_bin, max_msms_bin, min_nl_bin, max_nl_bin = get_bin_parameters(bin_size)

    print("\n" + "=" * 70)
    print(f"开始实验: {experiment_name}")
    print(f"Features: {', '.join(modalities)}")
    print(f"Dataset: {db_name}")
    print(f"Bin size: {bin_size}")
    print(f"Ion modes: {ion_modes}")
    print(f"MS2 dimension: {max_msms_bin}")
    print(f"NL dimension: {max_nl_bin}")
    print("=" * 70)

    train_files = get_files("train", ion_modes, base_dir)
    val_files = get_files("val", ion_modes, base_dir)
    print(f"Train CSV: {len(train_files)}")
    print(f"Val CSV:   {len(val_files)}")

    train_dataset = MultiModalDataset(train_files, bin_size, max_msms_bin, max_nl_bin)
    val_dataset = MultiModalDataset(val_files, bin_size, max_msms_bin, max_nl_bin)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    labels = train_dataset.labels
    n_samples = len(labels)
    n_positive = (labels == 1).sum().item()
    n_negative = (labels == 0).sum().item()

    ratio = n_negative / n_positive
    alpha_pos = ratio
    alpha = torch.tensor([1.0, alpha_pos], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2)

    print(f"Train samples: {n_samples}")
    print(f"PFAS: {n_positive}")
    print(f"Non-PFAS: {n_negative}")
    print(f"Class ratio: {ratio:.2f}")
    print(f"Class weight: [1.0, {alpha_pos:.4f}]")

    model = MultiModalNet(max_msms_bin, max_nl_bin, precursor_dim, latent_dim, modalities).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.6, patience=3, min_lr=1e-6)

    best_val_acc = 0.0
    best_val_pr_auc = 0.0
    best_val_loss = float("inf")
    best_after3_acc = 0.0
    best_after3_pr_auc = 0.0
    best_after3_loss = float("inf")

    save_path_acc = os.path.join(model_dir, f"model_best_acc_{experiment_name}.pth")
    save_path_pr = os.path.join(model_dir, f"model_best_pr_auc_{experiment_name}.pth")
    save_path_loss = os.path.join(model_dir, f"model_best_val_loss_{experiment_name}.pth")
    save_path_acc_after3 = os.path.join(model_dir, f"{experiment_name}_best_acc_after3.pth")
    save_path_pr_after3 = os.path.join(model_dir, f"{experiment_name}_best_pr_auc_after3.pth")
    save_path_loss_after3 = os.path.join(model_dir, f"{experiment_name}_best_loss_after3.pth")
    history_csv = os.path.join(model_dir, f"training_history_{experiment_name}.csv")

    history = []
    training_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        all_train_preds, all_train_labels, all_train_probs = [], [], []
        total_train_loss = 0.0

        for batch in train_loader:
            (msms_x, nl_x, precursor_x), y_batch = batch
            msms_x = msms_x.to(device)
            nl_x = nl_x.to(device)
            precursor_x = precursor_x.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model((msms_x, nl_x, precursor_x))
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * msms_x.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_train_preds.extend(preds.detach().cpu().numpy())
            all_train_labels.extend(y_batch.detach().cpu().numpy())
            all_train_probs.extend(probs.detach().cpu().numpy())

        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_loss = total_train_loss / len(train_dataset)
        train_tn, train_fp, train_fn, train_tp = confusion_matrix(all_train_labels, all_train_preds, labels=[0, 1]).ravel()
        train_roc_auc = roc_auc_score(all_train_labels, all_train_probs)
        train_pr_auc = average_precision_score(all_train_labels, all_train_probs)

        model.eval()
        all_val_preds, all_val_labels, all_val_probs = [], [], []
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                (msms_x, nl_x, precursor_x), y_val = batch
                msms_x = msms_x.to(device)
                nl_x = nl_x.to(device)
                precursor_x = precursor_x.to(device)
                y_val = y_val.to(device)
                logits = model((msms_x, nl_x, precursor_x))
                val_loss_batch = criterion(logits, y_val)
                total_val_loss += val_loss_batch.item() * msms_x.size(0)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                all_val_preds.extend(preds.detach().cpu().numpy())
                all_val_labels.extend(y_val.detach().cpu().numpy())
                all_val_probs.extend(probs.detach().cpu().numpy())

        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_loss = total_val_loss / len(val_dataset)
        val_tn, val_fp, val_fn, val_tp = confusion_matrix(all_val_labels, all_val_preds, labels=[0, 1]).ravel()
        val_roc_auc = roc_auc_score(all_val_labels, all_val_probs)
        val_pr_auc = average_precision_score(all_val_labels, all_val_probs)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        elapsed_time = time.time() - training_start_time

        history.append({
            "Ablation": ablation_name,
            "Features": "+".join(modalities),
            "Epoch": epoch + 1,
            "Learning_Rate": current_lr,
            "Train_Accuracy": train_acc,
            "Train_Loss": train_loss,
            "Train_TP": train_tp,
            "Train_FN": train_fn,
            "Train_TN": train_tn,
            "Train_FP": train_fp,
            "Train_ROC_AUC": train_roc_auc,
            "Train_PR_AUC": train_pr_auc,
            "Val_Accuracy": val_acc,
            "Val_Loss": val_loss,
            "Val_TP": val_tp,
            "Val_FN": val_fn,
            "Val_TN": val_tn,
            "Val_FP": val_fp,
            "Val_ROC_AUC": val_roc_auc,
            "Val_PR_AUC": val_pr_auc,
            "Elapsed_Time_seconds": elapsed_time
        })

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Train ROC-AUC: {train_roc_auc:.4f} - Train PR-AUC: {train_pr_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val ROC-AUC: {val_roc_auc:.4f} - Val PR-AUC: {val_pr_auc:.4f}")
        print(f"Train: TP={train_tp}, FN={train_fn}, TN={train_tn}, FP={train_fp} | Val: TP={val_tp}, FN={val_fn}, TN={val_tn}, FP={val_fp}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path_acc)
            print(f"Best Accuracy model saved at epoch {epoch + 1}: {best_val_acc:.4f}")
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            torch.save(model.state_dict(), save_path_pr)
            print(f"Best PR-AUC model saved at epoch {epoch + 1}: {best_val_pr_auc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path_loss)
            print(f"Best Val Loss model saved at epoch {epoch + 1}: {best_val_loss:.4f}")

        if epoch + 1 >= 3:
            if val_acc > best_after3_acc:
                best_after3_acc = val_acc
                torch.save(model.state_dict(), save_path_acc_after3)
                print(f"Best Accuracy after Epoch 3 saved at epoch {epoch + 1}: {best_after3_acc:.4f}")
            if val_pr_auc > best_after3_pr_auc:
                best_after3_pr_auc = val_pr_auc
                torch.save(model.state_dict(), save_path_pr_after3)
                print(f"Best PR-AUC after Epoch 3 saved at epoch {epoch + 1}: {best_after3_pr_auc:.4f}")
            if val_loss < best_after3_loss:
                best_after3_loss = val_loss
                torch.save(model.state_dict(), save_path_loss_after3)
                print(f"Best Val Loss after Epoch 3 saved at epoch {best_after3_loss:.4f}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv, index=False)
    total_training_time = time.time() - training_start_time

    print("=" * 60)
    print(f"Finish: {experiment_name}")
    print(f"Features: {', '.join(modalities)}")
    print(f"Training history: {history_csv}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Best Val PR-AUC: {best_val_pr_auc:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Total Training Time: {total_training_time:.2f} s ({total_training_time / 60:.2f} min)")
    print("=" * 60)

    del model, optimizer, scheduler, train_dataset, val_dataset, train_loader, val_loader, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

overall_start_time = time.time()
experiment_count = 0
total_experiments = len(base_dirs) * len(bin_sizes) * len(ion_mode_sets) * len(ablation_configs)

for base_dir in base_dirs:
    for bin_size in bin_sizes:
        for ion_modes in ion_mode_sets:
            for ablation_name, modalities in ablation_configs:
                experiment_count += 1
                print("\n" + "#" * 70)
                print(f"Experiment {experiment_count}/{total_experiments}")
                print(f"Dataset: {os.path.basename(os.path.normpath(base_dir))}")
                print(f"Bin size: {bin_size}")
                print(f"Ion modes: {ion_modes}")
                print(f"Ablation: {ablation_name} ({'+'.join(modalities)})")
                print("#" * 70)
                try:
                    train_one_experiment(base_dir, bin_size, ion_modes, ablation_name, modalities)
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

overall_time = time.time() - overall_start_time

print("\n" + "=" * 70)
print("All finish")
print(f"Database number: {len(base_dirs)}")
print(f"Bin size number: {len(bin_sizes)}")
print(f"Ion mode number: {len(ion_mode_sets)}")
print(f"Ablation configure number: {len(ablation_configs)}")
print(f"Experiments count: {total_experiments}")
print(f"Total Runtime: {overall_time:.2f} s ({overall_time / 60:.2f} min)")
print("=" * 70)