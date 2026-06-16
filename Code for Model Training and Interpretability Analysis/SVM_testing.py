import os
import pandas as pd
import ast
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, ConcatDataset
from sklearn.svm import LinearSVC  # 线性核

base_dir = r"D:\MSMS\MSMS_H+H-\Train_data"
all_files = [
    f for f in os.listdir(base_dir)
    if f.endswith(".csv") and "NEG" in f and (
        "standard" in f.lower() or "nist" in f.lower() or "gnps" in f.lower() or "mona" in f.lower()
    )
]

model_path = "linear_svm_model.pkl"
cf2_mass = 49.9968
ppm_tol = 5  # ppm 偏差
max_msms_bin = 9500
max_nl_bin = 5000
precursor_dim = 3

class MultiModalDataset(Dataset):
    def __init__(self, file_list):
        self.data = []
        self.labels = []
        self.files = []
        self.msms_lens = []
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

            for _, row in df.iterrows():
                try:
                    msms_x = np.zeros(max_msms_bin, dtype=np.float32)
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

                    feature_vec = np.concatenate([msms_x, nl_x, precursor_x])
                    self.data.append(feature_vec)
                    self.labels.append(label)
                    self.files.append(f)
                    self.msms_lens.append(len(feats))
                    self.losses.append(row["Neutral_losses"] if "Neutral_losses" in df.columns else "[]")
                except:
                    continue
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.data[idx],
            self.labels[idx],
            self.files[idx],
            self.msms_lens[idx],
            self.losses[idx]
        )

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

dataset_nonstandard = MultiModalDataset(nonstandard_files)
dataset_standard = MultiModalDataset(standard_files)

_, test_idx = train_test_split(
    np.arange(len(dataset_nonstandard)),
    test_size=0.1,
    random_state=42,
    stratify=dataset_nonstandard.labels
)
test_dataset_split = Subset(dataset_nonstandard, test_idx)

test_dataset = ConcatDataset([test_dataset_split, dataset_standard])

svm_clf = joblib.load(model_path)

records = []
for i in range(len(test_dataset)):
    x, y_true, fname, msms_len, losses = test_dataset[i]
    y_pred = svm_clf.predict(x.reshape(1, -1))[0]  # 单样本预测
    cf2_flag = has_cf2_loss(losses)

    records.append({
        "file": fname,
        "true_label": int(y_true),
        "pred_label": int(y_pred),
        "has_cf2": cf2_flag,
        "msms_len": msms_len
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
        error_msms = file_df[file_df["correct"] == False]["msms_len"].values
        print(f"Standard file wrong samples MS/MS feature lengths: {error_msms}")

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
