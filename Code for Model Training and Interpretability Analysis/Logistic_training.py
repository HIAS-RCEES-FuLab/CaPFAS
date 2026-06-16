import numpy as np
import pandas as pd
import os
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from sklearn.linear_model import LogisticRegression  # 引入逻辑回归

base_dir = r"D:\MSMS\MSMS_H+H-\Train_data"
neg_files = [
    f for f in os.listdir(base_dir)
    if f.endswith(".csv") and "NEG" in f and ("nist" in f.lower() or "gnps" in f.lower() or "mona" in f.lower())
]
print(f"Training on NEG files: {neg_files}")

max_msms_bin = 9500
max_nl_bin = 5000
precursor_dim = 3

class MultiModalDataset(Dataset):
    def __init__(self, file_list):
        self.data = []
        self.labels = []

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
                except:
                    continue

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MultiModalDataset(neg_files)
X = dataset.data
y = dataset.labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

logreg_clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

logreg_clf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, logreg_clf.predict(X_train))
test_acc = accuracy_score(y_test, logreg_clf.predict(X_test))
print(f"Logistic Regression - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")

save_path = "logistic_regression_model.pkl"
joblib.dump(logreg_clf, save_path)
print(f"✅ Logistic Regression model saved to {save_path}")
