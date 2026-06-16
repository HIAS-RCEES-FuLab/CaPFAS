import os
import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Setting
base_dir = r"D:\MSMS\MSMS_H+H-\Train_data"
neg_files = [
    f for f in os.listdir(base_dir)
    if f.endswith(".csv") and "NEG" in f and ("nist" in f.lower() or "gnps" in f.lower() or "mona" in f.lower())
]
print(f"Training on NEG files: {neg_files}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_bin_index = 9500

class MSMSDataset(Dataset):
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

            for feat_str in df["Mass_spectral_features"].dropna():
                try:
                    feats = ast.literal_eval(feat_str)
                    x = np.zeros(max_bin_index, dtype=np.float32)
                    for bin_idx, mz, inten, kmd in feats:
                        if bin_idx < max_bin_index:
                            x[int(bin_idx)] = inten
                    self.data.append(x)
                    self.labels.append(label)
                except:
                    continue

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class EnhancedNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.model(x)

dataset = MSMSDataset(neg_files)
train_idx, temp_idx = train_test_split(
    np.arange(len(dataset)),
    test_size=0.3,
    random_state=42,
    stratify=dataset.labels
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=1/3,
    random_state=42,
    stratify=dataset.labels[temp_idx]
)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

labels = dataset.labels
n_samples = len(labels)
n_positive = (labels==1).sum().item()
n_negative = (labels==0).sum().item()
weight = torch.tensor([n_samples/(2*n_negative), n_samples/(2*n_positive)], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weight)

model = EnhancedNN(max_bin_index).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 50
best_val_acc = 0.0
save_path = "msms_mlp_model.pth"

for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_train_preds = []
    all_train_labels = []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(y_batch.cpu().numpy())

    train_acc = accuracy_score(all_train_labels, all_train_preds)

    model.eval()
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            logits = model(X_val)
            preds = torch.argmax(logits, dim=1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(y_val.cpu().numpy())

    val_acc = accuracy_score(all_val_labels, all_val_preds)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_dataset):.4f} - "
          f"Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"💾 Best model saved at epoch {epoch+1} with Val Acc: {best_val_acc:.4f}")
