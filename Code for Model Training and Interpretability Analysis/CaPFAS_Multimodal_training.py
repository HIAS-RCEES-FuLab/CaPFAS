import os
import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Setting
base_dir = r"D:\MSMS\MSMS_H+H-\Train_data"
neg_files = [
    f for f in os.listdir(base_dir)
    if f.endswith(".csv") and "NEG" in f and ("nist" in f.lower() or "gnps" in f.lower() or "mona" in f.lower())
]
print(f"Training on NEG files: {neg_files}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_msms_bin = 9500
max_nl_bin = 5000
precursor_dim = 3
latent_dim = 128

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

            for idx, row in df.iterrows():
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

                    self.data.append((msms_x, nl_x, precursor_x))
                    self.labels.append(label)
                except:
                    continue

        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SubNet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MultiModalNet(nn.Module):
    def __init__(self, msms_dim, nl_dim, precursor_dim, latent_dim):
        super().__init__()
        self.msms_net = SubNet(msms_dim, latent_dim, hidden_dims=[1024, 512])
        self.nl_net = SubNet(nl_dim, latent_dim, hidden_dims=[512, 256])
        self.precursor_net = SubNet(precursor_dim, latent_dim, hidden_dims=[32, 64])
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim*3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        msms_x, nl_x, precursor_x = x
        msms_latent = self.msms_net(msms_x)
        nl_latent = self.nl_net(nl_x)
        precursor_latent = self.precursor_net(precursor_x)
        fused = torch.cat([msms_latent, nl_latent, precursor_latent], dim=1)
        out = self.fusion(fused)
        return out

dataset = MultiModalDataset(neg_files)
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

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

labels = dataset.labels
n_samples = len(labels)
n_positive = (labels == 1).sum().item()
n_negative = (labels == 0).sum().item()
weight = torch.tensor([n_samples / (2 * n_negative), n_samples / (2 * n_positive)], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weight)

model = MultiModalNet(max_msms_bin, max_nl_bin, precursor_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 50
best_val_acc = 0.0
save_path = "multimodal_model.pth"

for epoch in range(epochs):
    model.train()
    all_train_preds = []
    all_train_labels = []
    total_loss = 0

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
        total_loss += loss.item() * msms_x.size(0)

        preds = torch.argmax(logits, dim=1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(y_batch.cpu().numpy())

    train_acc = accuracy_score(all_train_labels, all_train_preds)

    model.eval()
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            (msms_x, nl_x, precursor_x), y_val = batch
            msms_x = msms_x.to(device)
            nl_x = nl_x.to(device)
            precursor_x = precursor_x.to(device)
            y_val = y_val.to(device)

            logits = model((msms_x, nl_x, precursor_x))
            preds = torch.argmax(logits, dim=1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(y_val.cpu().numpy())

    val_acc = accuracy_score(all_val_labels, all_val_preds)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_dataset):.4f} - "
          f"Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"💾 Best multimodal model saved at epoch {epoch+1} with Val Acc: {best_val_acc:.4f}")
