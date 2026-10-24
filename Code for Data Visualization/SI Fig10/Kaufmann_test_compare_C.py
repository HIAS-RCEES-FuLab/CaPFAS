import os
import glob
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# Read Kaufmann 95% HDR
# =========================
data = np.load("PFAS_Kaufmann_95HDR.npz")
X, Y, mask = data["X"], data["Y"], data["mask"]

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("mask shape:", mask.shape)
print("X range:", X.min(), X.max())
print("Y range:", Y.min(), Y.max())
print("X unique:", len(np.unique(X)))
print("Y unique:", len(np.unique(Y)))
print("HDR pixels:", int(mask.sum()), "/", mask.size)

if mask.any():
    print("HDR X range:", X[mask].min(), X[mask].max())
    print("HDR Y range:", Y[mask].min(), Y[mask].max())

# Directory containing the test data. The data are not publicly available because they include commercially sourced mass spectra.
base_dir = r"test_data_dir"

files = []
for mode in ["N", "P"]:
    files.extend(glob.glob(os.path.join(base_dir, mode, "*.csv")))

def parse_carbon(formula):
    match = re.search(r"C(\d*)", formula)
    if match:
        return int(match.group(1)) if match.group(1) else 1
    return np.nan

# =========================
# Processing
# =========================
all_mz_per_C = []
all_md_per_C = []
all_inside = []
all_cf2 = []

x_grid = X[:, 0]
y_grid = Y[0, :]

for file in files:

    df = pd.read_csv(file)

    mz_col = None
    for col in df.columns:
        if "mz" in col.lower() or "mass" in col.lower():
            mz_col = col
            break

    df = df[["Formula", mz_col, "CF2"]].copy()
    df.columns = ["Formula", "mz", "CF2"]

    df["mz"] = pd.to_numeric(df["mz"], errors="coerce")
    df["CF2"] = pd.to_numeric(df["CF2"], errors="coerce")

    df = df.dropna(subset=["Formula", "mz", "CF2"])

    df["C"] = df["Formula"].apply(parse_carbon)

    df = df.dropna(subset=["C"])
    df = df[df["C"] > 0]
    df = df[df["CF2"].isin([0, 1])]

    if len(df) == 0:
        continue

    mz = df["mz"].values
    C = df["C"].values
    cf2 = df["CF2"].astype(int).values

    # =========================
    # Kaufmann coordinate
    # =========================
    mass_defect = mz - np.round(mz)
    mz_per_C = mz / C
    md_per_C = mass_defect / C

    # =========================
    # Project
    # =========================
    ix = np.abs(
        x_grid[:, None] - mz_per_C[None, :]
    ).argmin(axis=0)

    iy = np.abs(
        y_grid[:, None] - md_per_C[None, :]
    ).argmin(axis=0)

    # HDR
    inside = mask[ix, iy].astype(int)

    all_mz_per_C.extend(mz_per_C)
    all_md_per_C.extend(md_per_C)
    all_inside.extend(inside)
    all_cf2.extend(cf2)

    print(
        f"✔ {os.path.basename(file)}: "
        f"{inside.sum()}/{len(inside)} "
        f"({inside.mean()*100:.2f}%)"
    )

# =========================
# convert to numpy
# =========================
all_mz_per_C = np.array(all_mz_per_C)
all_md_per_C = np.array(all_md_per_C)
all_inside = np.array(all_inside).astype(int)
all_cf2 = np.array(all_cf2).astype(int)


# =========================
# Performance
# =========================
y_true = all_cf2
y_pred = all_inside

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()


accuracy = accuracy_score(
    y_true,
    y_pred
)

TPR = tp / (tp + fn) if tp + fn > 0 else 0
FPR = fp / (fp + tn) if fp + tn > 0 else 0


print("\n==========================")
print("Kaufmann 95% HDR classification result")
print("==========================")

print(f"TP: {tp}")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")

print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"TPR:      {TPR:.4f} ({TPR*100:.2f}%)")
print(f"FPR:      {FPR:.4f} ({FPR*100:.2f}%)")


# =========================
# Save result
# =========================
result_df = pd.DataFrame({
    "m/z_per_C": all_mz_per_C,
    "mass_defect_per_C": all_md_per_C,
    "CF2": all_cf2,
    "Prediction": all_inside
})

result_df.to_csv(
    "Kaufmann_95HDR_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

metrics_df = pd.DataFrame([{
    "TP": tp,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "Accuracy": accuracy,
    "TPR": TPR,
    "FPR": FPR
}])

metrics_df.to_csv(
    "Kaufmann_95HDR_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================
# Plotting
# =========================
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 14
fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(
    X,
    Y,
    mask,
    levels=[0.5, 1],
    alpha=0.25
)

ax.contour(
    X,
    Y,
    mask,
    levels=[0.5],
    linewidths=0.8
)

ax.scatter(
    all_mz_per_C[all_inside == 1],
    all_md_per_C[all_inside == 1],
    s=5,
    alpha=0.6,
    label="Inside 95% HDR",
    zorder=2
)

ax.scatter(
    all_mz_per_C[all_inside == 0],
    all_md_per_C[all_inside == 0],
    s=5,
    alpha=0.6,
    label="Outside 95% HDR",
    zorder=1
)

ax.set_xlabel("m/z / C")
ax.set_ylabel("Mass defect / C")
ax.set_xlim(10, 120)
ax.set_ylim(-0.05, 0.02)
ax.legend()
plt.tight_layout()
plt.savefig(
    "Kaufmann_2D_scatter.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True
)
plt.show()