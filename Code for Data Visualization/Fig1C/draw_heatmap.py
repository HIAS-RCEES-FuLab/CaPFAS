import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# Positive ionization mode data available on Zenodo. https://doi.org/10.5281/zenodo.23618979
files = ["fragment_neutral_loss_match_result_P_table.csv", "fragment_neutral_loss_match_result_N_table.csv"]

data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print("Spectrum level samples:", data.shape)

data["Feature_annotation"] = data["Feature_annotation"].astype(int)
data["KMD_annotation"] = ((data["Feature_KMD"] > -0.15) & (data["Feature_KMD"] < 0.15)).astype(int)

def evaluate(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("======================")
    print(name)
    print("TP:", tp, "FP:", fp, "TN:", tn, "FN:", fn)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))
    return cm

def plot_cm(cm, filename):
    cm_plot = cm[[1, 0]][:, [1, 0]]

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both", length=4, width=1)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)

    ax.vlines(0.5, -0.5, 1.5, colors="black", linewidth=1)
    ax.hlines(0.5, -0.5, 1.5, colors="black", linewidth=1)
    ax.vlines([-0.5, 1.5], -0.5, 1.5, colors="black", linewidth=1)
    ax.hlines([-0.5, 1.5], -0.5, 1.5, colors="black", linewidth=1)

    plt.tight_layout()
    output_file = filename
    plt.savefig(output_file, dpi=600, transparent=True, bbox_inches="tight")
    plt.show()
    plt.close()

# Spectrum level - Feature
spectrum_data = data.dropna(subset=["PFAS_label", "Feature_annotation"])
cm = evaluate(
    spectrum_data["PFAS_label"].astype(int),
    spectrum_data["Feature_annotation"].astype(int),
    "Spectrum level - Feature"
)
plot_cm(cm, "Spectrum_level_Feature.png")

# Spectrum level - KMD
spectrum_kmd = data.dropna(subset=["PFAS_label", "Feature_KMD"])
cm = evaluate(
    spectrum_kmd["PFAS_label"].astype(int),
    spectrum_kmd["KMD_annotation"].astype(int),
    "Spectrum level - KMD"
)
plot_cm(cm, "Spectrum_level_KMD.png")

# Compound level - Feature
compound_feature = data.dropna(
    subset=["Canonical_SMILES", "PFAS_label", "Feature_annotation"]
).groupby("Canonical_SMILES", as_index=False).agg({
    "PFAS_label": "max",
    "Feature_annotation": "max"
})

cm = evaluate(
    compound_feature["PFAS_label"].astype(int),
    compound_feature["Feature_annotation"].astype(int),
    "Compound level - Feature"
)
plot_cm(cm, "Compound_level_Feature.png")

# Compound level - KMD
compound_kmd = data.dropna(
    subset=["Canonical_SMILES", "PFAS_label", "Feature_KMD"]
).groupby("Canonical_SMILES", as_index=False).agg({
    "PFAS_label": "max",
    "KMD_annotation": "max"
})

cm = evaluate(
    compound_kmd["PFAS_label"].astype(int),
    compound_kmd["KMD_annotation"].astype(int),
    "Compound level - KMD"
)
plot_cm(cm, "Compound_level_KMD.png")
