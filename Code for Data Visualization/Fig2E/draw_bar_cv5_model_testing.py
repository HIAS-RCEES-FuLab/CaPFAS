import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

plt.rcParams["font.family"] = "Arial"

summary_path = "all_test_results_summary.csv"
df = pd.read_csv(summary_path)

filtered = df[
    (df["metric"] == "pr_auc") &
    (df["model_version"] == "after3")
]

bin_names = {
    0.01: "bin0p01",
    0.1: "bin0p1",
    1: "bin1"
}

# 5-fold Mean ± SD
summary_results = {}
for bin_size in [1, 0.1, 0.01]:
    data = filtered[filtered["bin_size"] == bin_size]
    print(f"\n{'=' * 60}")
    print(f"bin_size = {bin_size}, number = {len(data)}")
    print(f"5-fold Accuracy: {data['Accuracy'].mean():.6f} ± {data['Accuracy'].std():.6f}")
    print(f"5-fold ROC_AUC:  {data['ROC_AUC'].mean():.6f} ± {data['ROC_AUC'].std():.6f}")
    print(f"5-fold PR_AUC:   {data['PR_AUC'].mean():.6f} ± {data['PR_AUC'].std():.6f}")
    summary_results[bin_size] = {
        "Accuracy_mean": data["Accuracy"].mean(),
        "Accuracy_std": data["Accuracy"].std(),
        "ROC_AUC_mean": data["ROC_AUC"].mean(),
        "ROC_AUC_std": data["ROC_AUC"].std(),
        "PR_AUC_mean": data["PR_AUC"].mean(),
        "PR_AUC_std": data["PR_AUC"].std()
    }

# 15-model Ensemble: 3 bin sizes × 5 folds
all_probabilities = []
ensemble_labels = None
model_count = 0

for bin_size in [0.01, 0.1, 1]:
    bin_name = bin_names[bin_size]
    for fold in range(1, 6):
        filename = (
            f"MS2_dataset_split_SMILES_type5_12_to_1_"
            f"fold{fold}_{bin_name}_P_N_pr_auc_after3_predictions.csv"
        )
        file_path = filename
        pred = pd.read_csv(file_path)
        if ensemble_labels is None:
            ensemble_labels = pred["label"].values
        all_probabilities.append(pred["probability"].values)
        model_count += 1

if model_count == 15:
    all_probabilities = np.array(all_probabilities)
    ensemble_probability = np.mean(all_probabilities, axis=0)
    ensemble_pred = (ensemble_probability > 0.5).astype(int)
    ensemble_acc = accuracy_score(ensemble_labels, ensemble_pred)
    ensemble_roc = roc_auc_score(ensemble_labels, ensemble_probability)
    ensemble_pr = average_precision_score(ensemble_labels, ensemble_probability)
    print(f"\n{'=' * 60}")
    print("15-model Ensemble results:")
    print(f"Models: {model_count}")
    print(f"Accuracy: {ensemble_acc:.6f}")
    print(f"ROC_AUC:  {ensemble_roc:.6f}")
    print(f"PR_AUC:   {ensemble_pr:.6f}")

results = [
    [
        "Bin size = 1",
        summary_results[1]["Accuracy_mean"],
        summary_results[1]["ROC_AUC_mean"],
        summary_results[1]["PR_AUC_mean"]
    ],
    [
        "Bin size = 0.1",
        summary_results[0.1]["Accuracy_mean"],
        summary_results[0.1]["ROC_AUC_mean"],
        summary_results[0.1]["PR_AUC_mean"]
    ],
    [
        "Bin size = 0.01",
        summary_results[0.01]["Accuracy_mean"],
        summary_results[0.01]["ROC_AUC_mean"],
        summary_results[0.01]["PR_AUC_mean"]
    ],
    [
        "15-model Ensemble",
        ensemble_acc,
        ensemble_roc,
        ensemble_pr
    ]
]

result_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "ROC-AUC", "PR-AUC"]
)

std_df = pd.DataFrame([
    [
        summary_results[1]["Accuracy_std"],
        summary_results[1]["ROC_AUC_std"],
        summary_results[1]["PR_AUC_std"]
    ],
    [
        summary_results[0.1]["Accuracy_std"],
        summary_results[0.1]["ROC_AUC_std"],
        summary_results[0.1]["PR_AUC_std"]
    ],
    [
        summary_results[0.01]["Accuracy_std"],
        summary_results[0.01]["ROC_AUC_std"],
        summary_results[0.01]["PR_AUC_std"]
    ]
], columns=["Accuracy", "ROC-AUC", "PR-AUC"])

# Plotting
metrics = ["Accuracy", "ROC-AUC", "PR-AUC"]
colors = ["#85C1E9", "#F5CBA7", "#F1948A", "#D5B8E0"]
x = np.arange(len(metrics))
width = 0.8 / len(result_df)

fig, ax = plt.subplots(figsize=(3, 2))

for i, (_, row) in enumerate(result_df.iterrows()):
    values = [
        row["Accuracy"],
        row["ROC-AUC"],
        row["PR-AUC"]
    ]
    position = x + (i - (len(result_df) - 1) / 2) * width

    if i < 3:
        errors = [
            std_df.loc[i, "Accuracy"],
            std_df.loc[i, "ROC-AUC"],
            std_df.loc[i, "PR-AUC"]
        ]
        ax.bar(
            position,
            values,
            width=width,
            color=colors[i],
            yerr=errors,
            capsize=3.5,
            error_kw={
                "elinewidth": 1.4,
                "capthick": 1.4
            }
        )
    else:
        ax.bar(
            position,
            values,
            width=width,
            color=colors[i]
        )

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0.6, 1)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(axis="x", labelbottom=False)
ax.tick_params(axis="y", labelleft=False)

plt.tight_layout()

output_path = "5fold_binsize_15model_ensemble_performance.png"

fig.savefig(
    output_path,
    dpi=600,
    transparent=True,
    bbox_inches="tight"
)

plt.show()