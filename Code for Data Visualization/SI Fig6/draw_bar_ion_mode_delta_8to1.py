import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

plt.rcParams["font.family"] = "Arial"
df = pd.read_csv("all_test_results_summary.csv")

bins = ["bin1", "bin0p1", "bin0p01"]
bin_labels = ["1", "0.1", "0.01"]
results = []

for b, bin_label in zip(bins, bin_labels):
    bin_size = float(bin_label)
    filtered = df[(df["model_version"] == "after3") & (df["metric"] == "pr_auc") & (df["ratio_number"] == 8) & (df["bin_size"] == bin_size) & (df["ion_mode"].isin(["P_N", "P", "N"]))]
    summary = filtered.groupby("ion_mode")[["TP", "FN", "TN", "FP"]].sum()
    result = summary.copy()
    result.loc["P+N_sum"] = summary.loc[["P", "N"]].sum()
    result["Accuracy"] = (result["TP"] + result["TN"]) / (result["TP"] + result["TN"] + result["FP"] + result["FN"])
    accuracy_pn = result.loc["P_N", "Accuracy"]
    accuracy_sum = result.loc["P+N_sum", "Accuracy"]

    n_file = f"MS2_dataset_split_SMILES_type5_8_to_1_{b}_N_pr_auc_after3_predictions.csv"
    p_file = f"MS2_dataset_split_SMILES_type5_8_to_1_{b}_P_pr_auc_after3_predictions.csv"
    pn_file = f"MS2_dataset_split_SMILES_type5_8_to_1_{b}_P_N_pr_auc_after3_predictions.csv"

    df_n = pd.read_csv(n_file)
    df_p = pd.read_csv(p_file)
    df_pn = pd.read_csv(pn_file)

    df_pn_sum = pd.concat([df_p[["label", "probability"]], df_n[["label", "probability"]]], ignore_index=True)
    roc_sum = roc_auc_score(df_pn_sum["label"], df_pn_sum["probability"])
    pr_sum = average_precision_score(df_pn_sum["label"], df_pn_sum["probability"])
    roc_pn = roc_auc_score(df_pn["label"], df_pn["probability"])
    pr_pn = average_precision_score(df_pn["label"], df_pn["probability"])

    results.append({
        "bin_size": bin_label,
        "P_N_Accuracy": accuracy_pn,
        "P+N_Accuracy": accuracy_sum,
        "Delta_Accuracy": accuracy_pn - accuracy_sum,
        "P_N_ROC_AUC": roc_pn,
        "P+N_ROC_AUC": roc_sum,
        "Delta_ROC_AUC": roc_pn - roc_sum,
        "P_N_PR_AUC": pr_pn,
        "P+N_PR_AUC": pr_sum,
        "Delta_PR_AUC": pr_pn - pr_sum
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("PN_vs_PNsum_comparison.csv", index=False)

x = np.arange(len(bin_labels))
width = 0.24
delta_accuracy = results_df["Delta_Accuracy"].values * 100
delta_roc = results_df["Delta_ROC_AUC"].values * 100
delta_pr = results_df["Delta_PR_AUC"].values * 100

fig, ax = plt.subplots(figsize=(3, 2))
ax.bar(x - width, delta_accuracy, width, color="#A8D5BA", edgecolor="none")
ax.bar(x, delta_roc, width, color="#F2C28B", edgecolor="none")
ax.bar(x + width, delta_pr, width, color="#B8B5E2", edgecolor="none")
ax.axhline(0, color="black", linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels([])
ax.tick_params(axis="x", direction="out", length=3, width=0.8)

ax.set_yticks([-1.2, 0, 1.2, 2.4])
ax.set_yticklabels([])
ax.tick_params(axis="y", direction="out", length=3, width=0.8)
ax.set_ylim(-1.2, 2.4)

ax.set_xlabel("")
ax.set_ylabel("")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

all_delta = np.concatenate([delta_accuracy, delta_roc, delta_pr])
y_min, y_max = min(0, all_delta.min()), max(0, all_delta.max())
margin = max((y_max - y_min) * 0.25, 0.1)
ax.set_ylim(y_min - margin, y_max + margin)

plt.tight_layout()
output_path = "PN_vs_PNsum_Delta_Accuracy_ROC_PR_8to1.png"
fig.savefig(output_path, dpi=600, transparent=True, bbox_inches="tight")
plt.show()