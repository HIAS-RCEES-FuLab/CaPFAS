import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

plt.rcParams["font.family"] = "Arial"

ratios = ["4_to_1", "8_to_1", "12_to_1", "16_to_1"]
bins = ["bin0p01", "bin0p1", "bin1"]
weights = [1, 1, 1]
results = []

for ratio in ratios:
    print("\n=========================")
    print("Ratio:", ratio)
    print("=========================")

    prob_list = []

    for b in bins:
        filename = f"MS2_dataset_split_SMILES_type5_{ratio}_{b}_P_N_pr_auc_after3_predictions.csv"
        df = pd.read_csv(filename)

        prob = df["probability"]
        prob_list.append(prob)

        label = df["label"]

        roc_auc = roc_auc_score(label, prob)

        results.append([
            ratio,
            b,
            roc_auc
        ])

    ensemble_probability = (
        prob_list[0] * weights[0] +
        prob_list[1] * weights[1] +
        prob_list[2] * weights[2]
    ) / sum(weights)

    ensemble_roc_auc = roc_auc_score(
        label,
        ensemble_probability
    )

    results.append([
        ratio,
        "Weighted Ensemble (1:1:1)",
        ensemble_roc_auc
    ])

result_df = pd.DataFrame(
    results,
    columns=[
        "Down-sampling ratio",
        "Model",
        "ROC-AUC"
    ]
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(result_df)
models = [
    "bin1",
    "bin0p1",
    "bin0p01",
    "Weighted Ensemble (1:1:1)"
]

colors = [
    "#85C1E9",  # bin1
    "#F5CBA7",  # bin0.1
    "#F1948A",  # bin0.01
    "#C39BD3"   # Ensemble
]

x = np.arange(len(ratios))
width = 0.8 / len(models)

fig, ax = plt.subplots(figsize=(3, 2))

for i, model in enumerate(models):

    values = []

    for ratio in ratios:
        row = result_df[
            (result_df["Down-sampling ratio"] == ratio) &
            (result_df["Model"] == model)
        ]

        values.append(row["ROC-AUC"].iloc[0])

    ax.bar(
        x + (i - (len(models) - 1) / 2) * width,
        values,
        width=width,
        color=colors[i]
    )

ax.set_xticks(x)
ax.set_xticklabels(["4:1", "8:1", "12:1", "16:1"])

ax.set_ylim(0, 1.0)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.tick_params(axis="x", labelbottom=False)
ax.tick_params(axis="y", labelleft=False)

plt.tight_layout()

output_path = r"ROC_AUC_downsampling_ratio.png"

fig.savefig(
    output_path,
    dpi=600,
    transparent=True,
    bbox_inches="tight"
)

plt.show()