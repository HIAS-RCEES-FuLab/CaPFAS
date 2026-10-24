import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r"all_traditional_ML_test_results.csv"

df = pd.read_csv(file_path)

print(df[["Test_Accuracy", "Test_ROC_AUC", "Test_PR_AUC"]])

for metric in ["Test_Accuracy", "Test_ROC_AUC", "Test_PR_AUC"]:
    print(f"\n{metric}:")
    print(df[metric].values[:5])

metrics = ["Test_Accuracy", "Test_ROC_AUC", "Test_PR_AUC"]

# =========================
# CaPFAS
# =========================
model_7 = {
    "Test_Accuracy": 0.880,
    "Test_PR_AUC": 0.897,
    "Test_ROC_AUC": 0.940
}

order = [0, 1, 2, 3, 4]

colors = [
    "#85C1E9",
    "#F5CBA7",
    "#A9DFBF",
    "#D7BDE2",
    "#FFE699",
    "#F1948A"
]

x = np.arange(3)
width = 0.11

fig, ax = plt.subplots(figsize=(8, 3.6))

for i, row_idx in enumerate(order):
    values = [df[metric].values[row_idx] for metric in metrics]
    offset = (i - 2.5) * width

    ax.bar(
        x + offset,
        values,
        width=width,
        color=colors[i],
        edgecolor="black",
        linewidth=0.5
    )

# CaPFAS
values_7 = [model_7[metric] for metric in metrics]
offset = (5 - 2.5) * width

ax.bar(
    x + offset,
    values_7,
    width=width,
    color=colors[5],
    edgecolor="black",
    linewidth=0.5
)

ax.set_xticks(x)
ax.set_xticklabels([])

ax.set_ylim(0, 1.0)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.tick_params(axis="y", labelleft=False)

plt.tight_layout()

output_file = r"traditional_ML_comparison.png"
plt.savefig(
    output_file,
    dpi=300,
    transparent=True,
    bbox_inches="tight"
)
plt.show()