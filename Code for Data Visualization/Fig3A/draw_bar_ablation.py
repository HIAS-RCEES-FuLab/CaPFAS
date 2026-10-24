import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r"all_test_results_best_pr_auc_after3.csv"
output_file = r"ablation_6models_bar.png"

df = pd.read_csv(file_path)
metrics = ["Accuracy", "ROC_AUC", "PR_AUC"]
order = [2, 0, 1, 4, 5, 3]

colors = ["#85C1E9", "#F5CBA7", "#A9DFBF"]
light_colors = ["#B7DDF2", "#F8D5B3", "#C5E8D0"]

final_model = {
    "Accuracy": 0.880,
    "ROC_AUC": 0.940,
    "PR_AUC": 0.897
}

print(df[metrics])

x = np.arange(len(order))
width = 0.22

fig, ax = plt.subplots(figsize=(5, 4))

for i, metric in enumerate(metrics):
    offset = (i - 1) * width
    positions = x + offset
    values = [df.loc[idx, metric] for idx in order]
    baseline = final_model[metric]

    ax.bar(
        positions,
        [baseline] * len(order),
        width=width,
        color=light_colors[i],
        edgecolor="none",
        zorder=1
    )

    ax.bar(
        positions,
        values,
        width=width,
        color=colors[i],
        edgecolor="black",
        linewidth=0.5,
        zorder=2
    )

ax.set_xticks(x)
ax.set_xticklabels([])
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels([])
ax.tick_params(axis="y", labelleft=True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    output_file,
    dpi=300,
    transparent=True,
    bbox_inches="tight"
)
plt.show()