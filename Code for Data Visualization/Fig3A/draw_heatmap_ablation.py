import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

file_path = r"all_test_results_best_pr_auc_after3.csv"
output_file = r"ablation_6models_delta_heatmap.png"

df = pd.read_csv(file_path)
metrics = ["Accuracy", "ROC_AUC", "PR_AUC"]
order = [2, 0, 1, 4, 5, 3]

final_model = {
    "Accuracy": 0.880,
    "ROC_AUC": 0.940,
    "PR_AUC": 0.897
}

delta_data = np.array([
    [df.loc[idx, metric] - final_model[metric] for metric in metrics]
    for idx in order
])

print(pd.DataFrame(
    delta_data,
    index=[f"Model {i}" for i in order],
    columns=metrics
).round(4))

plt.rcParams["font.family"] = "Arial"
fig, ax = plt.subplots(figsize=(4.2, 3.2))

max_abs = np.max(np.abs(delta_data))
if max_abs == 0:
    max_abs = 0.001

norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

soft_cmap = LinearSegmentedColormap.from_list(
    "soft_blue_white_orange",
    ["#A9C9E2", "#F7F7F5", "#E9B99D"]
)

im = ax.imshow(
    delta_data,
    cmap=soft_cmap,
    norm=norm,
    aspect="auto"
)

ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False
)

ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
ax.grid(
    which="minor",
    color="black",
    linestyle="-",
    linewidth=0.5
)
ax.tick_params(which="minor", bottom=False, left=False)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(0.5)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()

plt.savefig(
    output_file,
    dpi=300,
    transparent=True,
    bbox_inches="tight",
    pad_inches=0.02
)
plt.show()
