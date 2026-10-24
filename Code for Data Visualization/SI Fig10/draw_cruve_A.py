import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =========================
# data
# =========================
thresholds = [0.1, 0.3, 0.5]

previous_accuracy = [0.635, 0.641, 0.627]
our_accuracy = [0.381, 0.458, 0.606]

previous_tpr = [0.397, 0.064, 0.006]
our_tpr = [0.965, 0.762, 0.437]

previous_fpr = [0.225, 0.018, 0.005]
our_fpr = [0.964, 0.722, 0.294]

rcParams["font.family"] = "Arial"

# =========================
# Plotting
# =========================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

datasets = [
    ("Accuracy", previous_accuracy, our_accuracy),
    ("TPR", previous_tpr, our_tpr),
    ("FPR", previous_fpr, our_fpr)
]

for ax, (ylabel, previous, ours) in zip(axes, datasets):

    ax.plot(
        thresholds,
        previous,
        marker="o",
        markersize=7,
        linewidth=2,
        label="Previous study dataset"
    )

    ax.plot(
        thresholds,
        ours,
        marker="o",
        markersize=7,
        linewidth=2,
        label="Our PFAS MS/MS dataset"
    )

    ax.set_xlabel("Similarity threshold", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)

    ax.set_xticks(thresholds)
    ax.tick_params(axis="both", labelsize=13)

    ax.set_ylim(0, 1.0)

    # 去掉上、右边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.4
    )

# =========================
# 只在第一个 panel 添加 legend
# =========================
axes[0].legend(
    fontsize=11,
    frameon=False,
    loc="best"
)

plt.tight_layout()

# =========================
# 保存
# =========================
plt.savefig(
    "Molecular_network_performance_comparison.png",
    dpi=600,
    bbox_inches="tight",
    transparent=True
)

plt.show()