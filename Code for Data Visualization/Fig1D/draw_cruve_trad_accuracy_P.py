import os
import pandas as pd
import matplotlib.pyplot as plt

files = {
    "Diagnostic fragments + neutral losses": "fragment_neutral_loss_feature_greedy_process_P.csv",
    "KMD AND": "KMD_AND_fragment_neutral_loss_feature_greedy_process_P.csv",
    "KMD OR": "KMD_OR_fragment_neutral_loss_feature_greedy_process_P.csv"
}

colors = ["#7FA8D1", "#B0DEEF", "#EFA18B"]

fig, ax = plt.subplots(figsize=(3.5, 2.3))

max_feature_num = 0

for (label, filename), color in zip(files.items(), colors):
    input_file = filename
    data = pd.read_csv(input_file)
    accuracy = pd.to_numeric(data["Accuracy"], errors="coerce").dropna().reset_index(drop=True)
    feature_num = range(1, len(accuracy) + 1)
    max_feature_num = max(max_feature_num, len(accuracy))

    ax.plot(
        feature_num,
        accuracy,
        linewidth=1.2,
        color=color,
        label=label
    )

    max_accuracy = accuracy.max()
    print(label, max_accuracy)

    ax.axhline(
        y=max_accuracy,
        color=color,
        linestyle="--",
        linewidth=1,
        alpha=0.7
    )

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(0, max_feature_num)
ax.set_ylim(0, 1.0)

ax.tick_params(axis="x", labelbottom=False, length=0)
ax.tick_params(axis="y", labelleft=False, length=4)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

plt.tight_layout()

output_file = "fragment_neutral_loss_KMD_accuracy_comparison_P.png"
plt.savefig(output_file, dpi=600, transparent=True, bbox_inches="tight")

plt.show()