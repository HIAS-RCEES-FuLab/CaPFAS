import pandas as pd
import matplotlib.pyplot as plt

input_file = r"fragment_neutral_loss_feature_greedy_process_N.csv"
output_file = r"fragment_neutral_loss_feature_process_N_recall.png"

data = pd.read_csv(input_file)
recall = pd.to_numeric(data["Recall"], errors="coerce").dropna().reset_index(drop=True)

feature_num = range(1, len(recall) + 1)

plt.figure(figsize=(3.5, 2.3))
plt.bar(feature_num, recall, width=0.8)

max_recall = recall.max()

plt.axhline(
    y=max_recall,
    color="gray",
    linestyle="--",
    linewidth=0.8
)

plt.xlabel("")
plt.ylabel("")

plt.xticks([])

ax = plt.gca()
ax.tick_params(axis="y", labelleft=False)

plt.ylim(0, 1.0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(output_file, dpi=600, transparent=True, bbox_inches="tight")
plt.show()

# 输出最大 Recall
print("=" * 40)
print(f"Maximum Recall: {max_recall:.4f}")
print(f"Feature number: {recall.idxmax() + 1}")
print("=" * 40)
