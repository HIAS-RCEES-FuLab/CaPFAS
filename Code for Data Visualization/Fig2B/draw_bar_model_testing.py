
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r"all_test_results_summary.csv"
df = pd.read_csv(file_path)

df_pn = df[
    (df["ion_mode"] == "P_N") &
    (df["model_version"] == "after3") &
    (df["metric"] == "pr_auc")
].copy()

ratios = sorted(df_pn["ratio_number"].unique())
bin_sizes = sorted(df_pn["bin_size"].unique(), reverse=True)

print("ratio_number:", ratios)
print("bin_size:", bin_sizes)

for metric in ["Accuracy", "PR_AUC", "ROC_AUC"]:
    min_idx = df_pn[metric].idxmin()
    max_idx = df_pn[metric].idxmax()
    min_row = df_pn.loc[min_idx]
    max_row = df_pn.loc[max_idx]
    print(f"\n{metric}:")
    print(f"  min: {min_row[metric]:.4f} (bin size = {min_row['bin_size']}, ratio = {min_row['ratio_number']})")
    print(f"  max: {max_row[metric]:.4f} (bin size = {max_row['bin_size']}, ratio = {max_row['ratio_number']})")

def plot_metric(metric, output_name):
    x = np.arange(len(ratios))
    width = 0.8 / len(bin_sizes)
    colors = ["#85C1E9", "#F5CBA7", "#F1948A"]
    y_limits = {
        "Accuracy": (0.2, 1.0),
        "PR_AUC": (0.2, 1.0),
        "ROC_AUC": (0.4, 1.0)
    }

    fig, ax = plt.subplots(figsize=(4, 2))

    for i, bin_size in enumerate(bin_sizes):
        values = []
        for ratio in ratios:
            row = df_pn[
                (df_pn["ratio_number"] == ratio) &
                (df_pn["bin_size"] == bin_size)
            ]
            values.append(row[metric].iloc[0])

        ax.bar(
            x + (i - (len(bin_sizes) - 1) / 2) * width,
            values,
            width=width,
            color=colors[i]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ratios)
    ax.set_ylim(y_limits[metric])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelleft=False)

    plt.tight_layout()
    fig.savefig(output_name, dpi=600, transparent=True, bbox_inches="tight")
    plt.show()

plot_metric("Accuracy", "Accuracy_PN_after3.png")
plot_metric("PR_AUC", "PR_AUC_PN_after3.png")
plot_metric("ROC_AUC", "ROC_AUC_PN_after3.png")
