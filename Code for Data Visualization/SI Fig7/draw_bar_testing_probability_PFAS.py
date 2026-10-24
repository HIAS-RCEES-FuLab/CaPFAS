import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = [
    "MS2_dataset_split_SMILES_type5_12_to_1_bin0p01_P_N_pr_auc_after3_predictions.csv",
    "MS2_dataset_split_SMILES_type5_12_to_1_bin0p1_P_N_pr_auc_after3_predictions.csv",
    "MS2_dataset_split_SMILES_type5_12_to_1_bin1_P_N_pr_auc_after3_predictions.csv"
]

for f in files:

    file = f

    df = pd.read_csv(file)

    print("\n", f)

    df_pfas = df[df["label"] == 1].copy()

    probability = df_pfas["probability"]

    total = len(probability)

    # probability > 0.5
    above_05 = (probability > 0.5).sum()
    ratio = above_05 / total

    print("PFAS number:", total)
    print("Probability > 0.5 number:", above_05)
    print("Probability > 0.5 ratio:", f"{ratio:.2%}")

    fig, ax = plt.subplots(figsize=(3, 2))

    bins = np.linspace(0, 1, 21)

    ax.hist(
        probability,
        bins=bins,
        color="#FAADA5",
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_xlim(0, 1.05)

    ax.set_ylim(0, 650)
    ax.set_yticks([0, 200, 400, 600])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    median = probability.median()
    print("Probability median:", f"{median:.4f}")

    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelleft=False)

    plt.tight_layout()

    output_name = f.replace(
        ".csv",
        "_label1_probability_distribution.png"
    )

    output_path = output_name

    fig.savefig(
        output_path,
        dpi=600,
        transparent=True,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)