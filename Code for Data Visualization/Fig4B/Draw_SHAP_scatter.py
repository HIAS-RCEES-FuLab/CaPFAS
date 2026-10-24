import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = "Arial"

files = {
    "N_mass_difference_neutral_loss": "PFAS_SHAP_N_mass_difference_neutral_loss_all_nonzero_with_count.csv",
    "P_mass_difference_neutral_loss": "PFAS_SHAP_P_mass_difference_neutral_loss_all_nonzero_with_count.csv",
    "N_MS2_fragment_sequence": "PFAS_SHAP_N_MS2_fragment_sequence_all_nonzero_with_count.csv",
    "P_MS2_fragment_sequence": "PFAS_SHAP_P_MS2_fragment_sequence_all_nonzero_with_count.csv"
}

for name, csv_path in files.items():
    df = pd.read_csv(csv_path, encoding="gbk")
    df = df[df["Detected_samples"] > 0].copy()

    plot_df = df.sort_values("Mean_abs_SHAP", ascending=True).reset_index(drop=True)

    shap_min = plot_df["Mean_abs_SHAP"].quantile(0.001)
    shap_max = plot_df["Mean_abs_SHAP"].quantile(0.999)

    print(f"{name}:")
    print(f"  0.1th percentile  = {shap_min:.6f}")
    print(f"  99.9th percentile = {shap_max:.6f}")
    print(f"  N fragments       = {len(plot_df)}")
    print()

    plot_df.to_csv(
        f"{name}_SHAP_plot_data.csv",
        index=False
    )

    plt.figure(figsize=(6, 4))

    plt.scatter(
        plot_df["Bin"],
        plot_df["Detected_samples"],
        c=plot_df["Mean_abs_SHAP"],
        cmap="coolwarm",
        norm=Normalize(
            vmin=shap_min,
            vmax=shap_max
        ),
        s=160,
        edgecolor="black",
        linewidth=0.6
    )

    plt.xticks([0, 20000, 40000, 60000, 80000, 100000])

    plt.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"{name}_SHAP_Bin_scatter_no_cbar.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {name}_SHAP_Bin_scatter_no_cbar.png\n")