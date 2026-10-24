import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"

files = {
    "N_MS1": "PFAS_SHAP_N_MS1_all_features.csv",
    "P_MS1": "PFAS_SHAP_P_MS1_all_features.csv",
    "N_mass": "PFAS_SHAP_N_mass_difference_neutral_loss_all_nonzero_with_count.csv",
    "P_mass": "PFAS_SHAP_P_mass_difference_neutral_loss_all_nonzero_with_count.csv",
    "N_seq": "PFAS_SHAP_N_MS2_fragment_sequence_all_nonzero_with_count.csv",
    "P_seq": "PFAS_SHAP_P_MS2_fragment_sequence_all_nonzero_with_count.csv"
}

colors = {
    "MS1": "#EFCB68",
    "MS2 fragment sequence": "#7FA8D1",
    "MS2 mass difference / neutral loss": "#EFA18B"
}

def plot_mode(mode):
    if mode == "N":
        file_list = [
            ("MS1", files["N_MS1"], "Feature"),
            ("MS2 fragment sequence", files["N_seq"], "Bin"),
            ("MS2 mass difference / neutral loss", files["N_mass"], "Bin")
        ]
    else:
        file_list = [
            ("MS1", files["P_MS1"], "Feature"),
            ("MS2 fragment sequence", files["P_seq"], "Bin"),
            ("MS2 mass difference / neutral loss", files["P_mass"], "Bin")
        ]

    data_list = []

    for feature_type, file, col in file_list:
        df = pd.read_csv(file,encoding="gbk")[[col, "Mean_abs_SHAP"]].copy()

        if feature_type == "MS2 fragment sequence":
            df["Feature"] = "Seq_" + df[col].astype(str)
        elif feature_type == "MS2 mass difference / neutral loss":
            df["Feature"] = "Diff_" + df[col].astype(str)
        else:
            df["Feature"] = df[col].astype(str)

        df["Feature_type"] = feature_type
        data_list.append(df[["Feature", "Mean_abs_SHAP", "Feature_type"]])

    df_all = (
        pd.concat(data_list, ignore_index=True)
        .sort_values("Mean_abs_SHAP", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    df_all = df_all.iloc[::-1]

    fig, ax = plt.subplots(figsize=(5, 4))

    bar_colors = [colors[x] for x in df_all["Feature_type"]]

    ax.barh(
        range(len(df_all)),
        df_all["Mean_abs_SHAP"],
        height=0.8,
        color=bar_colors
    )

    ax.set_yticks(range(len(df_all)))
    ax.set_xticks(ax.get_xticks())

    ax.tick_params(
        axis="both",
        which="both",
        length=4,
        width=0.8,
        labelbottom=False,
        labelleft=False
    )
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlim(0, 0.20)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    plt.tight_layout()

    plt.savefig(
        f"PFAS_SHAP_{mode}_mode_top10_horizontal.png",
        dpi=600,
        transparent=True,
        bbox_inches="tight"
    )

    plt.show()

plot_mode("N")
plot_mode("P")