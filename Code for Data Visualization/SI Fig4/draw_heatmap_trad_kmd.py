import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = {
    "KMD": r"D:\论文二修改\原始数据\数据集传统方法对比验证\KMD_range_scan_spectrum_compound.csv",
    "KMD_norm": r"D:\论文二修改\原始数据\数据集传统方法对比验证\KMD_norm_range_scan_spectrum_compound.csv"
}

def plot_cm(cm, filename):
    cm_plot = cm[[1, 0]][:, [1, 0]]
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both", length=4, width=1)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.vlines(0.5, -0.5, 1.5, colors="black", linewidth=1)
    ax.hlines(0.5, -0.5, 1.5, colors="black", linewidth=1)
    ax.vlines([-0.5, 1.5], -0.5, 1.5, colors="black", linewidth=1)
    ax.hlines([-0.5, 1.5], -0.5, 1.5, colors="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(filename, dpi=600, transparent=True, bbox_inches="tight")
    plt.show()
    plt.close()

for method, file in files.items():
    df = pd.read_csv(file)

    # =========================
    # Compound水平确定最佳边界
    # =========================
    compound = df[df["Level"] == "Compound"].copy()
    compound["TPR_minus_FPR"] = compound["TPR"] - compound["FPR"]
    best_compound = compound.loc[compound["TPR_minus_FPR"].idxmax()]

    kmd_lower = best_compound["KMD_lower"]
    kmd_upper = best_compound["KMD_upper"]

    print("\n======================")
    print(method)
    print("Compound optimal boundary")
    print("======================")
    print("KMD_lower:", kmd_lower)
    print("KMD_upper:", kmd_upper)

    # =========================
    # 两个层级均使用Compound边界
    # =========================
    for level in ["Spectrum", "Compound"]:
        sub = df[
            (df["Level"] == level) &
            (df["KMD_lower"] == kmd_lower) &
            (df["KMD_upper"] == kmd_upper)
        ].copy()

        if len(sub) == 0:
            print("未找到:", level)
            continue

        best = sub.iloc[0]

        print("\n----------------------")
        print(level)
        print("----------------------")
        print("KMD_lower:", best["KMD_lower"])
        print("KMD_upper:", best["KMD_upper"])
        print("TP:", best["TP"])
        print("TN:", best["TN"])
        print("FP:", best["FP"])
        print("FN:", best["FN"])
        print("TPR:", best["TPR"])
        print("FPR:", best["FPR"])

        cm = np.array([
            [best["TN"], best["FP"]],
            [best["FN"], best["TP"]]
        ])

        plot_cm(cm, f"{method}_{level}_CM.png")