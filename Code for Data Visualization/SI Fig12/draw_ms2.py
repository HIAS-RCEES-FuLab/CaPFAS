import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input_file = r"C:\Users\13709\Desktop\Standard_matched.csv"
output_dir = r"C:\Users\13709\Desktop\MS2_mirror_plots"
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(input_file, encoding="utf-8-sig")

required_cols = ["ID", "Source", "MS2_mz", "MS2_Intensity"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"文件缺少列：{col}")

def parse_array(value):
    if pd.isna(value):
        return np.array([], dtype=float)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    value = str(value).strip()
    if not value:
        return np.array([], dtype=float)
    pattern = r'np\.float\d+\(\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*\)|([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)'
    return np.asarray([float(a if a else b) for a, b in re.findall(pattern, value)], dtype=float)

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "axes.linewidth": 1.0,
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

for compound_id, group in df.groupby("ID", sort=False):
    source_col = group["Source"].astype(str).str.strip().str.upper()
    std_rows = group[source_col.eq("STD")]
    sample_rows = group[source_col.eq("SAMPLE")]

    if len(std_rows) != 1 or len(sample_rows) != 1:
        print(f"跳过 ID={compound_id}：STD={len(std_rows)} 条，Sample={len(sample_rows)} 条")
        continue

    sample, std = sample_rows.iloc[0], std_rows.iloc[0]
    mz_sample = parse_array(sample["MS2_mz"])
    int_sample = parse_array(sample["MS2_Intensity"])
    mz_std = parse_array(std["MS2_mz"])
    int_std = parse_array(std["MS2_Intensity"])

    if len(mz_sample) != len(int_sample) or len(mz_std) != len(int_std):
        print(f"跳过 ID={compound_id}：质荷比与强度数组长度不一致")
        continue

    valid_sample = np.isfinite(mz_sample) & np.isfinite(int_sample) & (mz_sample >= 0) & (int_sample >= 0)
    valid_std = np.isfinite(mz_std) & np.isfinite(int_std) & (mz_std >= 0) & (int_std >= 0)
    mz_sample, int_sample = mz_sample[valid_sample], int_sample[valid_sample]
    mz_std, int_std = mz_std[valid_std], int_std[valid_std]

    if len(int_sample) and int_sample.max() > 0:
        int_sample = int_sample / int_sample.max() * 100
    if len(int_std) and int_std.max() > 0:
        int_std = int_std / int_std.max() * 100

    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.vlines(mz_sample, 0, int_sample, color="black", linewidth=1.2)
    ax.vlines(mz_std, 0, -int_std, color="red", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)

    all_mz = np.concatenate([mz_sample, mz_std])
    ax.set_xlim(0, max(100, np.ceil(all_mz.max() / 50) * 50) if len(all_mz) else 500)
    ax.set_ylim(-105, 105)
    ax.set_xlabel("m/z", fontname="Arial", fontsize=10)
    ax.set_ylabel("Intensity (%)", fontname="Arial", fontsize=10)
    ax.tick_params(axis="both", labelbottom=True, labelleft=True, direction="out", width=1.0, length=3, labelsize=10)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")

    plt.tight_layout()
    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(compound_id))
    output_path = os.path.join(output_dir, f"{safe_id}_MS2_mirror.png")
    plt.savefig(output_path, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"已保存：{output_path}")

print(f"\n全部完成，图片保存目录：{output_dir}")