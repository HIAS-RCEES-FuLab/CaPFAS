import os
import glob
import numpy as np
import re
import pandas as pd

# load kaufmann 95HDR
data = np.load("PFAS_Kaufmann_95HDR.npz")
X, Y, mask = data["X"], data["Y"], data["mask"]

# read pfas csv file
base_dir = r"D:\MSMS\MSMS_H+H-\Train_data"
files = glob.glob(os.path.join(base_dir, "*_PFAS_NEG*.csv"))

print(f"找到 {len(files)} 个 NEG 文件")

def parse_carbon(formula):
    match = re.search(r'C(\d*)', formula)
    if match:
        return int(match.group(1)) if match.group(1) else 1
    return np.nan

total_points = 0
inside_points = 0

file_stats = []

for file in files:
    df = pd.read_csv(file)

    mz_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'mz' in col_lower or 'mass' in col_lower:
            mz_col = col
            break

    df = df[['Formula', mz_col]].copy()
    df.columns = ['Formula', 'mz']

    df['mz'] = pd.to_numeric(df['mz'], errors='coerce')
    df = df.dropna(subset=['Formula', 'mz'])

    df['C'] = df['Formula'].apply(parse_carbon)
    df = df.dropna(subset=['C'])
    df = df[df['C'] > 0]

    if len(df) == 0:
        continue

    mz = df['mz'].values
    C = df['C'].values

    mass_defect = mz - np.round(mz)

    mz_per_C = mz / C
    md_per_C = mass_defect / C

    ix = np.abs(X[:, 0][:, None] - mz_per_C).argmin(axis=0)
    iy = np.abs(Y[0, :][None, :] - md_per_C[:, None]).argmin(axis=1)

    inside = mask[ix, iy]

    file_total = len(df)
    file_inside = inside.sum()

    total_points += file_total
    inside_points += file_inside

    file_stats.append({
        "file": os.path.basename(file),
        "total": file_total,
        "inside": int(file_inside),
        "coverage_%": round(file_inside / file_total * 100, 2)
    })

    print(f"✔ {os.path.basename(file)}: {file_inside}/{file_total} ({file_inside/file_total*100:.2f}%)")

print("\n==========================")
print(f"Total points: {total_points}")
print(f"Points within 95% HDR: {inside_points}")

if total_points > 0:
    print(f"Overall proportion: {inside_points / total_points * 100:.2f}%")

# -------------------------
stats_df = pd.DataFrame(file_stats)
stats_df.to_csv(
    r"D:\MSMS\MSMS_H+H-\Train_data\Kaufmann_coverage_stats.csv",
    index=False,
    encoding="utf-8-sig"
)

summary_df = pd.DataFrame({
    "Metric": ["Total features", "Inside 95% HDR", "Coverage (%)"],
    "Value": [
        total_points,
        inside_points,
        round(inside_points / total_points * 100, 2) if total_points > 0 else 0
    ]
})

summary_df.to_csv(
    r"D:\MSMS\MSMS_H+H-\Train_data\Kaufmann_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

final_table = stats_df.copy()
final_table.loc[len(final_table)] = [
    "ALL",
    total_points,
    inside_points,
    round(inside_points / total_points * 100, 2) if total_points > 0 else 0
]
final_table.to_csv(
    r"D:\MSMS\MSMS_H+H-\Train_data\Kaufmann_final_table.csv",
    index=False,
    encoding="utf-8-sig"
)

all_mz_per_C = []
all_md_per_C = []
all_inside = []

for file in files:
    df = pd.read_csv(file)

    if 'Formula' not in df.columns:
        continue

    mz_col = None
    for col in df.columns:
        if 'mz' in col.lower() or 'mass' in col.lower():
            mz_col = col
            break
    if mz_col is None:
        continue

    df = df[['Formula', mz_col]].copy()
    df.columns = ['Formula', 'mz']
    df['mz'] = pd.to_numeric(df['mz'], errors='coerce')
    df = df.dropna(subset=['Formula', 'mz'])

    df['C'] = df['Formula'].apply(parse_carbon)
    df = df.dropna(subset=['C'])
    df = df[df['C'] > 0]

    if len(df) == 0:
        continue

    mz = df['mz'].values
    C = df['C'].values

    mass_defect = mz - np.round(mz)

    mz_per_C = mz / C
    md_per_C = mass_defect / C

    ix = np.abs(X[:, 0][:, None] - mz_per_C).argmin(axis=0)
    iy = np.abs(Y[0, :][None, :] - md_per_C[:, None]).argmin(axis=1)

    inside = mask[ix, iy]

    all_mz_per_C.extend(mz_per_C)
    all_md_per_C.extend(md_per_C)
    all_inside.extend(inside)

all_mz_per_C = np.array(all_mz_per_C)
all_md_per_C = np.array(all_md_per_C)
all_inside = np.array(all_inside).astype(bool)

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 14
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 14

fig, ax = plt.subplots(figsize=(8, 6))

ax.contourf(
    X, Y, mask,
    levels=[0.5, 1],
    alpha=0.25
)

ax.contour(
    X, Y, mask,
    levels=[0.5],
    linewidths=1.2
)

ax.scatter(
    all_mz_per_C[all_inside],
    all_md_per_C[all_inside],
    s=8,
    alpha=0.7,
    label="Inside 95% HDR"
)

ax.scatter(
    all_mz_per_C[~all_inside],
    all_md_per_C[~all_inside],
    s=8,
    alpha=0.4,
    label="Outside 95% HDR"
)

ax.set_xlabel("m/z / C")
ax.set_ylabel("Mass defect / C")
ax.set_xlim(10, 120)
ax.set_ylim(-0.05, 0.02)
ax.legend()

plt.tight_layout()
plt.savefig(
    r"D:\MSMS\MSMS_H+H-\Train_data\Kaufmann_2D_scatter.png",
    dpi=300,
    bbox_inches='tight',
    transparent=True
)

plt.show()