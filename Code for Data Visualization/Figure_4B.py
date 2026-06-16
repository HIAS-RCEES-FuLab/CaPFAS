import pandas as pd
import matplotlib.pyplot as plt

# file path
csv_file = r"D:\MSMS\MSMS_H+H-\Train_data\feature_dimensions_CF2_CF_fragments_summary_merged.csv"
df = pd.read_csv(csv_file)

if "File" in df.columns:
    df = df[df["File"].str.contains("PFAS_NEG")]

colors = ['#8dd3c7' if 'nonPFAS' in fname else '#ffffb3' for fname in df['File']]

# ----------- KMD -----------
kmd_cols = ["KMD_q05", "KMD_q25", "KMD_q50", "KMD_q75", "KMD_q95"]
kmd_summary = df[kmd_cols].values
kmd_mean = df["KMD_mean"].values

kmd_boxes = []
for row in kmd_summary:
    d = {
        'whislo': row[0],
        'q1': row[1],
        'med': row[2],
        'q3': row[3],
        'whishi': row[4],
        'fliers': []
    }
    kmd_boxes.append(d)

fig, ax = plt.subplots(figsize=(4, 3), facecolor='none')
bx = ax.bxp(kmd_boxes, showfliers=False, patch_artist=True)

for patch, color in zip(bx['boxes'], colors):
    patch.set_facecolor(color)
for line in bx['medians']:
    line.set_color('black')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("KMD_boxplot.png", transparent=True)
plt.show()

# ----------- KMD_to_Exact_mass -----------
kmd_exact_cols = ["KMD_to_Exact_mass_q05", "KMD_to_Exact_mass_q25",
                  "KMD_to_Exact_mass_q50", "KMD_to_Exact_mass_q75", "KMD_to_Exact_mass_q95"]
kmd_exact_summary = df[kmd_exact_cols].values * 1000

kmd_exact_boxes = []
for row in kmd_exact_summary:
    d = {
        'whislo': row[0],
        'q1': row[1],
        'med': row[2],
        'q3': row[3],
        'whishi': row[4],
        'fliers': []
    }
    kmd_exact_boxes.append(d)

fig, ax = plt.subplots(figsize=(4, 3), facecolor='none')
bx = ax.bxp(kmd_exact_boxes, showfliers=False, patch_artist=True)

for patch, color in zip(bx['boxes'], colors):
    patch.set_facecolor(color)
for line in bx['medians']:
    line.set_color('black')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("KMD_to_Exact_mass_boxplot.png", transparent=True)
plt.show()