import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 13
rcParams['axes.labelsize'] = 13
rcParams['xtick.labelsize'] = 13
rcParams['ytick.labelsize'] = 13
rcParams['legend.fontsize'] = 13

standard_file = r"standard_CFM_cosine_result.csv"
df = pd.read_csv(standard_file, encoding="gbk")

columns = {
    "MetFrag": "MetFrag_rank",
    "CFM-ID": "Target_CFM_rank",
    "CaPFAS": "SMILES_intensity_rank"
}

colors = {
    "CaPFAS": "#82B6D6",
    "MetFrag": "#E99A9A",
    "CFM-ID": "#EBD08A"
}

labels = ["Top 50", "Top 10", "Top 5", "Top 1"]
results = {}

for name, col in columns.items():
    values = pd.to_numeric(df[col], errors="coerce")
    results[name] = [
        ((values <= 50) & values.notna()).sum(),
        ((values <= 10) & values.notna()).sum(),
        ((values <= 5) & values.notna()).sum(),
        (values == 1).sum()
    ]

total = len(df)
print(f"Total standards: {total}\n")

for name in columns:
    print(f"{name}:")
    for label, count in zip(labels, results[name]):
        ratio = count / total * 100
        print(f"  {label}: {count}/{total} ({ratio:.1f}%)")
    print()

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(5, 5))

offsets = {
    "MetFrag": -width,
    "CFM-ID": 0,
    "CaPFAS": width
}

for name in columns:
    ax.bar(
        x + offsets[name],
        results[name],
        width,
        label=name,
        color=colors[name]
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13)
ax.set_ylabel("Number of correct structures", fontsize=13)
ax.set_xlabel("Candidate rank", fontsize=13)
ax.tick_params(axis='both', labelsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(r"candidate_ranking.png", dpi=300, bbox_inches="tight", transparent=True)
plt.show()