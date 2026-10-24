import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 13
rcParams['axes.labelsize'] = 13
rcParams['xtick.labelsize'] = 13
rcParams['ytick.labelsize'] = 13

df = pd.read_csv("Fragmentation.csv", encoding="gbk")
x = np.arange(len(df))

colors = {
    'MetFrag': '#E99A9A',
    'CFM-ID 4.0': '#E8C46F',
    'CaPFAS': '#82B6D6'
}

# =========================
# Matched peak number
# =========================
methods = [
    ('MetFrag', 'Metfrag_matched_peak_number'),
    ('CFM-ID 4.0', 'CFM_ID_4_matched_peak_number'),
    ('CaPFAS', 'CaPFAS_matched_peak_number')
]

fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)

for ax, (name, col) in zip(axes, methods):
    y = df[col].values
    mean = y.mean()

    ax.bar(
        x, y,
        color=colors[name],
        alpha=1,
        width=0.8,
        linewidth=0
    )

    ax.axhline(
        mean,
        color=colors[name],
        linestyle='--',
        linewidth=0.8
    )

    ax.set_ylabel("")
    ax.set_ylim(0, 15)
    ax.set_yticks([0, 5, 10, 15])
    ax.tick_params(axis='both', labelsize=13)

    ax.set_facecolor("none")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[-1].set_xlabel("Samples", fontsize=13)

fig.patch.set_alpha(0)
plt.tight_layout()

plt.savefig(
    r"peak_number.png",
    dpi=300,
    bbox_inches='tight',
    transparent=True
)

plt.show()


# =========================
# Matched intensity ratio
# =========================
methods = [
    ('MetFrag', 'Metfrag_matched_intensity_ratio'),
    ('CFM-ID 4.0', 'CFM_ID_4_matched_intensity_ratio'),
    ('CaPFAS', 'CaPFAS_matched_intensity_ratio')
]

fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)

for ax, (name, col) in zip(axes, methods):
    y = df[col].values
    mean = y.mean()

    ax.bar(
        x, y,
        color=colors[name],
        alpha=1,
        width=0.8,
        linewidth=0
    )

    ax.axhline(
        mean,
        color=colors[name],
        linestyle='--',
        linewidth=0.8
    )

    ax.set_ylabel("")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis='both', labelsize=13)

    ax.set_facecolor("none")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[-1].set_xlabel("Samples", fontsize=13)

fig.patch.set_alpha(0)
plt.tight_layout()

plt.savefig(
    r"intensity_ratio.png",
    dpi=300,
    bbox_inches='tight',
    transparent=True
)

plt.show()