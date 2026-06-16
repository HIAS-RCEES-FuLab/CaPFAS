import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'Arial'

df = pd.read_csv(
    r"result.csv",
    encoding="utf-8-sig"
)

x = np.arange(len(df))

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

methods = [
    ('MetFrag', 'Metfrag_matched_peak_number(contain percursor peak)'),
    ('CFM-ID 4.0', 'CFM_ID_4_matched_peak_number(contain percursor peak)'),
    ('CaPFAS', 'CaPFAS_matched_peak_number(contain percursor peak)')
]

ymax = max(df[col].max() for _, col in methods)

colors = {
    'MetFrag': '#b6e3b6',
    'CFM-ID 4.0': '#ffd27f',
    'CaPFAS': '#accde1'
}

for ax, (name, col) in zip(axes, methods):
    ax.bar(x, df[col], color=colors[name])
    ax.axhline(df[col].mean(), linestyle='--', color=colors[name])

    ax.set_ylabel(name)
    ax.set_ylim(0, ymax * 1.1)

axes[-1].set_xlabel("Samples")

for ax in axes:
    ax.set_facecolor("none")

fig.patch.set_alpha(0)

plt.tight_layout()
plt.savefig(
    r"peak_number.png",
    dpi=300,
    bbox_inches='tight',
    transparent=True
)
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

methods = [
    ('MetFrag', 'Metfrag_matched_intensity_ratio(contain percursor peak)'),
    ('CFM-ID 4.0', 'CFM_ID_4_matched_intensity_ratio(contain percursor peak)'),
    ('CaPFAS', 'CaPFAS_matched_intensity_ratio(contain percursor peak)')
]

for ax, (name, col) in zip(axes, methods):
    ax.bar(x, df[col], color=colors[name])
    ax.axhline(df[col].mean(), linestyle='--', color=colors[name])

    ax.set_ylabel(name)

axes[-1].set_xlabel("Samples")

for ax in axes:
    ax.set_facecolor("none")

fig.patch.set_alpha(0)

plt.tight_layout()
plt.savefig(
    r"intensity_ratio.png",
    dpi=300,
    bbox_inches='tight',
    transparent=True
)
plt.show()