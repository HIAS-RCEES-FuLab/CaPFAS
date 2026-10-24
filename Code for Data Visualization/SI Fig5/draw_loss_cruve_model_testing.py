import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

plt.rcParams["font.family"] = "Arial"

file_path = r"train_inf"

# =========================
# Select plotting target
# =========================
bin_name = "bin1_P_N"
#bin_name = "bin0p1_P_N"
#bin_name = "bin0p01_P_N"

colors = ["#85C1E9", "#BCBDDC", "#F5CBA7", "#F1948A"]

files = [
    f for f in os.listdir(file_path)
    if f.lower().endswith(".csv")
    and bin_name in f
]

# =========================
# Val Loss
# =========================
fig, ax = plt.subplots(figsize=(3, 2))

for i, f in enumerate(sorted(files)):
    df = pd.read_csv(os.path.join(file_path, f))

    epochs = np.arange(1, len(df) + 1)
    match = re.search(r"(\d+)_to_1", f)
    if match:
        label = f"{match.group(1)}:1"
    else:
        label = os.path.splitext(f)[0]
    ax.plot(
        epochs,
        df["Val_Loss"],
        linewidth=1.5,
        color=colors[i % len(colors)],
        label=label
    )

ax.set_xlim(0, 50)
ax.set_xticks([0, 25, 50])

ax.set_yticks([0.12, 0.18, 0.24, 0.3])
#ax.set_yticks([0.12, 0.24, 0.36, 0.48])
#ax.set_yticks([0.1, 0.4, 0.7, 1])
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.tick_params(axis="x", labelbottom=False)
ax.tick_params(axis="y", labelleft=False)

plt.tight_layout()

output_path = f"Val_Loss_{bin_name}.png"

fig.savefig(
    output_path,
    dpi=600,
    transparent=True,
    bbox_inches="tight"
)

plt.close(fig)