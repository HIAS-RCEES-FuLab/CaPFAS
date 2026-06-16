import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

datasets = [
    "GNPS_PFAS", "NIST20_PFAS", "MoNA_PFAS", "Standard",
    "GNPS_non", "NIST20_non", "MoNA_non"
]

pfas_correct = [0.6, 2.8, 16.4, 52.5]
nonpfas_wrong = [0.4, 0.7, 0.3]

correct = []
wrong = []

for v in pfas_correct:
    correct.append(v)
    wrong.append(100 - v)

for v in nonpfas_wrong:
    correct.append(100 - v)
    wrong.append(v)

correct = np.array(correct)
wrong = np.array(wrong)

x = np.arange(len(datasets))

fig, ax = plt.subplots(figsize=(6, 6))

ax.bar(
    x,
    correct,
    label="Correct",
    color="#ff8e8e"
)

ax.bar(
    x,
    wrong,
    bottom=correct,
    label="Incorrect",
    color="#9fb7d6"
)

ax.set_xticks(x)
ax.set_xticklabels([])

ax.set_yticklabels([])
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig("stacked_bar.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()