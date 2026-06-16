import matplotlib.pyplot as plt
import numpy as np

labels = ["site1", "site2"]
values = [350, 101]

colors = ["#D7BDE2", "#AED6F1"]

fig, ax = plt.subplots(figsize=(3, 4))

bars = ax.bar(labels, values, color=colors, width=0.8)

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_yticks(np.linspace(0, 360, 4).astype(int))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

plt.tight_layout()
plt.savefig("Level1_barplot.png", dpi=300, transparent=True)
plt.show()