import matplotlib.pyplot as plt
import numpy as np

labels = ["Site1", "Site2"]
# Total number of compounds
#values = [508, 158]
# Level 1
#values = [19, 12]
# Level 2
values = [196, 56]
# Level 3
#values = [293, 90]
colors = ["#D7BDE2", "#AED6F1"]

fig, ax = plt.subplots(figsize=(2.4, 3.6))
x = np.arange(len(labels))
bars = ax.bar(x, values, width=0.55, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlim(-0.5, 1.5)
ax.set_xticks(x)
ax.set_xticklabels([])
#ax.set_yticks(np.linspace(0, 510, 4))
#ax.set_yticks(np.linspace(0, 20, 5))
#ax.set_yticks(np.linspace(0, 1, 2))
ax.set_yticks(np.linspace(0, 210, 4))
#ax.set_yticks(np.linspace(0, 300, 4))
ax.set_yticklabels([])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("Level1_barplot.png", dpi=300, transparent=True)
plt.show()
