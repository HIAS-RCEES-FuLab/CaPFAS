import matplotlib.pyplot as plt
import numpy as np

values = [374, 111, 439, 116]

sub_values = [103, 77, 74, 44]

colors_main = ["#F4C5C5", "#B8CDE8", "#F4C5C5", "#B8CDE8"]

colors_sub = ["#D97575", "#7099CC", "#D97575", "#7099CC"]

labels = ["Influent (Method1)", "Influent (Method2)", "Effluent (Method1)", "Effluent (Method2)"]

fig, ax = plt.subplots(figsize=(4, 4))

for i in range(len(values)):
    ax.bar(i, sub_values[i], color=colors_sub[i], width=0.6)
    ax.bar(i, values[i]-sub_values[i], bottom=sub_values[i], color=colors_main[i], width=0.6)

ax.set_yticks([0, 150, 300, 450])
ax.set_ylim(0, 450)
ax.set_yticklabels([])
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

mid = 1.5
ax.axvline(mid, color='gray', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig("Barplot_4bars_stacked_swapped.png", dpi=300, transparent=True)
plt.show()