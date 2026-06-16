import matplotlib.pyplot as plt

# data
features = [
    "KMD_ratio", "nl_200", "bin_189", "KMD", "Exact_mass_norm",
    "nl_400", "nl_499", "bin_349", "nl_700", "bin_289", "nl_949"
]
shap_values = [
    1.6894, 1.0153, 0.7937, 0.6668, 0.4506,
    0.4322, 0.3017, 0.2438, 0.2195, 0.1792, 0.1789
]

colors = []
for f in features:
    if f.startswith("bin"):
        colors.append("#a6cee3")      # fragment-level
    elif f.startswith("nl"):
        colors.append("#fed9a6")      # neutral loss
    else:
        colors.append("#fbb4ae")      # precursor-level

fig, ax = plt.subplots(figsize=(8,6), facecolor='none')  # 背景透明
y_pos = range(len(features))
ax.barh(y_pos, shap_values, color=colors, edgecolor='black', linewidth=1)  # 加边框
ax.set_yticks(y_pos)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.invert_yaxis()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('')
ax.set_ylabel('')

plt.tight_layout()

plt.savefig("shap_barplot.png", dpi=300, transparent=True)
plt.show()