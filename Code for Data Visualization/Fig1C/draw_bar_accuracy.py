import matplotlib.pyplot as plt

#values = [69.6, 68.8]
values = [55.6, 63.4]
labels = ["Spectrum level", "Compound level"]

colors = [
    "#7FA8D1",
    "#B0DEEF"
]

fig, ax = plt.subplots(figsize=(1.2, 2), dpi=300)

ax.bar(
    labels,
    values,
    color=colors,
    width=0.8
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    axis="y",
    labelleft=False,
    length=4
)

ax.tick_params(
    axis="x",
    labelbottom=False,
    length=4
)

ax.set_yticks([0, 50, 100])
ax.set_ylim(0, 100)

fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

plt.tight_layout()

output_path = r"accuracy_spectrum_compound.png"

fig.savefig(
    output_path,
    dpi=600,
    transparent=True,
    bbox_inches="tight"
)

plt.show()