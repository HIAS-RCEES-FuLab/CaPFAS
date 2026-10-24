import numpy as np
import matplotlib.pyplot as plt

output_path = r"MS2_mirror_plot.png"

# ==================== First ====================
#mz1 = np.array([np.float32(59.01386), np.float32(90.39813), np.float32(96.96059), np.float32(161.02226), np.float32(177.0463), np.float32(197.05215), np.float32(216.05109), np.float32(217.0583), np.float32(235.04808), np.float32(237.03918), np.float32(237.06467), np.float32(237.08846), np.float32(237.18501), np.float32(281.05426), np.float32(281.1401), np.float32(281.17688), np.float32(281.24753)])

#intensity1 = np.array([np.float32(11331.21), np.float32(10808.062), np.float32(16006.786), np.float32(51676.707), np.float32(45231.484), np.float32(33272.805), np.float32(32153.365), np.float32(71107.34), np.float32(57085.78), np.float32(16933.531), np.float32(620703.94), np.float32(17364.01), np.float32(32186.53), np.float32(243215.42), np.float32(93483.7), np.float32(125263.78), np.float32(144908.88)])

mz1 = np.array([np.float32(50.12486), np.float32(77.44146), np.float32(79.95746), np.float32(80.96534), np.float32(406.9622), np.float32(408.88052), np.float32(426.96967), np.float32(427.22125)])

intensity1 = np.array([np.float32(13634.061), np.float32(11614.586), np.float32(38149.61), np.float32(91698.484), np.float32(59096.37), np.float32(11867.827), np.float32(161407.83), np.float32(14443.356)])

# ==================== Second ===================
#mz2 = np.array([93.045631, 175.029663, 177.045334, 195.035767, 197.051468, 216.049774, 217.057648, 235.048141, 237.063782, 281.053589])

#intensity2 = np.array([7.1, 6.6, 21.2, 10.1, 7.0, 10.0, 11.2, 23.3, 100.0, 16.2])

mz2 = np.array([406.961792, 426.967987])

intensity2 = np.array([470088.59375, 1630643.625])

intensity1_norm = intensity1 / intensity1.max() * 100
intensity2_norm = intensity2 / intensity2.max() * 100

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.linewidth"] = 1.0

fig, ax = plt.subplots(figsize=(6.4, 2.8))

ax.vlines(
    mz1, 0, intensity1_norm,
    color="black", linewidth=1.5,
    label="Spectrum 1"
)

ax.vlines(
    mz2, 0, -intensity2_norm,
    color="red", linewidth=1.5,
    label="Spectrum 2"
)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlim(0, 500)
#ax.set_xlim(0, 350)

ax.set_ylim(-105, 105)

ax.tick_params(axis="both", labelbottom=False, labelleft=False, direction="out")
#ax.legend(frameon=False, fontsize=11, loc="upper left")

plt.tight_layout()

plt.savefig(output_path, dpi=600, bbox_inches="tight", transparent=True)
plt.show()
