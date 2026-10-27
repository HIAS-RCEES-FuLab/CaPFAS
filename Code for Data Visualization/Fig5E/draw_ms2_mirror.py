import numpy as np
import matplotlib.pyplot as plt

output_path = r"MS2_mirror_plot.png"

# ==================== First ====================
mz1 = np.array([59.01378, 66.58502, 127.40117, 163.11087, 177.04575, 187.30441, 197.05258, 216.05006, 217.05827, 219.17468, 227.24715, 235.04962, 236.05315, 237.06456, 237.15218, 237.1849, 238.06863, 263.16733, 280.16296, 280.19885, 281.05453, 281.09833, 281.1396, 281.1763, 281.2175, 281.2483, 282.05615, 282.14264, 282.17975, 282.25162, 283.26437])

intensity1 = np.array([10712.985, 8435.212, 9376.13, 9090.312, 26481.242, 10031.497, 10851.451, 20069.68, 86379.95, 22264.373, 26866.219, 49706.344, 8673.232, 793146.0, 10366.548, 27953.326, 54379.688, 9842.995, 30796.545, 14149.626, 688138.2, 14796.977, 69020.99, 231335.3, 18710.365, 392168.56, 58649.58, 12633.329, 15203.273, 37569.59, 96499.875])

#mz1 = np.array([np.float32(50.12486), np.float32(77.44146), np.float32(79.95746), np.float32(80.96534), np.float32(406.9622), np.float32(408.88052), np.float32(426.96967), np.float32(427.22125)])

#intensity1 = np.array([np.float32(13634.061), np.float32(11614.586), np.float32(38149.61), np.float32(91698.484), np.float32(59096.37), np.float32(11867.827), np.float32(161407.83), np.float32(14443.356)])

# ==================== Second ===================
mz2 = np.array([np.float32(177.04555), np.float32(197.05247), np.float32(216.05022), np.float32(217.05818), np.float32(235.04886), np.float32(237.0644), np.float32(238.06757), np.float32(281.05435), np.float32(282.0573)])

intensity2 = np.array([np.float32(1488089.6), np.float32(1855136.4), np.float32(975208.75), np.float32(6762463.0), np.float32(3223981.5), np.float32(63351412.0), np.float32(4832566.5), np.float32(58658196.0), np.float32(4958723.0)])

#mz2 = np.array([406.961792, 426.967987])

#intensity2 = np.array([470088.59375, 1630643.625])

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
#ax.set_xlim(0, 500)
ax.set_xlim(0, 350)

ax.set_ylim(-105, 105)

ax.tick_params(axis="both", labelbottom=False, labelleft=False, direction="out")
#ax.legend(frameon=False, fontsize=11, loc="upper left")

plt.tight_layout()

plt.savefig(output_path, dpi=600, bbox_inches="tight", transparent=True)
plt.show()