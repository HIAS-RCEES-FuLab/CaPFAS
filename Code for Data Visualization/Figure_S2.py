# =========================
# PFAS Kaufmann Space Builder
# =========================

import pandas as pd
import numpy as np
import re
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# read csv file
df = pd.read_csv("PFAS.csv")
df = df[['MOLECULAR FORMULA', 'MONOISOTOPIC MASS']].dropna()

df['MONOISOTOPIC MASS'] = pd.to_numeric(
    df['MONOISOTOPIC MASS'], errors='coerce'
)
df = df.dropna(subset=['MONOISOTOPIC MASS'])

def parse_carbon(formula):
    match = re.search(r'C(\d*)', formula)
    if match:
        return int(match.group(1)) if match.group(1) else 1
    return np.nan

df['C'] = df['MOLECULAR FORMULA'].apply(parse_carbon)
df = df.dropna(subset=['C'])
df = df[df['C'] > 0]
mz = df['MONOISOTOPIC MASS'].values.astype(float)
C  = df['C'].values

mass_defect = mz - np.round(mz)
df['mz_per_C'] = mz / C
df['md_per_C'] = mass_defect / C
xy = np.vstack([df['mz_per_C'], df['md_per_C']])
kde = gaussian_kde(xy, bw_method=0.01)
xmin, xmax = df['mz_per_C'].min(), df['mz_per_C'].max()
ymin, ymax = df['md_per_C'].min(), df['md_per_C'].max()
X, Y = np.mgrid[
    xmin:xmax:300j,
    ymin:ymax:300j
]
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

# HDR coverage & 95% isoline
dx = (xmax - xmin) / X.shape[0]
dy = (ymax - ymin) / Y.shape[1]
cell_area = dx * dy

Z_flat = Z.ravel()

# density sorted
idx = np.argsort(Z_flat)[::-1]
Z_sorted = Z_flat[idx]

# cumulative probability
cum_prob = np.cumsum(Z_sorted * cell_area)

# 95% HDR threshold
idx_95 = np.searchsorted(cum_prob, 0.95)
idx_95 = min(idx_95, len(Z_sorted)-1)
iso_95 = Z_sorted[idx_95]

print("=== Kaufmann PFAS Reference ===")
print(f"95% density threshold: {iso_95:.4e}")

# percent coverage surface
rank = np.empty_like(idx)
rank[idx] = np.arange(len(idx))

coverage = cum_prob[rank].reshape(Z.shape)
coverage = np.clip(coverage, 0, 1)

Z_1m = 1.0 - coverage   # 1 − percent coverage
from scipy.ndimage import gaussian_filter
Z_1m_smooth = gaussian_filter(Z_1m, sigma=1)

# Kaufmann space
xlim = (0, 120)
ylim = (-0.05, 0.02)

mask = (X >= xlim[0]) & (X <= xlim[1]) & (Y >= ylim[0]) & (Y <= ylim[1])
display_idx_x = (X[:,0] >= xlim[0]) & (X[:,0] <= xlim[1])
display_idx_y = (Y[0,:] >= ylim[0]) & (Y[0,:] <= ylim[1])

X_disp = X[np.ix_(display_idx_x, display_idx_y)]
Y_disp = Y[np.ix_(display_idx_x, display_idx_y)]
Z_disp = Z_1m_smooth[np.ix_(display_idx_x, display_idx_y)]

from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(
    X_disp, Y_disp, Z_disp,
    cmap='viridis',
    linewidth=0,
    antialiased=True,
    alpha=0.9
)

ax.set_xlabel("m/z / C")
ax.set_ylabel("Mass defect / C")
ax.set_zlabel("1 - Percent Coverage")

fig.colorbar(
    surf,
    shrink=0.6,
    aspect=12,
    label="1 - Percent Coverage"
)

plt.tight_layout()
plt.savefig("figure.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()

Z_95_mask = Z >= iso_95   # kde density >= 95% HDR threshold
np.savez(
    "PFAS_Kaufmann_95HDR.npz",
    X=X,
    Y=Y,
    mask=Z_95_mask
)
