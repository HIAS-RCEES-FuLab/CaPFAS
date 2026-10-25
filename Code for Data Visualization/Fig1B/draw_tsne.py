import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.manifold import TSNE

pfas_file = r"PFAS_structure_annotation_scaffold_CF2_PFC.csv"
standard_file = r"Standard_unique.csv"
output_png = r"PFAS_structure_tSNE_PFC_no_standard.png"
output_csv = r"PFAS_structure_tSNE_PFC_no_standard.csv"

# =========================
# READ CSV FILE
# =========================
df = pd.read_csv(pfas_file, encoding='utf-8', low_memory=False)
df["Canonical_SMILES"] = df["Canonical_SMILES"].fillna("").astype(str).str.strip()
df = df[df["Canonical_SMILES"] != ""].copy()
print(f"PFAS record: {len(df):,}")

standard_df = pd.read_csv(standard_file, encoding='utf-8', low_memory=False)
standard_df["Canonical_SMILES"] = standard_df["Canonical_SMILES"].fillna("").astype(str).str.strip()
standard_df = standard_df[standard_df["Canonical_SMILES"] != ""].copy()
print(f"Standard record: {len(standard_df):,}")

# =========================
# REMOVE OVERLAP
# =========================
standard_smiles = set(standard_df["Canonical_SMILES"])
before_remove = len(df)
df = df[~df["Canonical_SMILES"].isin(standard_smiles)].copy()
removed_standard = before_remove - len(df)

# =========================
# REMOVE DEPLICATE
# =========================
before_unique = len(df)
df = df.drop_duplicates(subset="Canonical_SMILES", keep="first").copy()
removed_internal = before_unique - len(df)

# =========================
# PFC NUMBER
# =========================
print(f"\nPFAS PFC = 1: {(df['PFC'] == 1).sum():,}")
print(f"PFAS PFC = 0: {(df['PFC'] == 0).sum():,}")

# =========================
# MORGAN FINGERPRINT
# =========================
radius = 2
fp_size = 1024
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=fp_size
    )
    arr = np.zeros(fp_size, dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr

pfas_fps = []
for i, smiles in enumerate(df["Canonical_SMILES"]):
    pfas_fps.append(smiles_to_fp(smiles))
    if (i + 1) % 100000 == 0:
        print(f"  PFAS: {i + 1:,}/{len(df):,}")

pfas_valid = np.array([fp is not None for fp in pfas_fps])

print(f"\nPFAS success: {pfas_valid.sum():,}, fail: {(~pfas_valid).sum():,}")

df = df.iloc[np.where(pfas_valid)[0]].copy()
pfas_fps = [fp for fp in pfas_fps if fp is not None]
all_fps = np.vstack(pfas_fps).astype(np.float32)
print(f"\nPFAS fingerprints: {len(all_fps):,}")
print(f"Fingerprint shape: {all_fps.shape}")

# =========================
# t-SNE
# =========================
print("\nStart t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="random",
    max_iter=1000,
    random_state=42,
    metric="jaccard"
)

embedding = tsne.fit_transform(all_fps)

print("t-SNE finish")

# =========================
# Save t-SNE coordinate
# =========================
df["tSNE_1"] = embedding[:, 0]
df["tSNE_2"] = embedding[:, 1]
df["Dataset"] = "PFAS"
df["tSNE_Group"] = np.where(
    df["PFC"] == 1,
    "PFC",
    "Other_PFAS"
)

output_df = df[[
    "Canonical_SMILES",
    "Perfluoroalkyl_chain",
    "Scaffold",
    "PFC",
    "tSNE_1",
    "tSNE_2",
    "Dataset",
    "tSNE_Group"
]].copy()

output_df.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)

# =========================
# PLOT
# =========================
blue = "#6F9FC2"
red = "#E57373"

mask_blue = df["PFC"] == 0
mask_red = df["PFC"] == 1

fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(
    df.loc[mask_blue, "tSNE_1"],
    df.loc[mask_blue, "tSNE_2"],
    s=15,
    c=blue,
    alpha=0.9,
    linewidths=0,
    rasterized=True
)

ax.scatter(
    df.loc[mask_red, "tSNE_1"],
    df.loc[mask_red, "tSNE_2"],
    s=20,
    c=red,
    alpha=0.9,
    linewidths=0,
    rasterized=True
)

ax.set_axis_off()

plt.subplots_adjust(
    left=0,
    right=1,
    bottom=0,
    top=1
)

plt.savefig(
    output_png,
    dpi=600,
    transparent=True,
    bbox_inches="tight",
    pad_inches=0
)

plt.close()

print("\n=========================")
print(f"PFAS Number: {len(df):,}")
print(f"PFAS PFC = 0，Blue: {mask_blue.sum():,}")
print(f"PFAS PFC = 1，Red: {mask_red.sum():,}")
print(f"PNG: {output_png}")
print(f"CSV: {output_csv}")
