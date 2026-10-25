import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==================== Parameter ====================
pfas_file = r"data/PFAS_structure_annotation_scaffold_CF2.csv"
standard_file = r"data/Standard_unique.csv"
# Contains nearly 9 million chemical structures; the dataset is too large to be publicly provided.
morgan_folder = r"data/OECD_PFAS/MorganFP_1024_R2"
output_dir = r"data/PFAS_tSNE_repeated_500000"
os.makedirs(output_dir, exist_ok=True)

morgan_sample_size = 500000
random_seed = 42
radius = 2
fp_size = 1024
pca_dim = 64
perplexity = 30
max_iter = 1000

pfas_color = "#FAADA5"
morgan_color = "#A1C6DF"

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    arr = np.zeros(fp_size, dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr

# ==================== PFAS ====================
print("=" * 60)
print("Read PFAS")
print("=" * 60)

df = pd.read_csv(pfas_file, low_memory=False)
required_columns = ["Canonical_SMILES", "Perfluoroalkyl_chain", "Scaffold"]

df["Canonical_SMILES"] = df["Canonical_SMILES"].fillna("").astype(str).str.strip()
df = df[df["Canonical_SMILES"] != ""].copy()
df.reset_index(drop=True, inplace=True)

print("Generate PFAS Morgan fingerprints...")

pfas_fps = []
for i, smiles in enumerate(df["Canonical_SMILES"]):
    pfas_fps.append(smiles_to_fp(smiles))
    if (i + 1) % 100000 == 0:
        print(f"  PFAS: {i + 1:,}/{len(df):,}")
pfas_valid = np.array([fp is not None for fp in pfas_fps])

df = df.iloc[np.where(pfas_valid)[0]].copy()
df.reset_index(drop=True, inplace=True)
pfas_fps = [fp for fp in pfas_fps if fp is not None]

pfas_fps = np.vstack(pfas_fps)

# ==================== Standard ====================
print("\n" + "=" * 60)
print("Read Standard")
print("=" * 60)

standard_df = pd.read_csv(standard_file, low_memory=False)

standard_df["Canonical SMILES"] = standard_df["Canonical SMILES"].fillna("").astype(str).str.strip()
standard_df = standard_df[standard_df["Canonical SMILES"] != ""].copy()
standard_df.reset_index(drop=True, inplace=True)

print("Generate Standard Morgan fingerprints...")

standard_fps = []
for i, smiles in enumerate(standard_df["Canonical SMILES"]):
    standard_fps.append(smiles_to_fp(smiles))
    if (i + 1) % 100000 == 0:
        print(f"  Standard: {i + 1:,}/{len(standard_df):,}")

standard_valid = np.array([fp is not None for fp in standard_fps])

standard_df = standard_df.iloc[np.where(standard_valid)[0]].copy()
standard_df.reset_index(drop=True, inplace=True)
standard_fps = [fp for fp in standard_fps if fp is not None]

standard_fps = np.vstack(standard_fps)

# ==================== MorganFP ====================
print("\n" + "=" * 60)
print("Read MorganFP")
print("=" * 60)

npy_files = sorted(glob.glob(os.path.join(morgan_folder, "*.npy")))

all_morgan_fps = []
all_morgan_df = []

for file_index, morgan_npy in enumerate(npy_files, 1):
    morgan_basename = os.path.splitext(os.path.basename(morgan_npy))[0]
    metadata_file = os.path.join(morgan_folder, morgan_basename.replace("_MorganFP_1024_R2", "_metadata") + ".csv")

    print("\n" + "-" * 60)
    print(f"Process file {file_index}/{len(npy_files)}")
    print(f"NPY: {os.path.basename(morgan_npy)}")

    morgan_fps_part = np.load(morgan_npy)
    morgan_df_part = pd.read_csv(metadata_file, low_memory=False)

    valid_morgan = morgan_df_part["Valid_SMILES"].astype(str).str.strip().str.lower().eq("true").values
    morgan_fps_part = morgan_fps_part[valid_morgan]
    morgan_df_part = morgan_df_part.loc[valid_morgan].copy()

    all_morgan_fps.append(morgan_fps_part)
    all_morgan_df.append(morgan_df_part.reset_index(drop=True))

morgan_fps = np.vstack(all_morgan_fps)
morgan_df = pd.concat(all_morgan_df, ignore_index=True)

# ==================== Count ====================
pfas_count = len(pfas_fps)
standard_count = len(standard_fps)
morgan_count = len(morgan_fps)

n_batches = int(np.ceil(morgan_count / morgan_sample_size))
last_batch_size = morgan_count - (n_batches - 1) * morgan_sample_size

print("\n" + "=" * 60)
print(f"PFAS: {pfas_count:,}")
print(f"Standard: {standard_count:,}")
print(f"MorganFP: {morgan_count:,}")

# ==================== Random ====================
print("\n" + "=" * 60)

rng = np.random.default_rng(random_seed)
shuffled_indices = rng.permutation(morgan_count)
available_indices = shuffled_indices

# ====================  PCA + t-SNE ====================
print("\n" + "=" * 60)
print("Begin PCA + t-SNE")
print("=" * 60)

used_morgan_indices = []

for batch in range(1, n_batches + 1):
    print("\n" + "#" * 60)
    print(f" {batch}/{n_batches} batch")
    print("#" * 60)

    start = (batch - 1) * morgan_sample_size
    end = min(start + morgan_sample_size, len(available_indices))
    morgan_sample_indices = np.asarray(available_indices[start:end], dtype=np.int64)
    morgan_sample_indices.sort()

    sampled_morgan_fps = morgan_fps[morgan_sample_indices]
    sampled_morgan_df = morgan_df.iloc[morgan_sample_indices].reset_index(drop=True)
    used_morgan_indices.extend(morgan_sample_indices.tolist())

    current_morgan_count = len(morgan_sample_indices)
    total_count = pfas_count + standard_count + current_morgan_count

    print(f"PFAS: {pfas_count:,}")
    print(f"Standard: {standard_count:,}")
    print(f"MorganFP: {current_morgan_count:,}")
    print(f"MorganFP Index: {morgan_sample_indices.min():,} - {morgan_sample_indices.max():,}")

    sampled_fps = np.vstack([pfas_fps, standard_fps, sampled_morgan_fps])
    sampled_labels = np.concatenate([
        np.repeat("PFAS", pfas_count),
        np.repeat("Standard", standard_count),
        np.repeat("MorganFP", current_morgan_count)
    ])

    # ==================== PCA ====================
    print("\nPCA...")
    pca = PCA(n_components=pca_dim, random_state=random_seed)
    sampled_pca = pca.fit_transform(sampled_fps.astype(np.float32))
    cumulative_variance = pca.explained_variance_ratio_.sum()
    print(f"PCA {pca_dim} cumulative variance: {cumulative_variance * 100:.2f}%")

    # ==================== t-SNE ====================
    print("\nt-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=max_iter,
        random_state=batch,
        metric="euclidean"
    )
    embedding = tsne.fit_transform(sampled_pca)
    print("t-SNE finished")

    # ==================== Output CSV ====================
    morgan_scaffold = sampled_morgan_df["Scaffold"].tolist() if "Scaffold" in sampled_morgan_df.columns else [np.nan] * current_morgan_count

    result_df = pd.DataFrame({
        "Canonical_SMILES": (
            df["Canonical_SMILES"].tolist()
            + standard_df["Canonical SMILES"].tolist()
            + sampled_morgan_df["SMILES"].tolist()
        ),
        "Perfluoroalkyl_chain": (
            df["Perfluoroalkyl_chain"].tolist()
            + [np.nan] * standard_count
            + [np.nan] * current_morgan_count
        ),
        "Scaffold": (
            df["Scaffold"].tolist()
            + [np.nan] * standard_count
            + morgan_scaffold
        ),
        "tSNE_1": embedding[:, 0],
        "tSNE_2": embedding[:, 1],
        "Dataset": sampled_labels,
        "Original_Index": np.concatenate([
            np.arange(pfas_count),
            np.arange(standard_count),
            morgan_sample_indices
        ])
    })

    result_df = result_df[[
        "Canonical_SMILES",
        "Perfluoroalkyl_chain",
        "Scaffold",
        "tSNE_1",
        "tSNE_2",
        "Dataset",
        "Original_Index"
    ]]

    csv_file = os.path.join(output_dir, f"tSNE_batch_{batch:02d}.csv")
    result_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"CSV: {csv_file}")
    print("Plotting...")

    fig, ax = plt.subplots(figsize=(5, 5))

    mask = sampled_labels == "MorganFP"
    ax.scatter(embedding[mask, 0], embedding[mask, 1], s=10, c=morgan_color, alpha=0.8, linewidths=0, rasterized=True)

    mask = sampled_labels == "Standard"
    ax.scatter(embedding[mask, 0], embedding[mask, 1], s=15, c=pfas_color, alpha=0.8, linewidths=0, rasterized=True)

    mask = sampled_labels == "PFAS"
    ax.scatter(embedding[mask, 0], embedding[mask, 1], s=10, c=pfas_color, alpha=0.8, linewidths=0, rasterized=True)

    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    png_file = os.path.join(output_dir, f"tSNE_batch_{batch:02d}.png")
    plt.savefig(png_file, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"PNG: {png_file}")
