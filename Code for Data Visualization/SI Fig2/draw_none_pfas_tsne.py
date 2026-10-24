import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

input_dir = r"non_PFAS_tSNE"
output_dir = r"non_PFAS_tSNE"
radius, fp_size = 2, 1024
pca_dim = 64

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
    arr = np.zeros(fp_size, dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr

for mode in ["N", "P"]:
    input_csv = os.path.join(input_dir, f"{mode}_non_PFAS_comparison.csv")
    output_csv = os.path.join(output_dir, f"{mode}_non_PFAS_tSNE.csv")
    output_png = os.path.join(output_dir, f"{mode}_non_PFAS_tSNE.png")

    df = pd.read_csv(input_csv).dropna(subset=["Canonical SMILES"]).copy()
    df["Canonical SMILES"] = df["Canonical SMILES"].astype(str)

    print("Morgan fingerprints...")
    fps = []
    valid_indices = []

    for i, smiles in enumerate(df["Canonical SMILES"]):
        fp = smiles_to_fp(smiles)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(i)
        if (i + 1) % 100000 == 0:
            print(f"  {i + 1:,}/{len(df):,}")

    df = df.iloc[valid_indices].copy()
    all_fps = np.vstack(fps).astype(np.float32)
    print(f"Fingerprint shape: {all_fps.shape}")

    # =========================
    # PCA: 1024D -> 64D
    # =========================
    print(f"PCA: {fp_size}D -> {pca_dim}D...")

    pca = PCA(n_components=pca_dim, random_state = 42)
    pca.fit(all_fps)

    pca_features = pca.transform(all_fps).astype(np.float32)

    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"PCA finished")

    del all_fps
    del fps

    # =========================
    # t-SNE: 64D -> 2D
    # =========================
    print("Begin t-SNE...")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="random",
        max_iter=1000,
        random_state=42,
        metric="euclidean"
    )

    embedding = tsne.fit_transform(pca_features)
    print("t-SNE finished")

    del pca_features

    df["tSNE_1"] = embedding[:, 0]
    df["tSNE_2"] = embedding[:, 1]

    df["Dataset"] = np.where(
        (df["root1"] == 1) & (df["root2"] == 1),
        "Both",
        np.where(df["root1"] == 1, "root1", "root2")
    )

    output_df = df[
        ["Canonical SMILES", "root1", "root2", "tSNE_1", "tSNE_2", "Dataset"]
    ]

    output_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    blue, red = "#6F9FC2", "#E57373"

    mask_root1 = df["root1"] == 1
    mask_root2 = df["root2"] == 1

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(
        df.loc[mask_root1, "tSNE_1"],
        df.loc[mask_root1, "tSNE_2"],
        s=15,
        c=blue,
        alpha=0.9,
        linewidths=0,
        rasterized=True
    )

    ax.scatter(
        df.loc[mask_root2, "tSNE_1"],
        df.loc[mask_root2, "tSNE_2"],
        s=20,
        c=red,
        alpha=0.9,
        linewidths=0,
        rasterized=True
    )

    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.savefig(
        output_png,
        dpi=600,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()

    both = ((df["root1"] == 1) & (df["root2"] == 1)).sum()
    only_root1 = ((df["root1"] == 1) & (df["root2"] == 0)).sum()
    only_root2 = ((df["root1"] == 0) & (df["root2"] == 1)).sum()