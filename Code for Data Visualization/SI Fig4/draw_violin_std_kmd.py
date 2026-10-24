import os
import pandas as pd
import matplotlib.pyplot as plt
# Directory containing the model development datasets.
# The data are not publicly available because they include commercially sourced data.
ms2_dir = r"MS2 DIR"
output = r"KMD_norm_compound_violin.png"

compound_data = {}

for split in ["train", "val", "test"]:
    split_dir = os.path.join(ms2_dir, split)
    if not os.path.exists(split_dir):
        continue
    print("Processing:", split)

    for mode in ["P", "N"]:
        mode_dir = os.path.join(split_dir, mode)
        if not os.path.exists(mode_dir):
            continue

        for root, dirs, files in os.walk(mode_dir):
            for file in files:
                if not file.endswith(".csv"):
                    continue

                df = pd.read_csv(os.path.join(root, file))

                if not all(c in df.columns for c in ["Feature_KMD_norm", "CF2", "Canonical SMILES"]):
                    continue

                temp = df[["Feature_KMD_norm", "CF2", "Canonical SMILES"]].copy()
                temp["Feature_KMD_norm"] = pd.to_numeric(temp["Feature_KMD_norm"], errors="coerce")
                temp["CF2"] = pd.to_numeric(temp["CF2"], errors="coerce")
                temp = temp.dropna()

                for _, row in temp.iterrows():
                    smiles = str(row["Canonical SMILES"]).strip()

                    if smiles not in compound_data:
                        compound_data[smiles] = {
                            "Feature_KMD_norm": row["Feature_KMD_norm"],
                            "CF2": row["CF2"]
                        }
                    elif row["CF2"] == 1:
                        compound_data[smiles]["CF2"] = 1

compound_df = pd.DataFrame(compound_data.values())

print("Compound数量:", len(compound_df))
print("PFAS:", (compound_df["CF2"] == 1).sum())
print("non-PFAS:", (compound_df["CF2"] == 0).sum())

pfas_kmd = compound_df.loc[compound_df["CF2"] == 1, "Feature_KMD_norm"]
non_pfas_kmd = compound_df.loc[compound_df["CF2"] == 0, "Feature_KMD_norm"]

fig, ax = plt.subplots(figsize=(1.8, 2.5))

ax.violinplot(
    [pfas_kmd, non_pfas_kmd],
    positions=[1, 2],
    showmedians=True,
    showextrema=False
)
ax.set_ylim(-1, 1.5)
ax.set_xticks([1, 2])
ax.set_yticks(ax.get_yticks())
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_ylabel("")
ax.tick_params(axis="both", which="both", length=4, width=1)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(output, dpi=600, transparent=True, bbox_inches="tight")
plt.show()

print("save:", output)