import os
import re
import pandas as pd
from rdkit import Chem

input_dir = r"Database_filtered_selected"
output_file = r"PFAS_structure_annotation.csv"

cf2_pattern = Chem.MolFromSmarts(
    "[C;H0;$(C(F)(F));!$(C(F)(F)[Cl,Br,I])]"
)

smiles_info = {}
def count_F(formula):
    if pd.isna(formula):
        return 0
    m = re.search(r"F(\d*)", str(formula))
    return int(m.group(1)) if m and m.group(1) else (1 if m else 0)

for db in ["GNPS", "MONA", "NIST20", "NIST23"]:
    for mode in ["N", "P"]:
        print(db,mode)
        folder = os.path.join(input_dir, db, mode)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if not file.endswith(".csv"):
                continue
            path = os.path.join(folder, file)
            try:
                df = pd.read_csv(
                    path,
                    dtype=str,
                    low_memory=False
                )
                if "Canonical_SMILES" not in df.columns or "Formula" not in df.columns:
                    continue

                for _, row in df.iterrows():
                    smi = str(row["Canonical_SMILES"]).strip()
                    if not smi or smi == "nan":
                        continue

                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        continue

                    if smi not in smiles_info:
                        smiles_info[smi] = {
                            "F>=2": 0,
                            "CF2": 0,
                            "Mode": set(),
                            "Database": set()
                        }

                    smiles_info[smi]["Mode"].add(mode)

                    smiles_info[smi]["Database"].add(db)

                    if count_F(row["Formula"]) >= 2:
                        smiles_info[smi]["F>=2"] = 1

                    if mol.HasSubstructMatch(cf2_pattern):
                        smiles_info[smi]["CF2"] = 1

            except Exception as e:
                print(f"{path} read fail: {e}")

result = []

for smi, info in smiles_info.items():
    result.append({
        "Canonical_SMILES": smi,
        "F>=2": info["F>=2"],
        "CF2": info["CF2"],
        "Mode": ";".join(sorted(info["Mode"])),
        "Database": ";".join(sorted(info["Database"]))
    })

result_df = pd.DataFrame(result)

# save
result_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


print("==========完成==========")
print(f"唯一结构数量: {len(result_df)}")
print(f"保存位置: {output_file}")
print(result_df.head())