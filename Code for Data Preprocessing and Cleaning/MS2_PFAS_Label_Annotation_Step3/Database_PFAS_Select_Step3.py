import os
import pandas as pd

input_dir = r"Database_filtered_selected"
input_file = r"PFAS_structure_annotation_scaffold.csv"

target_cols = [
    "F>=2",
    "CF2",
    "Perfluoroalkyl_chain",
    "Scaffold"
]
pfas_df = pd.read_csv(input_file, low_memory=False)
pfas_dict = pfas_df.set_index("Canonical_SMILES")[target_cols].to_dict("index")

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if not file.endswith(".csv"):
            continue
        csv_file = os.path.join(root, file)
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            for col in target_cols:
                if col not in df.columns:
                    df[col] = None
            match_count = 0

            # Canonical SMILES matching
            for idx, smi in df["Canonical_SMILES"].items():
                if pd.isna(smi):
                    continue
                if smi in pfas_dict:
                    match_count += 1
                    for col in target_cols:
                        df.at[idx, col] = pfas_dict[smi][col]
            # save file
            df.to_csv(
                csv_file,
                index=False,
                encoding="utf-8-sig"
            )

        except Exception as e:
            print("Error:", csv_file, e)

print("All finish")