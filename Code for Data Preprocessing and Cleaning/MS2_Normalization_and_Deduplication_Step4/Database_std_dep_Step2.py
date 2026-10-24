import os
import ast
import pandas as pd

input_dir = r"Database_filtered_selected"
output_dir = r"MS2_grouped"
batch_size = 500

os.makedirs(output_dir, exist_ok=True)

all_data = []
databases = ["GNPS","NIST20","NIST23","MONA"]

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if not file.endswith(".csv"):
            continue
        csv_file = os.path.join(root, file)
        try:
            df = pd.read_csv(csv_file, low_memory=False)

            if "Canonical_SMILES" not in df.columns or "Mass_spectral" not in df.columns or "Ion_mode" not in df.columns:
                continue

            database = next((db for db in databases if db in root), None)

            temp = df.dropna(subset=["Canonical_SMILES","Mass_spectral"]).copy()
            temp["Database"] = database

            all_data.append(temp)

            print("Read:", csv_file, "Database:", database, "Mode:", temp["Ion_mode"].unique(), "Count:", len(temp))

        except Exception as e:
            print("Erroe:", csv_file, e)

data = pd.concat(all_data, ignore_index=True)
print("Spectra number:", len(data))

for ion_mode, ion_df in data.groupby("Ion_mode"):
    mode_dir = os.path.join(output_dir, str(ion_mode))
    os.makedirs(mode_dir, exist_ok=True)

    groups = list(ion_df.groupby("Canonical_SMILES"))
    print(ion_mode, "Unique compound number:", len(groups))

    for part, start in enumerate(range(0, len(groups), batch_size), 1):
        batch_groups = groups[start:start+batch_size]
        batch_df = pd.concat([g for _, g in batch_groups], ignore_index=True)

        output_file = os.path.join(mode_dir, f"SMILES_group_{part}.csv")
        batch_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print("Save:", output_file, "Compound:", len(batch_groups), "Spectra:", len(batch_df))

print("All finish")