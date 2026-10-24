import os
import pandas as pd

dirs = {
    "NIST23": r"NIST23_processed",
    "NIST20": r"NIST20"
}

database_smiles = {}
database_df = {}

for name, folder in dirs.items():
    smiles_set = set()
    df_list = []
    total_rows = 0

    print(f"\nProcess database: {name}")

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)

            try:
                df = pd.read_csv(file_path)

                df_list.append(df)

                smiles = df["Canonical_SMILES"].dropna()
                smiles = smiles[smiles.astype(str).str.strip() != ""]

                smiles_set.update(smiles.astype(str))
                total_rows += len(smiles)

            except Exception as e:
                print(f"{file} read fail: {e}")

    database_smiles[name] = smiles_set
    database_df[name] = pd.concat(df_list, ignore_index=True)

    print(f"Record number: {total_rows}")
    print(f"Unique compound: {len(smiles_set)}")


# ===========================
# NIST23 vs NIST20 compare
# ===========================

nist23 = database_smiles["NIST23"]
nist20 = database_smiles["NIST20"]

common = nist23 & nist20
nist23_only = nist23 - nist20
nist20_only = nist20 - nist23

print("\n========== result compare ==========")
print(f"NIST23 unique compound: {len(nist23)}")
print(f"NIST20 unique compound: {len(nist20)}")
print(f"Shared compound: {len(common)}")
print(f"NIST23 new comound: {len(nist23_only)}")
print(f"NIST23 missing: {len(nist20_only)}")

nist20_df = database_df["NIST20"]

missing_df = nist20_df[
    nist20_df["Canonical_SMILES"].isin(nist20_only)
]

output_dir = r"NIST20_not_in_NIST23"
os.makedirs(output_dir, exist_ok=True)

chunk_size = 100000

for i in range(0, len(missing_df), chunk_size):
    chunk = missing_df.iloc[i:i + chunk_size]

    file_num = i // chunk_size + 1
    output = os.path.join(
        output_dir,
        f"NIST20_not_in_NIST23_part{file_num}.csv"
    )

    chunk.to_csv(output, index=False)
    print(f"Save: {output} ({len(chunk)} 行)")

print("\nSave finish")
print(f"Total: {len(missing_df)}")
print(f"Compound: {missing_df['Canonical_SMILES'].nunique()}")
print(f"File: {(len(missing_df) - 1) // chunk_size + 1}")