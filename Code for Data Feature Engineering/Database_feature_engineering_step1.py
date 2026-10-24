import os
import pandas as pd

standard_file = r"Standard_unique.csv"
input_dir = r"MS2_filtered_rebuild"
output_dir = r"MS2_filtered_remove_standard"

standard_df = pd.read_csv(
    standard_file,
    low_memory=False
)

standard_smiles = set(
    standard_df["Canonical_SMILES"]
    .dropna()
    .astype(str)
)

matched_smiles = set()

print("Standard Canonical SMILES number:", len(standard_smiles))

for ion_mode in ["N", "P"]:
    for db in ["GNPS", "MONA", "NIST20", "NIST23"]:

        input_path = os.path.join(
            input_dir,
            ion_mode,
            db
        )

        if not os.path.exists(input_path):
            continue

        output_path = os.path.join(
            output_dir,
            ion_mode,
            db
        )

        os.makedirs(
            output_path,
            exist_ok=True
        )

        files = sorted([
            f for f in os.listdir(input_path)
            if f.endswith(".csv")
        ])

        print("\nProcess:", ion_mode, db)

        out_index = 1
        total_before = 0
        total_after = 0
        total_remove = 0

        for file in files:

            df = pd.read_csv(
                os.path.join(input_path, file),
                low_memory=False
            )

            before = len(df)

            hit = set(
                df.loc[
                    df["Canonical_SMILES"].astype(str).isin(standard_smiles),
                    "Canonical_SMILES"
                ]
            )

            matched_smiles.update(hit)
            df = df[
                ~df["Canonical_SMILES"]
                .astype(str)
                .isin(standard_smiles)
            ]

            after = len(df)
            total_before += before
            total_after += after
            total_remove += before - after

            if after > 0:
                df.to_csv(
                    os.path.join(
                        output_path,
                        f"{out_index}.csv"
                    ),
                    index=False
                )
                out_index += 1

        print("Raw:", total_before)
        print("Delete:", total_remove)
        print("Remain:", total_after)

