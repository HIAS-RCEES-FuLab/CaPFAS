import os
import pandas as pd

input_dir = r"MS2_feature_processed_I_1"
output_dir = r"MS2_feature_processed_I_1_down_sampling"
batch_size = 500

os.makedirs(output_dir, exist_ok=True)

for ion_mode in ["N", "P"]:
    input_path = os.path.join(input_dir, ion_mode)
    if not os.path.exists(input_path):
        continue

    print("\n====================")
    print("Process:", ion_mode)

    all_data = []

    for db in os.listdir(input_path):
        db_path = os.path.join(input_path, db)
        if not os.path.isdir(db_path):
            continue

        for file in os.listdir(db_path):
            if not file.endswith(".csv"):
                continue

            csv_file = os.path.join(db_path, file)

            try:
                df = pd.read_csv(csv_file, low_memory=False)

                if not all(x in df.columns for x in ["Canonical_SMILES", "Mass_spectral"]):
                    continue

                temp = df.dropna(subset=["Canonical_SMILES", "Mass_spectral"]).copy()
                temp["Database"] = db
                all_data.append(temp)

                print("read:", csv_file, "count:", len(temp))

            except Exception as e:
                print("error:", csv_file, e)

    if len(all_data) == 0:
        continue

    data = pd.concat(all_data, ignore_index=True)
    print("Spectra count:", len(data))

    groups = list(data.groupby("Canonical_SMILES"))
    print("Unique compound:", len(groups))

    save_dir = os.path.join(output_dir, ion_mode)
    os.makedirs(save_dir, exist_ok=True)

    for part, start in enumerate(range(0, len(groups), batch_size), 1):
        batch_groups = groups[start:start + batch_size]

        batch_df = pd.concat(
            [group_df for _, group_df in batch_groups],
            ignore_index=True
        )

        output_file = os.path.join(save_dir, f"SMILES_group_{part}.csv")

        batch_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            "Save:",
            output_file,
            "| compound:",
            len(batch_groups),
            "| spectra:",
            len(batch_df)
        )

print("\nAll finish")