import os
import pandas as pd

dataset_name = "MS2_dataset_split_SMILES_type5_12_to_1"
base_dir = dataset_name

output_dir = r"SHAP"
os.makedirs(output_dir, exist_ok=True)

ion_modes = ["P", "N"]
splits = ["val", "test"]
sample_size = 3000
random_seed = 42

def get_files(split, mode):
    files = []
    split_dir = os.path.join(base_dir, split, mode)
    if not os.path.exists(split_dir):
        return files
    for root, _, fs in os.walk(split_dir):
        for f in fs:
            if f.lower().endswith(".csv"):
                files.append(os.path.join(root, f))
    return files

for mode in ion_modes:
    print("\n" + "=" * 60)
    print(f"Process {mode} mode")
    print("=" * 60)

    all_data = []

    for split in splits:
        files = get_files(split, mode)
        print(f"{split} file number: {len(files)}")

        for file in files:
            try:
                df = pd.read_csv(file, low_memory=False)
            except Exception as e:
                print(f"read fail: {file}")
                print(e)
                continue

            if "CF2" not in df.columns or "Canonical_SMILES" not in df.columns:
                continue

            df = df[(df["CF2"] == 1) & df["Canonical_SMILES"].notna()].copy()
            df = df[df["Canonical_SMILES"].astype(str).str.strip() != ""]

            if len(df) > 0:
                all_data.append(df)

    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"{mode} PFAS count: {len(merged_df)}")

    if len(merged_df) > sample_size:
        sampled_df = merged_df.sample(sample_size, random_state=random_seed)
    else:
        sampled_df = merged_df.copy()

    output_file = os.path.join(output_dir, f"{mode}_3000.csv")
    sampled_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"Save number: {len(sampled_df)}")
    print(output_file)
