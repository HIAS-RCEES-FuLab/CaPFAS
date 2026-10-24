import os
import pandas as pd

input_dir = r"MS2_feature_processed_I_1_down_sampling"
output_dir = r"MS2_feature_processed_I_1_down_sampling_type2_12_to_1"
sampling_ratio = 0.083333
random_state = 42

for ion_mode in ["N", "P"]:
    ion_input_dir = os.path.join(input_dir, ion_mode)
    ion_output_dir = os.path.join(output_dir, ion_mode)
    os.makedirs(ion_output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(ion_input_dir) if f.lower().endswith(".csv")]
    print(f"\n{'=' * 70}\nProcess {ion_mode} mode，count {len(csv_files)} CSV files\n{'=' * 70}")

    for file_name in csv_files:
        input_file = os.path.join(ion_input_dir, file_name)
        output_file = os.path.join(ion_output_dir, file_name)

        print(f"\nProcess file: {file_name}")

        df = pd.read_csv(input_file, low_memory=False)

        df_keep = df[df["CF2"] != 0].copy()
        df_cf20 = df[df["CF2"] == 0].dropna(subset=["Canonical_SMILES"]).copy()

        smiles_info = df_cf20[["Canonical_MILES", "Feature_Mass_scale"]].drop_duplicates("Canonical_SMILES").copy()

        sampled_smiles = []

        for mass_scale, group in smiles_info.groupby("Feature_Mass_scale", dropna=False):
            original_n = len(group)
            sample_n = max(1, int(round(original_n * sampling_ratio)))
            sampled = group.sample(n=sample_n, random_state=random_state)
            sampled_smiles.extend(sampled["Canonical_SMILES"].tolist())
            print(f"Feature_Mass_scale = {mass_scale}: {original_n} -> {sample_n}")

        df_cf20_sampled = df_cf20[df_cf20["Canonical_SMILES"].isin(sampled_smiles)].copy()
        df_result = pd.concat([df_cf20_sampled, df_keep]).sort_index()

        df_result.to_csv(output_file, index=False)

print("\nAll finish")