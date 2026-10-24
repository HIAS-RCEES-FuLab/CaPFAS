import os
import ast
import pandas as pd

# =========================
# 配置
# =========================
shap_root = r"SHAP/SHAP_result"

data_files = {
    "N": r"SHAP/N_3000.csv",
    "P": r"SHAP/P_3000.csv"
}

shap_files = {
    "N_MS2": os.path.join(shap_root, "N", "PFAS_SHAP_N_MS2_fragment_sequence_all_nonzero.csv"),
    "N_NL": os.path.join(shap_root, "N", "PFAS_SHAP_N_mass_difference_neutral_loss_all_nonzero.csv"),
    "P_MS2": os.path.join(shap_root, "P", "PFAS_SHAP_P_MS2_fragment_sequence_all_nonzero.csv"),
    "P_NL": os.path.join(shap_root, "P", "PFAS_SHAP_P_mass_difference_neutral_loss_all_nonzero.csv")
}

def check_bin(feature_string, target_bin):
    if pd.isna(feature_string) or str(feature_string).strip() == "":
        return False
    try:
        feats = ast.literal_eval(feature_string)
    except Exception:
        return False
    try:
        target_bin = int(target_bin)
    except:
        return False
    if not isinstance(feats, list):
        return False
    for item in feats:
        if isinstance(item, (list, tuple)):
            try:
                if int(item[0]) == target_bin:
                    return True
            except:
                continue
        else:
            try:
                if int(item) == target_bin:
                    return True
            except:
                continue
    return False

for name, shap_file in shap_files.items():

    print("\n" + "=" * 70)
    print(f"Begin process: {name}")

    shap_df = pd.read_csv(shap_file)

    # P/N mode
    mode = name[0]
    data_file = data_files[mode]
    print(f"Read data: {data_file}")
    df = pd.read_csv(
        data_file,
        low_memory=False
    )
    print(f"Sample count: {len(df)}")

    detected_counts = []

    # =========================
    # MS2 fragment
    # =========================
    if "MS2" in name:
        print("Feature_Mass_spectral")
        feature_series = df["Feature_Mass_spectral"]
        for bin_idx in shap_df["Bin"]:
            count = 0
            for value in feature_series:
                if check_bin(value, bin_idx):
                    count += 1
            detected_counts.append(count)

    # =========================
    # Mass difference + Neutral loss
    # =========================
    else:
        print("Feature_Mass_difference + Feature_Neutral_loss")
        for bin_idx in shap_df["Bin"]:
            count = 0
            for _, row in df.iterrows():
                mass_diff_exist = check_bin(
                    row["Feature_Mass_difference"],
                    bin_idx
                )
                nl_exist = check_bin(
                    row["Feature_Neutral_loss"],
                    bin_idx
                )
                if mass_diff_exist or nl_exist:
                    count += 1
            detected_counts.append(count)
    shap_df["Detected_samples"] = detected_counts
    output_file = shap_file.replace(
        ".csv",
        "_with_count.csv"
    )

    shap_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Finish: {output_file}")
    print(shap_df["Detected_samples"].describe())