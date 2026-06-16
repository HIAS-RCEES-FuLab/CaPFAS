import pandas as pd
import os
import ast

# data dir and output dir
data_dir = r"D:\MSMS\MSMS_pfas_judge"
output_dir = r"D:\MSMS\MSMS_H+H-"
os.makedirs(output_dir, exist_ok=True)

# CSV file list
files = {
    "NIST": ["nist_PFAS_NEG.csv", "nist_PFAS_POS.csv",
             "nist_nonPFAS_NEG.csv", "nist_nonPFAS_POS.csv"],
    "MoNA": ["mona_PFAS_NEG.csv", "mona_PFAS_POS.csv",
             "mona_nonPFAS_NEG.csv", "mona_nonPFAS_POS.csv"],
    "GNPS": ["gnps_PFAS_NEG.csv", "gnps_PFAS_POS.csv",
             "gnps_nonPFAS_NEG.csv", "gnps_nonPFAS_POS.csv"],
    "Standard": ["standard_PFAS_NEG.csv"]
}

# parameter setting
mode_map = {"[M-H]-": "NEG", "[M+H]+": "POS"}
columns_to_keep = ["Name", "Precursor_type", "Precursor_mz", "Ion_mode",
                   "SMILES", "Formula", "Num_peaks", "Mass_spectral"]
H_mass = 1.007276

# Decimal check
def has_more_than_2_decimal(ms_string):
    try:
        ms_list = ast.literal_eval(ms_string)
        for mz, intensity in ms_list:
            decimal_part = str(mz).split(".")[1] if "." in str(mz) else "0"
            if len(decimal_part) > 2:
                return True
        return False
    except:
        return True

for source, file_list in files.items():
    source_dir = os.path.join(output_dir, source)
    os.makedirs(source_dir, exist_ok=True)

    for fname in file_list:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            continue

        df = pd.read_csv(fpath)
        if "Precursor_type" not in df.columns:
            continue
        df["Precursor_type"] = df["Precursor_type"].fillna("Unknown")
        df = df[df["Precursor_type"].isin(mode_map.keys())].copy()
        if df.empty:
            continue

        for precursor, mode in mode_map.items():
            df_filtered = df[df["Precursor_type"] == precursor].copy()
            total_rows = len(df_filtered)
            if total_rows == 0:
                continue

            df_filtered = df_filtered[[col for col in columns_to_keep if col in df_filtered.columns]].copy()

            if "Mass_spectral" in df_filtered.columns:
                mask = df_filtered["Mass_spectral"].apply(has_more_than_2_decimal)
                df_final = df_filtered[mask].copy()
                discarded_due_to_precision = total_rows - len(df_final)
            else:
                df_final = df_filtered.copy()
                discarded_due_to_precision = 0

            if "Precursor_mz" in df_final.columns:
                df_final.loc[:, "Precursor_mz"] = pd.to_numeric(df_final["Precursor_mz"], errors="coerce")

                def calc_exact_mass(row):
                    mz = row["Precursor_mz"]
                    if pd.isna(mz):
                        return None
                    if row["Precursor_type"] == "[M+H]+":
                        return mz - H_mass
                    elif row["Precursor_type"] == "[M-H]-":
                        return mz + H_mass
                    else:
                        return mz

                df_final.loc[:, "Exact_mass"] = df_final.apply(calc_exact_mass, axis=1)

            saved_rows = len(df_final)
            compound_count = df_final["Name"].nunique() if "Name" in df_final.columns else 0

            if not df_final.empty:
                out_fname = f"{os.path.splitext(fname)[0]}_{mode}.csv"
                out_path = os.path.join(source_dir, out_fname)
                df_final.to_csv(out_path, index=False)

            print(
                f"File '{fname}' | Precursor '{precursor}': "
                f"Original {total_rows} rows, Discarded (low precision) {discarded_due_to_precision} rows, "
                f"Saved {saved_rows} rows, Compounds {compound_count}"
            )

