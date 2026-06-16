import pandas as pd
import os
import re
import ast

# parameter setting
base_dir = r"D:\MSMS\MSMS_H+H-\Train_data"
output_file = os.path.join(base_dir, "NEG_all_neutral_loss.csv")

files = [f for f in os.listdir(base_dir) if "NEG" in f and f.endswith(".csv")]

# read files
df_list = []
for f in files:
    path = os.path.join(base_dir, f)
    df = pd.read_csv(path)
    df = df.replace({r"\n": "", r"\r": ""}, regex=True)
    df_sub = df[["Precursor_mz", "Mass_spectral"]].copy()
    df_sub["Source"] = f
    df_list.append(df_sub)
print(df_list)
df_all = pd.concat(df_list, ignore_index=True)
print(f"Total samples: {len(df_all)}")

# Parse mass spectral data
def parse_spectrum(spec_str):
    if pd.isna(spec_str) or spec_str == "":
        return [], []

    try:
        spec = ast.literal_eval(spec_str)

        mz_list = [float(x[0]) for x in spec]
        int_list = [float(x[1]) for x in spec]

        return mz_list, int_list

    except:
        return [], []
df_all["MS2_mz_list"], df_all["MS2_int_list"] = zip(*df_all["Mass_spectral"].apply(parse_spectrum))

# -----------------------------
# compute Neutral Loss
# -----------------------------
def calc_neutral_loss(row):
    precursor = row["Precursor_mz"]
    mz_list = row["MS2_mz_list"]
    int_list = row["MS2_int_list"]
    if not mz_list or not int_list:
        return pd.Series(["", ""])
    # Normalize intensities
    max_int = max(int_list)
    int_norm = [i / max_int * 100 for i in int_list]

    # filter
    filtered_mz = [
        round(mz, 5)
        for mz, inten in zip(mz_list, int_norm)
        if mz <= precursor - 0.005 and inten > 5
    ]

    neutral_loss = [round(precursor - mz, 5) for mz in filtered_mz]

    return pd.Series([
        ";".join(map(str, filtered_mz)),
        ";".join(map(str, neutral_loss))
    ])

df_all[["MS2_mz_filtered", "Neutral_Loss"]] = df_all.apply(calc_neutral_loss, axis=1)

df_all["Precursor_mz"] = df_all["Precursor_mz"].round(5)
df_all.to_csv(output_file, index=False, encoding="utf-8-sig")