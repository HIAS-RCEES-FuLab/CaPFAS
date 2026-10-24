import os
import ast
import pandas as pd
import numpy as np
from KMD import kendrick_mass_defect

input_dir = r"MS2_filtered_remove_standard"
output_dir = r"MS2_feature_processed_I_1"
databases = ["GNPS", "MONA", "NIST20", "NIST23"]
proton_mass = 1.007276

def process_spectrum(spectrum):
    if pd.isna(spectrum) or spectrum == "":
        return []
    try:
        peaks = ast.literal_eval(spectrum)
    except:
        return []
    binned = {}
    for mz, intensity in peaks:
        mz, intensity = float(mz), float(intensity)
        if mz < 50 or mz > 1000 or intensity <= 1:
            continue
        mz_bin = int(mz * 100)
        if mz_bin not in binned or intensity > binned[mz_bin]:
            binned[mz_bin] = intensity
    return [(mz_bin, round(intensity, 4)) for mz_bin, intensity in sorted(binned.items())]

def process_neutral_loss(precursor_mz, spectrum):
    if pd.isna(precursor_mz) or pd.isna(spectrum) or spectrum == "":
        return []
    try:
        peaks = ast.literal_eval(spectrum) if isinstance(spectrum, str) else spectrum
    except:
        return []
    losses = set()
    for mz, intensity in peaks:
        mz = float(mz)
        if intensity <= 1 or mz >= precursor_mz:
            continue
        loss = precursor_mz - mz
        if loss < 12:
            continue
        loss_bin = int(loss * 100)
        losses.add(loss_bin)
    return sorted(losses)

def process_mass_difference(precursor_mz, spectrum, top_n=10):
    if pd.isna(precursor_mz) or pd.isna(spectrum) or spectrum == "":
        return []
    try:
        peaks = ast.literal_eval(spectrum) if isinstance(spectrum, str) else spectrum
    except:
        return []
    peaks = [x for x in peaks if float(x[0]) < precursor_mz and float(x[1]) > 1]
    if len(peaks) < 2:
        return []
    peaks = sorted(peaks,key=lambda x: float(x[1]),reverse=True)[:top_n]
    mz_list = sorted([float(x[0]) for x in peaks])
    differences = set()
    for i in range(len(mz_list)):
        for j in range(i + 1, len(mz_list)):
            diff = mz_list[j] - mz_list[i]
            if diff < 12:
                continue
            diff_bin = int(diff * 100)
            differences.add(diff_bin)
    return sorted(differences)

for ion_mode in ["N", "P"]:
    print("\nProcess:", ion_mode)
    for db in databases:
        input_path = os.path.join(input_dir, ion_mode, db)

        files = sorted([f for f in os.listdir(input_path) if f.endswith(".csv")])
        print("\nDatabase:", db)

        for file in files:
            df = pd.read_csv(os.path.join(input_path, file), low_memory=False)

            if ion_mode == "N":
                df["Feature_Exact_mass"] = (df["Precursor_mz"] + proton_mass).round(4)
            else:
                df["Feature_Exact_mass"] = (df["Precursor_mz"] - proton_mass).round(4)

            kmd_result = kendrick_mass_defect(df["Feature_Exact_mass"], unit_str="CF2")
            df["Feature_KMD"] = [round(x["KMD"], 4) for x in kmd_result]
            df["Feature_KM"] = [round(x["KM"], 4) for x in kmd_result]

            df["Feature_KMD_norm"] = (
                df["Feature_KMD"] / df["Feature_Exact_mass"] * 1000
            ).round(4)

            df["Feature_Mass_scale"] = np.ceil(
                df["Feature_Exact_mass"] / 200
            ).astype(int)

            df["Feature_Mass_spectral"] = df["Mass_spectral"].apply(process_spectrum)

            df["Feature_Neutral_loss"] = [
                process_neutral_loss(p, s)
                for p, s in zip(
                    df["Precursor_mz"],
                    df["Mass_spectral"]
                )
            ]

            df["Feature_Mass_difference"] = [
                process_mass_difference(p, s)
                for p, s in zip(
                    df["Precursor_mz"],
                    df["Mass_spectral"]
                )
            ]

            save_path = os.path.join(output_dir,ion_mode,db)
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(os.path.join(save_path, file),index=False)
            print("Save:", os.path.join(save_path, file))