import os
import pandas as pd
import ast

# path
csv_file = r"CFM.csv"
pre_dir = r"CFM_predicted_spectra_neg"

mz_tol = 0.005  # tolerence

# Parse CFM log
def parse_cfm_log(file_path):
    spectra = {
        "energy0": [],
        "energy1": [],
        "energy2": []
    }

    current_energy = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("energy"):
                current_energy = line
                continue

            if line.startswith("#") or line == "":
                continue

            if current_energy in spectra:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        mz = float(parts[0])
                        spectra[current_energy].append(mz)
                    except:
                        pass

    return spectra

def merge_mz_only(spectra, mz_tol=0.005):
    mz_all = []

    for energy in ["energy0", "energy1", "energy2"]:
        mz_all.extend(spectra[energy])

    mz_all.sort()

    merged = []
    for mz in mz_all:
        if not merged:
            merged.append(mz)
        else:
            if abs(mz - merged[-1]) > mz_tol:
                merged.append(mz)

    return merged

def match_spectra(exp_spectrum, pred_mz, precursor_mz, mz_tol=0.005):
    match_count = 0
    matched_intensity = 0.0
    matched_mz_list = []

    total_intensity = sum(i for _, i in exp_spectrum)

    for mz_exp, intensity in exp_spectrum:

        matched = False

        for mz_pred in pred_mz:
            if abs(mz_exp - mz_pred) <= mz_tol:
                matched = True
                break

        if not matched:
            if abs(mz_exp - precursor_mz) <= mz_tol:
                matched = True

        if matched:
            match_count += 1
            matched_intensity += intensity
            matched_mz_list.append(round(mz_exp, 4))

    matched_mz_list = sorted(set(matched_mz_list))

    ratio = round(matched_intensity / total_intensity, 3) if total_intensity > 0 else 0

    matched_mz_str = ";".join(map(str, matched_mz_list))

    return match_count, ratio, matched_mz_str

# main
df = pd.read_csv(csv_file)

merged_list = []
match_counts = []
match_ratios = []
matched_mz_all = []

for i in range(len(df)):
    log_file = os.path.join(pre_dir, f"{i}.log")

    if os.path.exists(log_file):
        spectra = parse_cfm_log(log_file)
        mz_merged = merge_mz_only(spectra, mz_tol=mz_tol)
    else:
        mz_merged = []

    merged_list.append(str(mz_merged))

    try:
        exp_spectrum = ast.literal_eval(df.loc[i, "Mass_spectral"])
    except:
        exp_spectrum = []

    precursor_mz = df.loc[i, "Precursor_mz"]

    exp_spectrum = [
        (mz, inten)
        for mz, inten in exp_spectrum
        if inten > 0.01 and mz <= precursor_mz + 0.005
    ]

    if exp_spectrum and mz_merged:
        count, ratio, matched_mz_str = match_spectra(
            exp_spectrum, mz_merged, precursor_mz, mz_tol=mz_tol
        )
    else:
        count, ratio, matched_mz_str = 0, 0, ""

    match_counts.append(count)
    match_ratios.append(ratio)
    matched_mz_all.append(matched_mz_str)

# write result
df["CFM_mz"] = merged_list
df["match_count"] = match_counts
df["match_ratio"] = match_ratios
df["matched_mz"] = matched_mz_all
df.to_csv(csv_file, index=False)
