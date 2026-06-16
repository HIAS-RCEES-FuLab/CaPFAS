import os
import pandas as pd
import ast
import numpy as np

# input dir
input_dir = r"D:\MSMS\MSMS_H+H-\Train_data"

cf2_mass = 49.9968
ppm_tol = 5

# setting
electron_mass = 0.00054858
cf3_mz = 68.995209 + electron_mass
c2f5_mz = 118.992015 + electron_mass
c3f7_mz = 168.988821 + electron_mass
so3f_mz = 98.955219 + electron_mass
cf3o_mz = 84.990124 + electron_mass
c3f5o_mz = 146.98693 + electron_mass
so2f_mz = 82.960304 + electron_mass
c4f9_mz = 218.985628 + electron_mass

fragment_mz_list = [cf3_mz, c2f5_mz, c3f7_mz, so3f_mz, cf3o_mz, c3f5o_mz, so2f_mz, c4f9_mz]

def has_target_loss(loss_list, target_mass=cf2_mass, ppm=ppm_tol):
    try:
        losses = ast.literal_eval(loss_list)
        tol = target_mass * ppm * 1e-6
        return any(abs(l - target_mass) <= tol for l in losses)
    except:
        return False

def count_fragments(feature_list, fragment_mz_list, ppm=ppm_tol):
    """统计特征碎片出现次数"""
    count_dict = {mz: 0 for mz in fragment_mz_list}
    try:
        feats = ast.literal_eval(feature_list)
        for _, mz, _, _ in feats:
            for target_mz in fragment_mz_list:
                tol = target_mz * ppm * 1e-6
                if abs(mz - target_mz) <= tol:
                    count_dict[target_mz] += 1
    except:
        pass
    return count_dict

def stats(arr):
    if not arr:
        return {"mean": None, "min": None, "max": None,
                "q05": None, "q25": None, "q50": None,
                "q75": None, "q95": None}
    arr = np.array(arr)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q05": float(np.percentile(arr, 5)),
        "q25": float(np.percentile(arr, 25)),
        "q50": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75)),
        "q95": float(np.percentile(arr, 95)),
    }

# summary
summary = []
pfas_kmd_all = []
pfas_kmd_ratio_all = []
nonpfas_kmd_all = []
nonpfas_kmd_ratio_all = []

for fname in os.listdir(input_dir):
    if not fname.endswith(".csv") or "NEG" not in fname:
        continue
    fpath = os.path.join(input_dir, fname)
    df = pd.read_csv(fpath)

    max_msf_len = 0
    max_nl_len = 0
    nl_min = float('inf')
    nl_max = float('-inf')
    cf2_present_count = 0
    cf2_absent_count = 0

    # frag count
    frag_counts_total = {mz:0 for mz in fragment_mz_list}
    frag_any_count_total = 0
    cf2_or_frag_count_total = 0

    # KMD/Exact_mass
    kmd_list = []
    kmd_ratio_list = []

    n_samples = len(df)

    for idx in range(n_samples):
        frag_flag = False
        msf_str = df.loc[idx, "Mass_spectral_features"] if "Mass_spectral_features" in df.columns else None
        if pd.notna(msf_str):
            try:
                spectrum = ast.literal_eval(msf_str)
                if isinstance(spectrum, list):
                    max_msf_len = max(max_msf_len, len(spectrum))
                    frag_counts = count_fragments(msf_str, fragment_mz_list)
                    for mz in fragment_mz_list:
                        frag_counts_total[mz] += frag_counts[mz]
                    if any(v > 0 for v in frag_counts.values()):
                        frag_flag = True
                        frag_any_count_total += 1
            except:
                pass

        cf2_flag = False
        nl_str = df.loc[idx, "Neutral_losses"] if "Neutral_losses" in df.columns else None
        if pd.notna(nl_str):
            try:
                losses = ast.literal_eval(nl_str)
                if isinstance(losses, list):
                    max_nl_len = max(max_nl_len, len(losses))
                    if losses:
                        nl_min = min(nl_min, min(losses))
                        nl_max = max(nl_max, max(losses))
                    cf2_flag = has_target_loss(nl_str)
                    if cf2_flag:
                        cf2_present_count += 1
                    else:
                        cf2_absent_count += 1
            except:
                pass

        if cf2_flag or frag_flag:
            cf2_or_frag_count_total += 1

        # --- KMD / Exact_mass ---
        if "KMD" in df.columns and "Exact_mass" in df.columns:
            kmd_val = df.loc[idx, "KMD"]
            mass_val = df.loc[idx, "Exact_mass"]
            if pd.notna(kmd_val) and pd.notna(mass_val) and mass_val != 0:
                kmd_ratio_list.append(kmd_val / mass_val)
            if pd.notna(kmd_val):
                kmd_list.append(kmd_val)

    nl_min_val = nl_min if nl_min != float('inf') else None
    nl_max_val = nl_max if nl_max != float('-inf') else None

    kmd_stats = stats(kmd_list)
    kmd_ratio_stats = stats(kmd_ratio_list)

    if "nonPFAS" in fname:
        nonpfas_kmd_all.extend(kmd_list)
        nonpfas_kmd_ratio_all.extend(kmd_ratio_list)
    elif "PFAS" in fname:
        pfas_kmd_all.extend(kmd_list)
        pfas_kmd_ratio_all.extend(kmd_ratio_list)

    summary.append({
        "File": fname,
        "Max_Mass_spectral_features_len": max_msf_len,
        "Max_Neutral_losses_len": max_nl_len,
        "Neutral_losses_min_mz": nl_min_val,
        "Neutral_losses_max_mz": nl_max_val,
        "CF2_present_count": cf2_present_count,
        "CF2_absent_count": cf2_absent_count,
        **{f"{mz}_count": frag_counts_total[mz] for mz in fragment_mz_list},
        "Any_CF_fragment_count": frag_any_count_total,
        "CF2_or_fragment_count": cf2_or_frag_count_total,
        "Total_samples": n_samples,
        # KMD stats
        "KMD_mean": kmd_stats["mean"],
        "KMD_min": kmd_stats["min"],
        "KMD_max": kmd_stats["max"],
        "KMD_q05": kmd_stats["q05"],
        "KMD_q25": kmd_stats["q25"],
        "KMD_q50": kmd_stats["q50"],
        "KMD_q75": kmd_stats["q75"],
        "KMD_q95": kmd_stats["q95"],
        # KMD/Exact_mass stats
        "KMD_to_Exact_mass_mean": kmd_ratio_stats["mean"],
        "KMD_to_Exact_mass_min": kmd_ratio_stats["min"],
        "KMD_to_Exact_mass_max": kmd_ratio_stats["max"],
        "KMD_to_Exact_mass_q05": kmd_ratio_stats["q05"],
        "KMD_to_Exact_mass_q25": kmd_ratio_stats["q25"],
        "KMD_to_Exact_mass_q50": kmd_ratio_stats["q50"],
        "KMD_to_Exact_mass_q75": kmd_ratio_stats["q75"],
        "KMD_to_Exact_mass_q95": kmd_ratio_stats["q95"],
    })

df_summary = pd.DataFrame(summary)

df_summary = pd.concat([
    df_summary,
    pd.DataFrame([{
        "File": "PFAS_DB_mean",
        "KMD_mean_all": np.mean(pfas_kmd_all) if pfas_kmd_all else None,
        "KMD_to_Exact_mass_mean_all": np.mean(pfas_kmd_ratio_all) if pfas_kmd_ratio_all else None
    },{
        "File": "nonPFAS_DB_mean",
        "KMD_mean_all": np.mean(nonpfas_kmd_all) if nonpfas_kmd_all else None,
        "KMD_to_Exact_mass_mean_all": np.mean(nonpfas_kmd_ratio_all) if nonpfas_kmd_ratio_all else None
    }])
], ignore_index=True)

out_path = os.path.join(input_dir, "feature_dimensions_summary_NEG_5ppm.csv")
df_summary.to_csv(out_path, index=False)
