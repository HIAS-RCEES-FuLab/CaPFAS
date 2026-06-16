import os
import pandas as pd
import ast
import numpy as np

# based dir
base_dir = r"D:\MSMS\MSMS_H+H-\Train_data"

# parameter setting
mz_lower = 50
mz_upper = 1000
mz_bin_width = 0.1
mz_bins = np.arange(mz_lower, mz_upper + mz_bin_width, mz_bin_width)

for root, dirs, files in os.walk(base_dir):
    db_min_bin = float("inf")
    db_max_bin = -float("inf")

    for fname in files:
        if not fname.endswith(".csv"):
            continue

        fpath = os.path.join(root, fname)
        df = pd.read_csv(fpath)

        if "Mass_spectral" not in df.columns:
            print(f"⚠️ Skip {fpath}, no 'Mass_spectral' column.")
            continue

        new_spectra_full = []

        for spec_str in df["Mass_spectral"].dropna():
            try:
                spectrum = ast.literal_eval(spec_str)
                spectrum_checked = [
                    (float(mz), float(i)) for mz, i in spectrum
                    if isinstance(mz, (int, float)) and isinstance(i, (int, float))
                ]
                if not spectrum_checked:
                    new_spectra_full.append("[]")
                    continue

                spectrum_checked.sort(key=lambda x: x[1], reverse=True)  # Sort by intensity in descending order
                spectrum_top = spectrum_checked[:50]  # top50
                global_max_intensity = max(inten for _, inten in spectrum_top)

                # --- binning ---
                binned = {}
                for mz, inten in spectrum_top:
                    if mz_lower <= mz <= mz_upper:
                        idx = int((mz - mz_lower) / mz_bin_width)
                        if idx not in binned:
                            binned[idx] = []
                        binned[idx].append((mz, inten))

                if not binned:
                    new_spectra_full.append("[]")
                    continue

                min_bin = min(binned.keys())
                max_bin = max(binned.keys())
                db_min_bin = min(db_min_bin, min_bin)
                db_max_bin = max(db_max_bin, max_bin)

                # (bin_index, m/z, normalized_intensity, KMD)
                full_features = []
                for idx in sorted(binned.keys()):
                    for mz_val, inten in binned[idx]:
                        norm_inten = round(inten / global_max_intensity, 4) if global_max_intensity > 0 else 0
                        full_features.append((idx, mz_val, norm_inten))

                new_spectra_full.append(str(full_features))

            except Exception as e:
                print(f"❌ Parse error in {fpath}: {e}")
                new_spectra_full.append(spec_str)

        df["Mass_spectral_features"] = new_spectra_full
        df.to_csv(fpath, index=False)
        print(f"✅ Updated file: {fpath}")

    if db_min_bin != float("inf"):
        print(f"[{root}] bin index range in all files: {db_min_bin} - {db_max_bin}")
