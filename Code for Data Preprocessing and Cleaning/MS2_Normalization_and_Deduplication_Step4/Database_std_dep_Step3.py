import os
import ast
import numpy as np
import pandas as pd

input_dir = r"MS2_grouped"
output_dir = r"MS2_filtered"
os.makedirs(output_dir, exist_ok=True)

def parse_spectrum(x):
    try:
        return [(float(mz), float(i)) for mz, i in ast.literal_eval(x)]
    except:
        return []

def cosine_ms2(spec1, spec2, mz_tol=0.005, min_intensity=1.0):
    spec1 = [(mz, i) for mz, i in spec1 if i >= min_intensity]
    spec2 = [(mz, i) for mz, i in spec2 if i >= min_intensity]

    peaks = []
    used = set()

    for mz1, i1 in spec1:
        best = None
        min_diff = mz_tol
        for j, (mz2, i2) in enumerate(spec2):
            if j in used:
                continue
            diff = abs(mz1 - mz2)
            if diff <= min_diff:
                min_diff = diff
                best = j
        if best is not None:
            peaks.append([i1, spec2[best][1]])
            used.add(best)
        else:
            peaks.append([i1, 0])

    for j, (mz2, i2) in enumerate(spec2):
        if j not in used:
            peaks.append([0, i2])

    v1 = np.array([x[0] for x in peaks])
    v2 = np.array([x[1] for x in peaks])

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0

    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


for ion_mode in ["N", "P"]:
    mode_dir = os.path.join(input_dir, ion_mode)
    save_dir = os.path.join(output_dir, ion_mode)
    os.makedirs(save_dir, exist_ok=True)

    for file in os.listdir(mode_dir):
        if not file.endswith(".csv"):
            continue

        csv_file = os.path.join(mode_dir, file)
        df = pd.read_csv(csv_file, low_memory=False)

        if "Canonical_SMILES" not in df.columns or "Mass_spectral" not in df.columns:
            continue

        print("Process:", ion_mode, file)

        remove_idx = set()

        for smiles, group in df.groupby("Canonical_SMILES"):
            idxs = group.index.tolist()

            if len(idxs) <= 1:
                continue

            spectra = {
                idx: parse_spectrum(df.loc[idx, "Mass_spectral"])
                for idx in idxs
            }

            # Sort
            idxs = sorted(
                idxs,
                key=lambda x: df.loc[x, "Num_peaks"],
                reverse=True
            )

            keep = []

            for idx in idxs:
                if idx in remove_idx:
                    continue

                duplicate = False

                for k in keep:
                    sim = cosine_ms2(
                        spectra[idx],
                        spectra[k]
                    )

                    if sim > 0.99:
                        duplicate = True
                        break

                if duplicate:
                    remove_idx.add(idx)
                else:
                    keep.append(idx)

        df = df.drop(index=list(remove_idx))

        output_file = os.path.join(save_dir, file)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(
            "Finish:",
            file,
            "delete:",
            len(remove_idx),
            "remain:",
            len(df)
        )