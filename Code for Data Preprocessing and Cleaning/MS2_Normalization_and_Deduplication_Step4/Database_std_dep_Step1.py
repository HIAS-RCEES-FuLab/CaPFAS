import os
import ast
import pandas as pd

input_dir = r"Database_filtered_selected"

def normalize_spectrum(x):
    try:
        peaks = ast.literal_eval(x)
        if not peaks:
            return []

        max_intensity = max(i[1] for i in peaks)
        if max_intensity == 0:
            return []

        normalized = [
            (mz, round(intensity / max_intensity * 100, 4))
            for mz, intensity in peaks
            if intensity / max_intensity * 100 >= 0.1
        ]
        return normalized
    except:
        return []

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if not file.endswith(".csv"):
            continue

        csv_file = os.path.join(root, file)

        try:
            df = pd.read_csv(csv_file, low_memory=False)

            if "Mass_spectral" not in df.columns:
                continue

            print("Process:", csv_file)

            df["Mass_spectral"] = df["Mass_spectral"].apply(normalize_spectrum)

            df["Num_peaks"] = df["Mass_spectral"].apply(len)

            df.to_csv(
                csv_file,
                index=False,
                encoding="utf-8-sig"
            )

        except Exception as e:
            print("Error:", csv_file, e)

print("All finish")