import os
import pandas as pd
import re

input_dir = r"Database_filtered_split"
output_dir = r"Database_filtered_selected"

allowed_elements = {'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Si'}

filtered_count = 0
csv_count = 0
saved_files = 0

def get_elements(formula):
    return set(re.findall(r"[A-Z][a-z]?", str(formula)))

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".csv"):
            csv_path = os.path.join(root, file)
            csv_count += 1

            df = pd.read_csv(csv_path)

            df["Precursor_mz"] = pd.to_numeric(df["Precursor_mz"], errors="coerce")

            mask = []

            for _, row in df.iterrows():
                keep = False

                if pd.notna(row["Precursor_mz"]) and row["Precursor_mz"] <= 1000:
                    if pd.notna(row["Formula"]):
                        elements = get_elements(row["Formula"])
                        if elements and elements.issubset(allowed_elements):
                            keep = True

                mask.append(keep)

            filtered_df = df[mask]

            filtered_count += len(filtered_df)

            if len(filtered_df) > 0:
                relative_path = os.path.relpath(root, input_dir)
                save_dir = os.path.join(output_dir, relative_path)

                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, file)
                filtered_df.to_csv(save_path, index=False)

                saved_files += 1

print("Finish")