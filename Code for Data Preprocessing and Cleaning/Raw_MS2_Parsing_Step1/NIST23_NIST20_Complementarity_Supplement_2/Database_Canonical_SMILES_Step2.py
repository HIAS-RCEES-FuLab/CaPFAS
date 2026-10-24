import os
import pandas as pd

dirs = {
    "NIST23": r"NIST23_processed",
    "NIST20": r"NIST20",
    "MONA": r"MONA",
    "GNPS": r"GNPS"
}

all_smiles = {"P": set(), "N": set()}
all_rows = {"P": 0, "N": 0}

for name, folder in dirs.items():
    print(f"\n========== {name} ==========")

    for mode in ["P", "N"]:
        smiles_set = set()
        total_rows = 0

        for file in os.listdir(folder):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(folder, file)

            try:
                df = pd.read_csv(file_path, dtype=str, low_memory=False)

                if "Canonical_SMILES" not in df.columns or "Ion_mode" not in df.columns:
                    continue

                if name == "GNPS":
                    df["Ion_mode"] = (
                        df["Ion_mode"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .replace({
                            "positive": "P",
                            "negative": "N"
                        })
                    )
                else:
                    df["Ion_mode"] = (
                        df["Ion_mode"]
                        .astype(str)
                        .str.strip()
                    )

                df = df[df["Ion_mode"] == mode]

                if len(df) == 0:
                    continue

                smiles = df["Canonical_SMILES"].dropna()
                smiles = smiles[
                    smiles.astype(str).str.strip() != ""
                ]

                total_rows += len(smiles)

                smiles_set.update(smiles.astype(str))

            except Exception as e:
                print(f"{file} read fail: {e}")

        # Single file result
        print(
            f"{mode}: MS2 count={total_rows}, "
            f"Unique coumpound ={len(smiles_set)}"
        )

        # Cumulative
        all_rows[mode] += total_rows
        all_smiles[mode].update(smiles_set)

print("\n========== All databases combined ==========")
for mode in ["P", "N"]:
    print(
        f"{mode}: "
        f"MS2 count={all_rows[mode]}, "
        f"Unique coumpound ={len(all_smiles[mode])}"
    )