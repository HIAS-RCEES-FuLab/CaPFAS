import os
import pandas as pd

dirs = {
    "NIST23": r"NIST23_processed",
    "NIST20": r"NIST20_not_in_NIST23",
    "MONA": r"MONA",
    "GNPS": r"GNPS"
}

output_dir = r"Database_filtered_split"
os.makedirs(output_dir, exist_ok=True)

precursor_map = {
    "M+H": "[M+H]+",
    "[M+H]": "[M+H]+",
    "M-H": "[M-H]-",
    "[M-H]": "[M-H]-"
}

nist_instrument_map = {
    "Thermo Finnigan Elite Orbitrap": "Orbitrap",
    "Orbitrap Fusion Lumos": "Orbitrap",
    "Thermo Finnigan Velos Orbitrap": "Orbitrap",
    "Agilent QTOF 6530": "QTOF"
}

def normalize_instrument(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    if ("orbitrap" in x or "q-exactive" in x or "q exactive" in x or "qexactive" in x or
        "qehf" in x or "qeft" in x or "qft" in x or "hybrid ft" in x or
        "hybrid ftms" in x or "ftms" in x or "ft-icr" in x or "fticr" in x or
        "ltq-ft" in x or "ltq ft" in x or "esi-ft" in x or "qit-ft" in x):
        return "Orbitrap"
    if ("qtof" in x or "q-tof" in x or "q tof" in x or "qtof/ms" in x or
        "q-tof/ms" in x or "synapt qtof" in x or "maxis hd qtof" in x or
        "maxis ii hd q-tof" in x):
        return "QTOF"
    return None

for name, folder in dirs.items():
    print(f"\nProcess: {name}")
    result = []

    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(folder, file)

        try:
            df = pd.read_csv(path, dtype=str, low_memory=False)

            # Precursor
            if "Precursor_type" not in df.columns:
                continue

            df["Precursor_type"] = df["Precursor_type"].replace(precursor_map).astype(str).str.strip()
            df = df[df["Precursor_type"].isin(["[M+H]+", "[M-H]-"])]

            # Instrument_type
            if name in ["NIST23", "NIST20"]:
                if "Instrument" in df.columns:
                    for k, v in nist_instrument_map.items():
                        df.loc[
                            df["Instrument"].str.contains(k, case=False, na=False),
                            "Instrument_type"
                        ] = v

            elif name == "MONA":
                if "Instrument_type" in df.columns:
                    df["Instrument_type"] = df["Instrument_type"].apply(normalize_instrument)

            elif name == "GNPS":
                if "Instrument" in df.columns:
                    df["Instrument_type"] = df["Instrument"].apply(normalize_instrument)

            if "Instrument_type" not in df.columns:
                continue

            # Only Orbitrap和QTOF
            df = df[df["Instrument_type"].isin(["Orbitrap", "QTOF"])]

            # Ion mode
            if "Ion_mode" not in df.columns:
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
                df = df[df["Ion_mode"].isin(["P", "N"])]
            else:
                df["Ion_mode"] = df["Ion_mode"].astype(str).str.strip()

            # Canonical SMILES
            if "Canonical_SMILES" not in df.columns:
                continue

            df["Canonical_SMILES"] = (
                df["Canonical_SMILES"]
                .astype(str)
                .str.strip()
            )

            df = df[
                (df["Canonical_SMILES"] != "") &
                (df["Canonical_SMILES"].str.lower() != "nan") &
                (df["Canonical_SMILES"].str.lower() != "none")
            ]

            if len(df) == 0:
                continue

            result.append(df)

        except Exception as e:
            print(f"{file} fail: {e}")

    if result:
        out_df = pd.concat(result, ignore_index=True)
        chunk_size = 100000

        for mode in out_df["Ion_mode"].dropna().unique():
            mode_df = out_df[out_df["Ion_mode"] == mode]

            if len(mode_df) == 0:
                continue

            split_dir = os.path.join(output_dir, name, str(mode))
            os.makedirs(split_dir, exist_ok=True)

            for i, start in enumerate(range(0, len(mode_df), chunk_size), 1):
                chunk = mode_df.iloc[start:start + chunk_size]

                out_file = os.path.join(
                    split_dir,
                    f"{name}_{mode}_{i}.csv"
                )

                chunk.to_csv(
                    out_file,
                    index=False,
                    encoding="utf-8-sig"
                )

                print(f"Save: {out_file} ({len(chunk)}条)")

            print(f"{name}-{mode}: {len(mode_df)}条")

print("All finish")