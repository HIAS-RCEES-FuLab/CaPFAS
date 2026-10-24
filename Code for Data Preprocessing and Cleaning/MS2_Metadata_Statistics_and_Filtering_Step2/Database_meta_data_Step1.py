import os
import pandas as pd

dirs = [
    r"NIST23_processed",
    r"NIST20_not_in_NIST23",
    r"MONA",
    r"GNPS"
]

# output dir
output_dir = r"Database_statistics"
os.makedirs(output_dir, exist_ok=True)

columns = [
    "Precursor_type",
    "Instrument_type",
    "Instrument",
    "Ion_mode"
]

for folder in dirs:

    print(f"\nProcess: {folder}")
    all_data = []

    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(folder, file)
        try:
            df = pd.read_csv(
                file_path,
                dtype=str,
                low_memory=False
            )

            for col in columns:
                if col not in df.columns:
                    continue
                tmp = df[col].fillna("Unknown").astype(str)
                tmp = (
                    tmp
                    .str.strip()
                    .replace("", "Unknown")
                )

                count_df = (
                    tmp
                    .value_counts()
                    .reset_index()
                )

                count_df.columns = [
                    "Value",
                    "Count"
                ]

                count_df["Category"] = col

                all_data.append(
                    count_df[
                        [
                            "Category",
                            "Value",
                            "Count"
                        ]
                    ]
                )

        except Exception as e:
            print(f"{file} read fail: {e}")

    if len(all_data) == 0:
        print("not data")
        continue

    stat_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # merge
    stat_df = (
        stat_df
        .groupby(
            [
                "Category",
                "Value"
            ],
            as_index=False
        )
        ["Count"]
        .sum()
    )

    # sort
    stat_df = stat_df.sort_values(
        [
            "Category",
            "Count"
        ],
        ascending=[
            True,
            False
        ]
    )

    # output
    name = os.path.basename(
        folder.rstrip("\\/")
    )

    output_file = os.path.join(
        output_dir,
        f"{name}_statistics.csv"
    )

    stat_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"finish: {output_file}")
    print(stat_df)


print("\nAll finish")