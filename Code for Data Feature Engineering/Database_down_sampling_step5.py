import os
import pandas as pd

input_dir = r"MS2_feature_processed_I_1_down_sampling_type2_12_to_1"
output_dir = r"MS2_feature_processed_I_1_down_sampling_result_12_to_1"

chunk_size = 100000

databases = ["GNPS", "MONA", "NIST20", "NIST23"]

os.makedirs(output_dir, exist_ok=True)

for ion_mode in ["N", "P"]:

    input_mode_dir = os.path.join(input_dir, ion_mode)
    output_mode_dir = os.path.join(output_dir, ion_mode)

    os.makedirs(output_mode_dir, exist_ok=True)

    csv_files = [
        f for f in os.listdir(input_mode_dir)
        if f.endswith(".csv")
    ]

    print("\n======", ion_mode, "======")
    db_data = {db: [] for db in databases}

    for file in sorted(csv_files):

        file_path = os.path.join(input_mode_dir, file)

        df = pd.read_csv(file_path, low_memory=False)

        for db in databases:

            temp = df[df["Database"] == db]

            if len(temp) > 0:
                db_data[db].append(temp)

    for db in databases:
        if not db_data[db]:
            continue
        print("\nProcess:", ion_mode, db)
        out_dir = os.path.join(output_mode_dir, db)
        os.makedirs(out_dir, exist_ok=True)
        data = pd.concat(
            db_data[db],
            ignore_index=True
        )

        data["ID"] = range(1, len(data) + 1)
        print("Count:", len(data))
        file_index = 1
        for start in range(0, len(data), chunk_size):
            end = min(
                start + chunk_size,
                len(data)
            )
            chunk = data.iloc[start:end]
            output_file = os.path.join(
                out_dir,
                f"{file_index}.csv"
            )
            chunk.to_csv(
                output_file,
                index=False
            )
            print(
                f"{file_index}.csv:",
                f"ID {start+1}-{end}"
            )
            file_index += 1

print("\nAll finish")