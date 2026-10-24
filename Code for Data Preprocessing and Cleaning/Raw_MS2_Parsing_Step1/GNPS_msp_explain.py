import re
import pandas as pd

msp_file = "GNPS/ALL_GNPS.msp"
output_csv = "GNPS/GNPS_data_"

patterns = {
    "Name": r"NAME:\s*(.*?)\s*(?=\n|$)",
    "Precursor_type": r"PRECURSORTYPE:\s*(.*?)\s*(?=\n|$)",
    "Precursor_mz": r"PRECURSORMZ:\s*(.*?)\s*(?=\n|$)",
    "Instrument_type": r"INSTRUMENTTYPE:\s*(.*?)\s*(?=\n|$)",
    "Instrument": r"INSTRUMENT:\s*(.*?)\s*(?=\n|$)",
    "Ion_mode": r"IONMODE:\s*(.*?)\s*(?=\n|$)",
    "InChIKey": r"INCHIKEY:\s*(.*?)\s*(?=\n|$)",
    "SMILES": r"SMILES:\s*(.*?)\s*(?=\n|$)",
    "Formula": r"FORMULA:\s*(.*?)\s*(?=\n|$)",
    "DB_ID": r"DB#=([\w\d]+)",
    "Num_peaks": r"Num Peaks:\s*(\d+)\s*(?=\n|$)"
}

new_column_order = [
    'ID', 'Name', 'Precursor_type', 'Precursor_mz', 'Instrument_type',
    'Instrument', 'Ionization', 'Collision_energy', 'Ion_mode',
    'InChIKey', 'SMILES', 'Pubchem_cid', 'Formula', 'MW', 'Exact_mass',
    'CAS_Num', 'DB_ID',
    'Num_peaks', 'Mass_spectral',
]

pos_data = []
neg_data = []
pos_count = 0  # POS
neg_count = 0  # NEG
batch_size = 100000  # batch
pos_file_count = 1
neg_file_count = 1
ID=0

with open(msp_file, 'r', encoding='utf-8') as file:
    spectrum = ""
    spectrum_data = {}
    empty_lines_count = 0
    for line in file:
        line = line.strip()

        if not line:
            empty_lines_count += 1
            if empty_lines_count >= 2:
                ID += 1
                if spectrum:
                    spectrum_data["ID"] = ID
                    for pattern_name, pattern in patterns.items():
                        matches = re.findall(pattern, spectrum, re.DOTALL)
                        if matches:
                            spectrum_data[pattern_name] = matches[0].strip()
                        else:
                            spectrum_data[pattern_name] = ""

                    ion_mode = spectrum_data.get("Ion_mode", "").strip().upper()

                    num_peaks_str = spectrum_data.get("Num_peaks", "").strip()
                    if num_peaks_str.isdigit():
                        num_peaks = int(num_peaks_str)
                    else:
                        num_peaks = None

                    mass_spectral_peaks_pattern = r"(\d+\.\d+)\s+(\d+\.\d+)"
                    peaks_data = re.findall(mass_spectral_peaks_pattern, spectrum)

                    raw_peaks = [
                        (float(mz), float(intensity))
                        for mz, intensity in peaks_data[:num_peaks]
                    ]

                    spectrum_data["Mass_spectral"] = raw_peaks
                    if ion_mode == "POSITIVE":
                        pos_data.append(spectrum_data)
                        pos_count += 1
                        if pos_count >= batch_size:
                            current_output_csv = f"{output_csv}POS_{pos_file_count}.csv"
                            df = pd.DataFrame(pos_data)
                            for col in new_column_order:
                                if col not in df.columns:
                                    df[col] = ""
                            df = df[new_column_order]
                            try:
                                df.to_csv(current_output_csv, index=False, encoding='utf-8')
                                print(f"POS save to {current_output_csv}")
                            except Exception as e:
                                print(f"Error: {e}")
                            # Reset
                            pos_data = []
                            pos_count = 0
                            pos_file_count += 1

                    elif ion_mode == "NEGATIVE":
                        neg_data.append(spectrum_data)
                        neg_count += 1
                        if neg_count >= batch_size:
                            current_output_csv = f"{output_csv}NEG_{neg_file_count}.csv"
                            df = pd.DataFrame(neg_data)
                            for col in new_column_order:
                                if col not in df.columns:
                                    df[col] = ""
                            df = df[new_column_order]
                            try:
                                df.to_csv(current_output_csv, index=False, encoding='utf-8')
                                print(f"NEG save to {current_output_csv}")
                            except Exception as e:
                                print(f"Error: {e}")
                            neg_data = []
                            neg_count = 0
                            neg_file_count += 1

                spectrum = ""
                spectrum_data = {}
                empty_lines_count = 0

        else:
            empty_lines_count = 0
            spectrum += line + "\n"

if pos_data:
    current_output_csv = f"{output_csv}POS_{pos_file_count}.csv"
    df = pd.DataFrame(pos_data)
    for col in new_column_order:
        if col not in df.columns:
            df[col] = ""
    df = df[new_column_order]
    df.to_csv(current_output_csv, index=False, encoding='utf-8')
    print(f"POS save to {current_output_csv}")

if neg_data:
    current_output_csv = f"{output_csv}NEG_{neg_file_count}.csv"
    df = pd.DataFrame(neg_data)
    for col in new_column_order:
        if col not in df.columns:
            df[col] = ""
    df = df[new_column_order]
    df.to_csv(current_output_csv, index=False, encoding='utf-8')
    print(f"NEG save to {current_output_csv}")