import os
import csv
import re
import math

input_dir = r"NIST23\NIST23_1"
input_dir2 = r"NIST23\NIST23_2"
output_dir = r"NIST23"
os.makedirs(output_dir, exist_ok=True)
chunk_size = 100000

fields = [
    "Name",
    "Precursor_type",
    "Precursor_mz",
    "Instrument_type",
    "Instrument",
    "Ionization",
    "Collision_energy",
    "Ion_mode",
    "InChIKey",
    "Formula",
    "MW",
    "Exact_Mass",
    "CAS_Num",
    "DB_ID",
    "Num_Peaks"
]

# NIST mapping
field_mapping = {
    "Precursor_mz": "PrecursorMZ",
    "CAS_Num": "CAS#",
    "DB_ID": "DB#",
}

def parse_block(block):
    data = {k: None
        for k in fields
    }
    mz_list = []
    intensity_list = []
    peak_section = False
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue

        # --------------------------
        # Num Peaks
        # --------------------------
        if line.startswith("Num Peaks:"):
            try:
                data["Num_Peaks"] = int(line.split(":",1)[1].strip())
            except:
                pass
            peak_section = True
            continue

        # --------------------------
        # Peaks
        # --------------------------
        if peak_section:
            parts = line.split()
            if len(parts)>=2:
                try:
                    mz = float(parts[0])
                    intensity = float(parts[1])
                    mz_list.append(mz)
                    intensity_list.append(intensity)
                except:
                    pass
            continue

        # --------------------------
        # Metadata
        # --------------------------
        for field in fields:
            source_field = field_mapping.get(field,field)
            if line.startswith(source_field + ":"):
                value = line.split(":",1)[1].strip()
                if field == "DB_ID":
                    try:
                        value = int(value)
                    except:
                        pass
                data[field] = value
                break

    # ==========================
    # Mass_spectral
    # ==========================
    data["Mass_spectral"] = list(zip(mz_list,intensity_list))
    return data

# ==========================
# Read MSPEC
# ==========================
def read_mspec_file(file):
    with open(file,"r",encoding="utf-8",errors="ignore") as f:
        text = f.read()
    blocks = re.split(r"(?=^Name:)",text,flags=re.MULTILINE)
    records = []
    for block in blocks:
        if block.startswith("Name:"):
            records.append(parse_block(block))
    return records

def check_spectrum(peaks):
    if not isinstance(peaks,list):
        return False
    for p in peaks:
        if not isinstance(p,tuple):
            return False
        if len(p)!=2:
            return False
        mz = p[0]
        intensity = p[1]
        if not isinstance(mz,(int,float)):
            return False
        if not isinstance(intensity,(int,float)):
            return False
        if not math.isfinite(mz):
            return False
        if not math.isfinite(intensity):
            return False
    return True

# ==========================
# Write CSV
# ==========================
def write_csv(records,index):
    output_csv = os.path.join(output_dir,f"NIST23_MSP_summary_{index:03d}.csv")
    columns = [
        "Name",
        "Precursor_type",
        "Precursor_mz",
        "Instrument_type",
        "Instrument",
        "Ionization",
        "Collision_energy",
        "Ion_mode",
        "InChIKey",
        "Formula",
        "MW",
        "Exact_Mass",
        "CAS_Num",
        "DB_ID",
        "Num_Peaks",
        "Mass_spectral"
    ]

    with open(output_csv,"w",newline="",encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        for row in records:
            row["Mass_spectral"] = str(row["Mass_spectral"])
            writer.writerow(row)

# ==========================
# main
# ==========================
all_records = []
files = []
# NIST23_1
for i in range(1,211):
    file = os.path.join(input_dir,f"{i}.MSPEC")
    if os.path.exists(file):
        files.append(file)

# NIST_2
for i in range(1,179):
     file=os.path.join(input_dir2,f"{i}.MSPEC")
     if os.path.exists(file):
         files.append(file)

print("MSPEC file number:",len(files))

for file in files:
    print("read:",os.path.basename(file))
    records = read_mspec_file(file)
    all_records.extend(records)
    print("Spectra count:",len(all_records))

print("===================")
print("Total spectra:",len(all_records))

# ==========================
# check
# ==========================
bad = []
for i,row in enumerate(all_records):
    spectral_ok = check_spectrum(row["Mass_spectral"])
    length_ok = (
        row["Num_Peaks"] is not None and len(row["Mass_spectral"]) == row["Num_Peaks"]
    )

    if not (spectral_ok and length_ok):
        bad.append([i+1, row["DB_ID"], row["Name"], row["Num_Peaks"], len(row["Mass_spectral"])])

print("===================")
print("Error spectra:",len(bad))

# ==========================
# Batch save
# ==========================
count = 0
file_index = 1
while count < len(all_records):
    chunk = all_records[count:count + chunk_size]
    write_csv(chunk,file_index)
    count += chunk_size
    file_index += 1
print("===================")
print("All finish")