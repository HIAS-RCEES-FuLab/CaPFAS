import pandas as pd
import os
import uuid
import subprocess
import re

# -----------------------------
# Parameter
# -----------------------------
csv_file = "pos.csv"
results_dir = "./results_pos"
metfrag_jar = "MetFragCommandLine-2.6.7.jar"

start_idx = 0
os.makedirs(results_dir, exist_ok=True)

# -----------------------------
# Function
# -----------------------------
def extract_floats(s):
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", s)]

# -----------------------------
# Read CSV
# -----------------------------
df = pd.read_csv(csv_file)

for idx, row in df.iterrows():
    if idx < start_idx:
        continue
    precursor_mz = float(row['m/z'])
    ms2_mz_list = extract_floats(row['MS2_mz'])
    ms2_intensity_list = extract_floats(row['MS2_Intensity'])
    row_dir = os.path.join(results_dir, f"compound_{idx}")
    os.makedirs(row_dir, exist_ok=True)

    peak_file_path = os.path.join(row_dir, "peak_list.txt")
    max_intensity = max(ms2_intensity_list)
    with open(peak_file_path, "w") as f:
        for mz, intensity in zip(ms2_mz_list, ms2_intensity_list):
            if mz > 0 and intensity > 0:
                norm_intensity = intensity / max_intensity * 999
                f.write(f"{mz} {int(round(norm_intensity))}\n")

    param_file_path = os.path.join(row_dir, "metfrag_params.txt")
    param_template = f"""
PeakListPath = {peak_file_path}
MetFragDatabaseType = PubChem
IonizedPrecursorMass = {precursor_mz}
DatabaseSearchRelativeMassDeviation = 5.0
PrecursorIonMode = 1
IsPositiveIonMode = true
FragmentPeakMatchAbsoluteMassDeviation = 0.001
FragmentPeakMatchRelativeMassDeviation = 5
MetFragCandidateWriter = XLS
SampleName = MetFrag_Result_{uuid.uuid4().hex[:8]}
ResultsPath = {row_dir}
MaximumTreeDepth = 2
NumberThreads = 2
UseSmiles = true
MetFragScoreTypes = FragmenterScore
MetFragScoreWeights = 1.0
MetFragPreProcessingCandidateFilter = UnconnectedCompoundFilter
MetFragPostProcessingCandidateFilter = InChIKeyFilter
"""
    with open(param_file_path, "w") as f:
        f.write(param_template.strip())

    java_path = r"c:\Program Files\Java\jdk-25\bin\java.exe"
    cmd = [java_path, "-jar", metfrag_jar, param_file_path]
    print(f"Line {idx} command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Line {idx} MetFrag error: {e}")
        continue