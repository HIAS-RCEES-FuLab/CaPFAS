import os
import pandas as pd
import subprocess

# Path setting
csv_file = r"CFM.csv"

# Output path
output_dir = r"CFM_predicted_spectra_neg"
os.makedirs(output_dir, exist_ok=True)

# Docker dir
docker_mount_dir_windows = os.path.abspath(output_dir).replace("\\", "/")
docker_mount_dir_docker = "/cfmid/public"

# read csv
df = pd.read_csv(csv_file)

start_idx = 0 # 比如从第11条开始预测

# Run docker cfm-predict
def run_cfm_in_docker(index, smiles):
    docker_out_file = f"{docker_mount_dir_docker}/{index}.log"
    print("挂载路径:", docker_mount_dir_windows)
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{docker_mount_dir_windows}:{docker_mount_dir_docker}",
        "-i",
        "wishartlab/cfmid:latest",
        "sh", "-c",
        f"cfm-predict '{smiles}' 0.001 /trained_models_cfmid4.0/[M-H]-/param_output.log "
        f"/trained_models_cfmid4.0/[M-H]-/param_config.txt 0 {docker_out_file}"
    ]
    subprocess.run(cmd, check=True)

# batch run
for i, row in df.iterrows():
    if i < start_idx:
        continue
    idx = row["index"] if "index" in df.columns else i
    smiles = row["SMILES"]
    if pd.isna(smiles) or smiles.strip() == "":
        continue
    run_cfm_in_docker(idx, smiles)