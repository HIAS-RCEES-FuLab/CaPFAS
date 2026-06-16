import pandas as pd
import os

# Downsample function
def sample_nonpfas(nonpfas_df, pfas_df, ratio=2, random_state=42):
    target_count = len(pfas_df) * ratio
    if len(nonpfas_df) > target_count:
        return nonpfas_df.sample(n=target_count, random_state=random_state)
    else:
        return nonpfas_df

# input dir and output dir setting
input_dir = r"D:\MSMS\MSMS_H+H-\GNPS"
output_dir = os.path.join(input_dir, "sampled")
os.makedirs(output_dir, exist_ok=True)

pfas_pos_file = os.path.join(input_dir, "gnps_PFAS_POS.csv")
pfas_neg_file = os.path.join(input_dir, "gnps_PFAS_NEG.csv")
nonpfas_pos_file = os.path.join(input_dir, "gnps_nonPFAS_POS.csv")
nonpfas_neg_file = os.path.join(input_dir, "gnps_nonPFAS_NEG.csv")

pfas_pos = pd.read_csv(pfas_pos_file)
pfas_neg = pd.read_csv(pfas_neg_file)
nonpfas_pos = pd.read_csv(nonpfas_pos_file)
nonpfas_neg = pd.read_csv(nonpfas_neg_file)

nonpfas_pos_sampled = sample_nonpfas(nonpfas_pos, pfas_pos, ratio=2)
nonpfas_neg_sampled = sample_nonpfas(nonpfas_neg, pfas_neg, ratio=2)

nonpfas_pos_sampled.to_csv(os.path.join(output_dir, "gnps_nonPFAS_POS_sampled.csv"), index=False)
nonpfas_neg_sampled.to_csv(os.path.join(output_dir, "gnps_nonPFAS_NEG_sampled.csv"), index=False)
