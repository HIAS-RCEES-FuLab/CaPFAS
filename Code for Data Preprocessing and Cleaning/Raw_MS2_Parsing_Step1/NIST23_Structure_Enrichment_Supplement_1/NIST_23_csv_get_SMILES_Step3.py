import os
import pandas as pd

cache_file = r"PubChem_InChIKey_cache.csv"
input_dir = r"NIST23_processed"

cache_df = pd.read_csv(cache_file)

cache_df = cache_df[["InChIKey", "SMILES", "Pubchem_cid"]]

smiles_dict = dict(zip(cache_df["InChIKey"], cache_df["SMILES"]))
cid_dict = dict(zip(cache_df["InChIKey"], cache_df["Pubchem_cid"]))

for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(input_dir, file)

        print(f"Process: {file}")
        df = pd.read_csv(file_path)

        # InChikey matching
        df["SMILES"] = df["InChIKey"].map(smiles_dict)
        df["Pubchem_cid"] = df["InChIKey"].map(cid_dict)

        # Save file
        df.to_csv(file_path, index=False)

        print(f"Finished: {file}")