import os
import pandas as pd
from rdkit import Chem

# Data dir
dirs = [
    r"NIST23_processed",
    r"NIST20",
    r"MONA",
    r"GNPS"
]


def canonicalize_smiles(smiles):
    """
    SMILES convert to Canonical SMILES
    """
    try:
        if pd.isna(smiles) or str(smiles).strip() == "":
            return ""
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""

for folder in dirs:
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            file_path = os.path.join(folder, file)
            print(f"process: {file_path}")
            try:
                df = pd.read_csv(file_path)

                # Canonical SMILES
                df["Canonical_SMILES"] = df["SMILES"].apply(canonicalize_smiles)

                df.to_csv(file_path, index=False)

                print("  Finish")

            except Exception as e:
                print(f"  Fail: {e}")