import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

input_file = r"PFAS_structure_annotation.csv"
output_file = r"PFAS_structure_annotation_scaffold.csv"

c2f4_pattern = Chem.MolFromSmarts("[C](F)(F)-[C](F)(F)")

def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)

        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return "Acyclic"

        return Chem.MolToSmiles(scaffold)

    except:
        return None

def check_c2f4(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return int(mol.HasSubstructMatch(c2f4_pattern))
    except:
        return 0

df = pd.read_csv(input_file)

df["Scaffold"] = df["Canonical_SMILES"].apply(get_scaffold)

df["Perfluoroalkyl_chain"] = df["Canonical_SMILES"].apply(check_c2f4)

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(df[
    [
        "Canonical_SMILES",
        "Scaffold",
        "Perfluoroalkyl_chain"
    ]
].head())