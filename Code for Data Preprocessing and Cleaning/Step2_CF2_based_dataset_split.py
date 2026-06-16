import pandas as pd
import glob
import os
import re
from rdkit import Chem

# molecular formula parsing
def parse_molecular_formula(formula):
    if pd.isna(formula):
        return None
    if '+' in formula or '-' in formula:
        return None
    element_counts = {}
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    for element, count in matches:
        count = int(count) if count else 1
        element_counts[element] = element_counts.get(element, 0) + count
    return element_counts

def contains_subformula(mol_elements, target_elements):
    for elem, count in target_elements.items():
        if mol_elements.get(elem, 0) < count:
            return False
    return True

# Check for the presence of molecular substructures
def smiles_has_substructure(candidate_smiles, query_smiles):
    try:
        if pd.isna(candidate_smiles):
            return False
        mol = Chem.MolFromSmiles(candidate_smiles)
        query = Chem.MolFromSmiles(query_smiles)
        if mol is None or query is None:
            return False
        return mol.HasSubstructMatch(query)
    except:
        return False

# Main workflow
def extract_pfas_nonpfas(file_pattern, target_subformula, substructure_smiles, mode_label):
    files = sorted(glob.glob(file_pattern))
    all_rows = []

    target_elements = parse_molecular_formula(target_subformula)

    for file in files:
        df = pd.read_csv(file)
        df = df.dropna(subset=['Formula'])

        df['ElementCounts'] = df['Formula'].apply(parse_molecular_formula)
        df = df[df['ElementCounts'].notna()]

        smiles_col = 'CanonicalSMILES' if 'CanonicalSMILES' in df.columns else 'SMILES'
        df = df[df[smiles_col].notna()]

        df['Has_CF2'] = df['ElementCounts'].apply(lambda x: contains_subformula(x, target_elements))
        df['Has_Substructure'] = df[smiles_col].apply(lambda x: smiles_has_substructure(x, substructure_smiles))

        df['Label'] = df.apply(lambda row: "PFAS" if (row['Has_CF2'] and row['Has_Substructure']) else "non-PFAS", axis=1)
        df['Mode'] = mode_label

        all_rows.append(df)

    if all_rows:
        result = pd.concat(all_rows, ignore_index=True)
    else:
        result = pd.DataFrame()

    return result

# Parameter settings
base_dir = r"D:\MSMS"
extract_dir = os.path.join(base_dir, "MSMS_extract")

target_subformula = 'CF2'
substructure_smiles = 'C(F)F'

pos_pattern = os.path.join(extract_dir, 'gnps_data_POS_*.csv')
neg_pattern = os.path.join(extract_dir, 'gnps_data_NEG_*.csv')

pos_df = extract_pfas_nonpfas(pos_pattern, target_subformula, substructure_smiles, mode_label="POS")
neg_df = extract_pfas_nonpfas(neg_pattern, target_subformula, substructure_smiles, mode_label="NEG")

pos_df[pos_df['Label']=="PFAS"].to_csv(os.path.join(base_dir, "gnps_PFAS_POS.csv"), index=False)
neg_df[neg_df['Label']=="PFAS"].to_csv(os.path.join(base_dir, "gnps_PFAS_NEG.csv"), index=False)
pos_df[pos_df['Label']=="non-PFAS"].to_csv(os.path.join(base_dir, "gnps_nonPFAS_POS.csv"), index=False)
neg_df[neg_df['Label']=="non-PFAS"].to_csv(os.path.join(base_dir, "gnps_nonPFAS_NEG.csv"), index=False)
