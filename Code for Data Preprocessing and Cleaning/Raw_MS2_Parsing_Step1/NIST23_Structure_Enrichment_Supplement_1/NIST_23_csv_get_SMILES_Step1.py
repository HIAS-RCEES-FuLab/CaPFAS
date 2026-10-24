import os
import pandas as pd
import requests
import time

input_dir = r"NIST23"
output_dir = r"NIST23_with_cid_SMILES"
cache_file = r"PubChem_InChIKey_cache.csv"

os.makedirs(output_dir, exist_ok=True)
columns = [
    "ID",
    "Name",
    "Precursor_type",
    "Precursor_mz",
    "Instrument_type",
    "Instrument",
    "Ionization",
    "Collision_energy",
    "Ion_mode",
    "InChIKey",
    "SMILES",
    "Pubchem_cid",
    "Formula",
    "MW",
    "Exact_mass",
    "CAS_Num",
    "DB_ID",
    "Num_peaks",
    "Mass_spectral"
]

cache = {}
if os.path.exists(cache_file):
    cache_df = pd.read_csv(cache_file, encoding="utf-8-sig")
    for _, row in cache_df.iterrows():
        cache[str(row["InChIKey"])] = (row["SMILES"],row["Pubchem_cid"])
print("Cache len:", len(cache))

def get_pubchem(inchikey):
    inchikey = str(inchikey).strip()
    if inchikey in cache:
        return cache[inchikey]
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        f"compound/inchikey/{inchikey}/property/"
        "CanonicalSMILES,IsomericSMILES/JSON"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            cache[inchikey] = (None, None)
            return None, None
        prop = r.json()["PropertyTable"]["Properties"][0]
        cid = prop.get("CID")
        smiles = (
            prop.get("CanonicalSMILES")
            or prop.get("ConnectivitySMILES")
            or prop.get("SMILES")
            or prop.get("IsomericSMILES")
        )
        cache[inchikey] = (smiles, cid)
        return smiles, cid

    except:
        cache[inchikey] = (None, None)
        return None, None

def save_cache():
    df = pd.DataFrame(
        [
            {
                "InChIKey": k,
                "SMILES": v[0],
                "Pubchem_cid": v[1]
            }
            for k, v in cache.items()
        ]
    )
    df.to_csv(
        cache_file,
        index=False,
        encoding="utf-8-sig"
    )
files = sorted(
    [
        f for f in os.listdir(input_dir)
        if f.endswith(".csv")
    ]
)

print("CSV number:", len(files))

for file in files:
    print("\nProcess:", file)
    input_file = os.path.join(input_dir,file)
    df = pd.read_csv(input_file,encoding="utf-8-sig")

    inchikeys = (df["InChIKey"].dropna().astype(str).str.strip().unique())

    print("Data:",len(df),"Unique InChIKey:",len(inchikeys))

    for i, inchikey in enumerate(inchikeys):
        if inchikey not in cache:
            get_pubchem(inchikey)
            time.sleep(0.05)
        if (i + 1) % 100 == 0:
            print("Request:",i + 1,"/",len(inchikeys))

    df["SMILES"] = df["InChIKey"].map(lambda x: cache.get(str(x).strip(),(None, None))[0])
    df["Pubchem_cid"] = df["InChIKey"].map(lambda x: cache.get(str(x).strip(),(None, None))[1])
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]
    output_file = os.path.join(output_dir,file)
    df.to_csv(output_file,index=False,encoding="utf-8-sig")
    save_cache()
    print("Save:",output_file,len(df))

print("\nFinish")
print("Final cache:", len(cache))