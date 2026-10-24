import pandas as pd
import requests
import time
import os


cache_file = r"PubChem_InChIKey_cache.csv"
df = pd.read_csv(cache_file,encoding="utf-8-sig")
print("Cache number:", len(df))

# ==========================
# Find SMILES is null
# ==========================
mask = (
    df["SMILES"].isna()
    |
    df["SMILES"]
    .astype(str)
    .str.strip()
    .isin(["", "nan", "None"])
)

missing = df[mask]

print("Need request:",len(missing))

# ==========================
# PubChem request
# ==========================
def get_pubchem(inchikey):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        f"compound/inchikey/{inchikey}/property/"
        "CanonicalSMILES,ConnectivitySMILES,IsomericSMILES/JSON"
    )

    for retry in range(3):
        try:
            r = requests.get(
                url,
                timeout=15
            )

            if r.status_code == 200:
                data = r.json()
                prop = (
                    data
                    ["PropertyTable"]
                    ["Properties"][0]
                )

                cid = prop.get("CID")

                smiles = (
                    prop.get("CanonicalSMILES")
                    or prop.get("ConnectivitySMILES")
                    or prop.get("SMILES")
                    or prop.get("IsomericSMILES")
                )
                return smiles, cid

            else:
                time.sleep(2)

        except Exception as e:
            print(
                "Error:",
                inchikey,
                "retry:",
                retry + 1,
                e
            )
            time.sleep(2)

    return None, None

success = 0
failed = 0
for count, (idx, row) in enumerate(
    missing.iterrows(),
    start=1
):
    inchikey = str(row["InChIKey"]).strip()
    smiles, cid = get_pubchem(inchikey)

    if smiles:
        df.loc[idx, "SMILES"] = smiles
        if cid:
            df.loc[idx, "Pubchem_cid"] = cid
        success += 1

    else:
        failed += 1

    if count % 100 == 0:
        df.to_csv(
            cache_file,
            index=False,
            encoding="utf-8-sig"
        )
        print(
            f"Process {count}/{len(missing)}",
            "success:",
            success,
            "finish:",
            failed
        )
    time.sleep(0.05)

# ==========================
# Final save
# ==========================
df.to_csv(
    cache_file,
    index=False,
    encoding="utf-8-sig"
)
print("\nFinish")