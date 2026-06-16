import os
import pandas as pd
import re

mass = {
    'H': 1.0078250322, 'C': 12.0, 'N': 14.0030740048,
    'O': 15.9949146196, 'F': 18.998403, 'P': 30.9737619985,
    'S': 31.9720711744, 'Cl': 34.968852682, 'Br': 78.91834,
    'I': 126.904473
}

def calc_formula_mass(formula):
    mass = 0.0
    pattern = r'([A-Z][a-z]*)(\d*)'
    for elem, count in re.findall(pattern, formula):
        count = int(count) if count else 1
        if elem not in mass:
            raise ValueError(f"Unknown element: {elem}")
        mass += mass[elem] * count
    return mass

def kendrick_mass_defect(exact_mass, unit_str="CF2"):
    exact_unit_mass = calc_formula_mass(unit_str)
    nominal_unit_mass = round(exact_unit_mass)

    if isinstance(exact_mass, (list, pd.Series)):
        results = []
        for m in exact_mass:
            km = m * nominal_unit_mass / exact_unit_mass
            kmd = round(km) - km
            results.append({
                'Exact_mass': m,
                'KM': km,
                'KMD': kmd,
                'Exact_unit_mass': exact_unit_mass,
                'Nominal_unit_mass': nominal_unit_mass
            })
        return results
    else:
        km = exact_mass * nominal_unit_mass / exact_unit_mass
        kmd = round(km) - km
        return {
            'Exact_mass': exact_mass,
            'KM': km,
            'KMD': kmd,
            'Exact_unit_mass': exact_unit_mass,
            'Nominal_unit_mass': nominal_unit_mass
        }

# target dir
base_dir = r"D:\MSMS\MSMS_H+H-\Train_data"
for root, dirs, files in os.walk(base_dir):
    for fname in files:
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(root, fname)
        df = pd.read_csv(fpath)
        df["KMD"] = df["Exact_mass"].apply(
            lambda m: kendrick_mass_defect(m, unit_str="CF2")["KMD"] if pd.notna(m) else None
        )
        df.to_csv(fpath, index=False)

