import re
import pandas as pd
# 原子精确质量字典
isotope_mass = {
    'H': 1.0078250322, 'C': 12.0, 'N': 14.0030740048,
    'O': 15.9949146196, 'F': 18.998403, 'P': 30.9737619985,
    'S': 31.9720711744, 'Cl': 34.968852682, 'Br': 78.91834,
    'I': 126.904473
}

def calc_formula_mass(formula):
    """根据化学式计算精确质量"""
    mass = 0.0
    pattern = r'([A-Z][a-z]*)(\d*)'
    for elem, count in re.findall(pattern, formula):
        count = int(count) if count else 1
        if elem not in isotope_mass:
            raise ValueError(f"Unknown element: {elem}")
        mass += isotope_mass[elem] * count
    return mass

def kendrick_mass_defect(exact_mass, unit_str="CF2"):
    """计算Kendrick Mass和Kendrick Mass Defect"""
    exact_unit_mass = calc_formula_mass(unit_str)
    nominal_unit_mass = round(exact_unit_mass)

    def calculate(m):
        km = m * nominal_unit_mass / exact_unit_mass
        kmd = km - round(km)
        return {
            "Exact_mass": m,
            "KM": km,
            "KMD": kmd,
            "Exact_unit_mass": exact_unit_mass,
            "Nominal_unit_mass": nominal_unit_mass
        }

    if isinstance(exact_mass, (list, pd.Series)):
        return [calculate(m) for m in exact_mass]
    else:
        return calculate(exact_mass)

# 示例
mass = 264.1234
result = kendrick_mass_defect(mass, unit_str="CF2")
print(result)
