import pandas as pd
import numpy as np

# APP-ID reference spectra
weisi_file = r"D:\MS\PFAS_compare\weisics\PFAS-library_neutral_loss.csv"
# input spectra
db_file = r"D:\MSMS\MSMS_H+H-\Train_data\NEG_all_neutral_loss.csv"
# Only one iteration is performed for reference MS/MS database data, meaning only the initially connected network nodes are retained
max_iterations = 1
output_edge_file_template = r"D:\MSMS\MSMS_H+H-\Train_data\MS2_similarity_edges_byID_Scenario3_weisi_to_DB_propagation_{}.csv"
output_edge_file = output_edge_file_template.format(max_iterations)
# parameter setting
abs_tol = 0.01
sim_threshold = 0.5
COMMON_NL = [0, 43.98983, 79.95682]
PFAS_FEATURE_IONS = [98.95576, 82.96085]
MIN_MZ = 100

# read file
df_weisi = pd.read_csv(weisi_file)
df_db = pd.read_csv(db_file)

def str2list(s):
    if pd.isna(s) or s == "":
        return []
    return [float(x) for x in s.split(";")]

df_weisi['MS2_mz_list'] = df_weisi['MS2_mz_filtered'].apply(str2list)
df_weisi['NL_list'] = df_weisi['Neutral_Loss'].apply(str2list)
df_db['MS2_mz_list'] = df_db['MS2_mz_filtered'].apply(str2list)
df_db['NL_list'] = df_db['Neutral_Loss'].apply(str2list)

def filter_fragments(frag_list, nl_list):
    nl_filtered = [x for x in nl_list if all(abs(x - nl) > 1e-2 for nl in COMMON_NL)]
    frag_filtered = [x for x in frag_list if x >= MIN_MZ or any(abs(x - pf) <= 1e-2 for pf in PFAS_FEATURE_IONS)]
    return frag_filtered, nl_filtered

df_weisi[['MS2_mz_list', 'NL_list']] = df_weisi.apply(
    lambda row: pd.Series(filter_fragments(row['MS2_mz_list'], row['NL_list'])), axis=1
)
df_db[['MS2_mz_list', 'NL_list']] = df_db.apply(
    lambda row: pd.Series(filter_fragments(row['MS2_mz_list'], row['NL_list'])), axis=1
)

def match_count(listA, listB):
    if not listA or not listB:
        return 0
    arrB = np.array(listB)
    count = 0
    for a in listA:
        if np.any(np.abs(arrB - a) <= abs_tol):
            count += 1
    return count

# main
edges = []
current_sources = df_weisi.copy()
visited_ids = set(current_sources['ID'] if 'ID' in current_sources else [f"weisi_{i}" for i in range(len(current_sources))])

for iteration in range(max_iterations):
    print(f"Iteration {iteration+1}, source nodes: {len(current_sources)}")
    new_targets = []

    for i, rowA in current_sources.iterrows():
        A_frag, A_nl = rowA['MS2_mz_list'], rowA['NL_list']
        A_id = rowA['ID'] if 'ID' in rowA else f"weisi_{i}"
        A_source = rowA['Source'] if 'Source' in rowA else "weisi"

        for j, rowB in df_db.iterrows():
            B_frag, B_nl = rowB['MS2_mz_list'], rowB['NL_list']
            B_id = rowB['ID'] if 'ID' in rowB else f"db_{j}"
            B_source = rowB['Source'] if 'Source' in rowB else "mergedDB"

            if A_id == B_id:
                continue

            N_AB = match_count(A_frag, B_frag) + match_count(A_nl, B_nl)
            N_BA = match_count(B_frag, A_frag) + match_count(B_nl, A_nl)

            N_A = len(A_frag) + len(A_nl)
            N_B = len(B_frag) + len(B_nl)

            if N_A > 0 and N_B > 0:
                sim = ((N_AB * N_BA) / (N_A * N_B)) ** 0.5
            else:
                sim = 0

            if sim >= sim_threshold and sim > 0:
                edges.append({
                    "source": A_id,
                    "target": B_id,
                    "source_file": A_source,
                    "target_file": B_source,
                    "weight": round(sim, 5)
                })
                if B_id not in visited_ids:
                    new_targets.append(rowB)
                    visited_ids.add(B_id)

    if not new_targets:
        break
    current_sources = pd.DataFrame(new_targets)

edges_df = pd.DataFrame(edges)
edges_df.to_csv(output_edge_file, index=False, encoding="utf-8-sig")
print(f"Total edge: {len(edges_df)}")
