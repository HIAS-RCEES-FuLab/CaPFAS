import pandas as pd
import numpy as np

# parameter setting
edge_file = r"D:\MSMS\MSMS_H+H-\Train_data\MS2_similarity_edges_byID_Scenario3_weisi_to_DB_propagation_1.csv"
db_file = r"D:\MSMS\MSMS_H+H-\Train_data\NEG_all_neutral_loss.csv"
output_file = r"D:\MSMS\MSMS_H+H-\Train_data\coverage_stats.csv"

# -----------------------------
# Read input files
# -----------------------------
df_edge = pd.read_csv(edge_file)
df_db = pd.read_csv(db_file)

# Edge processing:
# Remove duplicate targets and count occurrences
df_unique = df_edge.drop_duplicates(subset=['target'])
edge_count = df_unique['target_file'].value_counts()

# Database processing:
# Count entries for each source
db_count = df_db['Source'].value_counts()

# Merge results and calculate coverage
result = pd.DataFrame({
    'matched_count': edge_count,
    'total_count': db_count
}).fillna(0)

# Avoid division by zero when calculating coverage
result['coverage_%'] = np.where(
    result['total_count'] > 0,
    result['matched_count'] / result['total_count'] * 100,
    0
)

result['coverage_%'] = result['coverage_%'].round(2)

result['coverage_str'] = result['coverage_%'].astype(str) + '%'

result = result.sort_values(by='coverage_%', ascending=False)

print("Number of unique targets after deduplication:", len(df_unique))
print("\nCoverage statistics by source:")
print(result)

result.to_csv(output_file, encoding="utf-8-sig")

print(f"\nResults have been saved to: {output_file}")