import pandas as pd
import matplotlib.pyplot as plt

file_path = r"all_test_results_best_pr_auc_after3.csv"

df = pd.read_csv(file_path)
total = 4458

# =========================
# get data
# =========================
tp_1 = df.iloc[0]["TP"]
tp_5 = df.iloc[4]["TP"]
improvement_1 = tp_5 - tp_1
unidentified_1 = total - tp_5

tp_2 = df.iloc[1]["TP"]
tp_6 = df.iloc[5]["TP"]
improvement_2 = tp_6 - tp_2
unidentified_2 = total - tp_6

tp_4 = df.iloc[3]["TP"]
tp_ref = 3582
improvement_3 = tp_ref - tp_4
unidentified_3 = total - tp_ref

data = [
    (tp_1, improvement_1, unidentified_1),
    (tp_2, improvement_2, unidentified_2),
    (tp_4, improvement_3, unidentified_3)
]

colors = [
    "#99CCFF",
    "#F4A6A6",
    "#D9D9D9"
]

for i, values in enumerate(data, start=1):
    fig, ax = plt.subplots(figsize=(3, 3))

    explode = [0.04, 0.04, 0.04]

    ax.pie(
        values,
        startangle=90,
        colors=colors,
        explode=explode,
        shadow=True,
        labels=None,
        autopct=None
    )

    ax.axis("equal")

    output_file = rf"TP_improvement_pie_{i}.png"

    plt.savefig(
        output_file,
        dpi=300,
        transparent=True,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"已保存：{output_file}")