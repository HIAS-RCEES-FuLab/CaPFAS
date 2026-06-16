import matplotlib.pyplot as plt

# Plotting data
neutral_loss_data = {
    "GNPS": {"present": 439 + 1065, "absent": 9338 + 12469},
    "MONA": {"present": 112 + 178, "absent": 940 + 800},
    "NIST": {"present": 517 + 3840, "absent": 3096 + 7644},
    "Standard": {"present": 83, "absent": 313}
}

fragment_neg_data = {
    "GNPS": {"present": 561, "absent": 9777 - 561},
    "MONA": {"present": 260, "absent": 1052 - 260},
    "NIST": {"present": 896, "absent": 3613 - 896},
    "Standard": {"present": 102, "absent": 396 - 102}
}

nl_or_fragment_neg_data = {
    "GNPS": {"present": 935, "absent": 9777 - 935},
    "MONA": {"present": 290, "absent": 1052 - 290},
    "NIST": {"present": 1151, "absent": 3613 - 1151},
    "Standard": {"present": 129, "absent": 396 - 129}
}

# color setting
colors_list = [
    ["#ffb3e6", "#99ccff"],
    ["#ffcc99", "#99ccff"],
    ["#ff9999", "#66b3ff"]
]

data_groups = [
    ("Neutral_loss_total", neutral_loss_data, colors_list[0]),
    ("Fragment_neg", fragment_neg_data, colors_list[1]),
    ("NL_or_Fragment_neg", nl_or_fragment_neg_data, colors_list[2])
]

for group_name, dataset, colors in data_groups:
    print(f"\n=== {group_name} ===")
    for db, counts in dataset.items():
        present = counts["present"]
        absent = counts["absent"]
        total = present + absent
        present_pct = present / total * 100
        absent_pct = absent / total * 100
        print(f"{db}: present={present} ({present_pct:.2f}%), absent={absent} ({absent_pct:.2f}%)")
        # plotting
        sizes = [present, absent]
        explode = (0.1, 0)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            sizes,
            labels=None,
            autopct=None,
            startangle=90,
            colors=colors,
            explode=explode,
            shadow=True
        )
        plt.tight_layout()
        fig.savefig(f"{db}_{group_name}_pie.png", dpi=300, transparent=True)
        plt.close(fig)
