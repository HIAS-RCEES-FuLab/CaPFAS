import matplotlib.pyplot as plt

# Site 1
#values = [471, 37]
# Site 2
values = [142, 16]

explode = [0.04, 0.04]

colors = ['#A1C1DF', '#DFA47D']

plt.figure(figsize=(6,6))
ax = plt.gca()

ax.pie(
    values,
    startangle=90,
    colors=colors,
    explode=explode,
    shadow=True,
    labels=None,
    autopct=None
)

plt.axis('equal')

plt.tight_layout()
plt.savefig("pie_chart_IDresult.png", dpi=300, transparent=True)

plt.show()