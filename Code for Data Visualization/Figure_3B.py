import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'

datasets = ['GNPS_PFAS', 'GNPS_non_PFAS', 'NIST_PFAS', 'NIST_non_PFAS', 'MONA_PFAS', 'MONA_non_PFAS', 'Standard']

models = ['CaPFAS','RF', 'XGBoost', 'SVM', 'LOG', 'KNN']

scores = np.array([
    [0.917, 0.856, 0.729, 0.801, 0.844, 0.729],  # GNPS_PFAS
    [0.904, 0.974, 0.932, 0.823, 0.846, 0.866],  # GNPS_non_PFAS
    [0.914, 0.863, 0.818, 0.880, 0.910, 0.804],  # NIST_PFAS
    [0.904, 0.977, 0.961, 0.847, 0.874, 0.901],  # NIST_non_PFAS
    [0.904, 0.814, 0.814, 0.860, 0.898, 0.797],  # MONA_PFAS
    [0.953, 1.000, 0.986, 0.879, 0.937, 0.899],  # MONA_non_PFAS
    [0.870, 0.691, 0.853, 0.671, 0.739, 0.623],  # Standard
])

scores = scores.T

num_datasets = len(datasets)
angles = np.linspace(0, 2 * np.pi, num_datasets, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': 'polar'})
ax.set_ylim(0, 1)

colors = ['#7FB3D5', '#F5B7B1', '#82E0AA', '#F7DC6F', '#F5CBA7', '#BB8FCE', '#76D7C4']

# 绘制每个模型
for i, model in enumerate(models):
    values = scores[i].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, alpha=0.8, color=colors[i], label=model)
    ax.fill(angles, values, color=colors[i], alpha=0.2)

ax.set_xticks(angles[:-1])
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.gcf().patch.set_alpha(0.0)
plt.gca().patch.set_alpha(0.0)

plt.tight_layout()
plt.savefig('model_dataset_radar_soft_colors.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
