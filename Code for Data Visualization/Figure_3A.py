import matplotlib.pyplot as plt
import numpy as np

datasets = ["GNPS_PFAS","GNPS_non_PFAS","NIST_PFAS","NIST_non_PFAS",
            "MONA_PFAS","MONA_non_PFAS","Standard"]

models = ["MS1-CaPFAS","MS/MS-CaPFAS","Mass_difference-CaPFAS","Multimodal"]

performance = np.array([
    [0.687, 0.870, 0.654, 0.917],
    [0.596, 0.905, 0.919, 0.904],
    [0.805, 0.842, 0.741, 0.914],
    [0.576, 0.8651,0.935,0.904],
    [0.939, 0.825,0.597,0.904],
    [0.622, 0.925,0.897,0.953],
    [0.911, 0.719,0.441,0.870]
])

colors = ["#9fb7d6", "#accde1", "#f2b897", "#ff8e8e"]
x = np.arange(len(datasets))
width = 0.18

plt.figure(figsize=(10,6))
ax = plt.gca()

# 绘制条形图
for i, model in enumerate(models):
    ax.bar(x + i*width - width*(len(models)-1)/2, performance[:, i], width, color=colors[i])

# 隐藏轴名称
ax.set_xlabel('')
ax.set_ylabel('')
ax.xaxis.label.set_visible(False)
ax.yaxis.label.set_visible(False)

# 保留刻度线，但隐藏刻度文字
ax.set_xticks(x)
ax.set_xticklabels([])
ax.set_yticklabels([])

# Y轴范围
ax.set_ylim(0.4, 1.0)

# 网格
ax.grid(axis='y', linestyle='--', alpha=0.6)

# 隐藏右边和上边框
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 透明背景
plt.gcf().patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

plt.tight_layout()
plt.show()
