import matplotlib.pyplot as plt
import numpy as np

# 设置字体（如果中文或渲染异常可以去掉）
plt.rcParams['font.family'] = 'Arial'

# 横轴标签
metrics = ['F1-Macro', 'F1-Micro', 'AUC']
x = np.arange(len(metrics))  # [0, 1, 2]
width = 0.22  # 每组宽度

# 方法颜色和标签
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
labels = ['Origin', 'w/o HoSE', 'w/o HeED', 'All']

# 数据字典：每个模型下的4种方法
data = {
    'HeCo': [
        [0.5784, 0.6012, 0.7500],
        [0.5938, 0.6247, 0.7702],
        [0.5861, 0.6109, 0.7721],
        [0.5990, 0.6294, 0.7785],
    ],
    'HERO': [
        [0.6122, 0.6523, 0.7902],
        [0.6265, 0.6586, 0.8021],
        [0.6186, 0.6542, 0.7932],
        [0.6342, 0.6623, 0.8085],
    ],
    'HGMAE': [
        [0.6142, 0.6461, 0.7819],
        [0.6264, 0.6575, 0.7933],
        [0.6296, 0.6589, 0.7984],
        [0.6262, 0.6577, 0.7945],
    ]
}

fig, ax = plt.subplots(figsize=(16, 7))

# 绘制柱状图
width = 0.2   # 适当加宽，视觉更饱满
gap = 0.0

for i, (model_name, method_scores) in enumerate(data.items()):
    base = x + i * (len(metrics) + 0.8)  # 控制每组模型之间的间距
    for j in range(4):  # 4 种方法
        offset = j * width  # 不加间距
        ax.bar(base + offset, method_scores[j], width,
               label=labels[j] if i == 0 else "", color=colors[j])



# 设置坐标轴标签
xticks = []
xticklabels = []
for i, model in enumerate(data.keys()):
    for j, metric in enumerate(metrics):
        xticks.append(x[j] + i * (len(metrics) + 1) + 1.5 * width)
        xticklabels.append(f'{model}\n{metric}')

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

# 设置图例在上方
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.10))

# 其他样式
ax.set_ylabel('Score')
ax.set_ylim(0.5, 0.83)
ax.set_title('Freebase Dataset Results')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("./result/graph/freebase_bar.svg", format="svg")
plt.show()





