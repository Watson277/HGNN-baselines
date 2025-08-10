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
        [0.8789, 0.8606, 0.9542],
        [0.8839, 0.8846, 0.9752],
        [0.8795, 0.8770, 0.9582],
        [0.9010, 0.8986, 0.9794],
    ],
    'HERO': [
        [0.9086, 0.9075, 0.9712],
        [0.9123, 0.9124, 0.9833],
        [0.9096, 0.9084, 0.9753],
        [0.9223, 0.9253, 0.9881],
    ],
    'HGMAE': [
        [0.9039, 0.9006, 0.9792],
        [0.9099, 0.9029, 0.9823],
        [0.9119, 0.9089, 0.9828],
        [0.9138, 0.9094, 0.9846],
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
ax.set_ylim(0.85, 1.0)
ax.set_title('ACM Dataset Results')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("./result/graph/acm_bar.svg", format="svg")
plt.show()



