import matplotlib.pyplot as plt
import numpy as np

# 数据
x2 = [0.4721, 0.7087]
y3 = [0.6833, 0.7365]
y2 = [0.5178, 0.5184]
y1 = [0.4841, 0.4915]

# 转换为 numpy 方便操作
x = np.arange(len(x2))  # [0, 1]
bar_width = 0.25

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制柱状图
plt.bar(x - bar_width, y1, width=bar_width, label='F1-Micro', color='skyblue')
plt.bar(x,             y2, width=bar_width, label='F1-macro', color='lightgreen')
plt.bar(x + bar_width, y3, width=bar_width, label='AUC', color='salmon')

# 添加标签、刻度、图例
plt.xticks(x, [f'{v:.4f}' for v in x2])  # 显示 x2 的值作为 x 轴标签
plt.xlabel('Homophily')
plt.ylabel('Values')
plt.title("IMDP")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 可选：在柱子顶部标出数值
def add_labels(values, xpos, width):
    for xi, yi in zip(xpos, values):
        plt.text(xi, yi + 0.005, f'{yi:.4f}', ha='center', fontsize=8)

add_labels(y1, x - bar_width, bar_width)
add_labels(y2, x, bar_width)
add_labels(y3, x + bar_width, bar_width)

# 保存和显示
plt.tight_layout()
plt.savefig("./result/graph/bar_chart.svg", dpi=300, bbox_inches='tight')
plt.show()
