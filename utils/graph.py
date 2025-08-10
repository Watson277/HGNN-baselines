import matplotlib.pyplot as plt

# 数据
x1 = [0.8003, 0.8130, 0.8179, 0.8429, 0.8460, 0.8766]
y1 = [0.9154, 0.9175, 0.9186, 0.9192, 0.9197, 0.9233]
y2 = [0.9119, 0.9183, 0.9212, 0.9225, 0.9238, 0.9259]
y3 = [0.9101, 0.9206, 0.9219, 0.9236, 0.9252, 0.9271]

# x2 = [0.4721, 0.7087]
# y1 = [0.6833, 0.7365]
# y2 = [0.5178, 0.5184]
# y3 = [0.4841, 0.4915] 


# 创建图形
plt.figure(figsize=(8, 6))

# 绘制三条曲线
plt.plot(x1, y1, marker='o', linestyle='-', label='AUC', color='blue')
plt.plot(x1, y2, marker='s', linestyle='-', label='F1-Micro', color='green')
plt.plot(x1, y3, marker='^', linestyle='-', label='F1-Macro', color='red')

# 设置标题和坐标轴
plt.title("ACM")
plt.xlabel("Homophily")
plt.ylabel("Values")

# 添加图例和网格
plt.legend()
plt.grid(True)

# 保存图像
plt.tight_layout()
plt.savefig("./result/graph/three_curves_plot.svg", dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
