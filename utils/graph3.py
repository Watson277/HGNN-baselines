

eta = [0.1, 0.3, 0.4, 0.5, 0.6, 0.8]
theta = [0.1, 0.3, 0.4, 0.5, 0.6, 0.8]

# HeCo-ACM
# Ma_F1_1 = [0.883, 0.885, 0.889, 0.888, 0.8942, 0.889]
# Ma_F1_2 = [0.875, 0.8752, 0.8943, 0.8942, 0.901, 0.8802]
# Ma_F1_3 = [0.855, 0.865, 0.867, 0.877, 0.8642, 0.868]

# HeCo-DBLP
# Ma_F1_1 = [0.883, 0.885, 0.891, 0.902, 0.912, 0.902]
# Ma_F1_2 = [0.875, 0.885, 0.894, 0.9042, 0.9138, 0.9032]
# Ma_F1_3 = [0.85, 0.86, 0.867, 0.87, 0.907, 0.868]

# HERO-ACM
# Ma_F1_1 = [0.883, 0.885, 0.889, 0.888, 0.8942, 0.889]
# Ma_F1_2 = [0.882, 0.8952, 0.9043, 0.9042, 0.9123, 0.9002]
# Ma_F1_3 = [0.835, 0.835, 0.847, 0.877, 0.8742, 0.873]

# HERO-DBLP
Ma_F1_1 = [0.883, 0.885, 0.889, 0.89, 0.9042, 0.889]
Ma_F1_2 = [0.875, 0.9052, 0.9243, 0.9342, 0.9348, 0.9102]
Ma_F1_3 = [0.855, 0.865, 0.867, 0.877, 0.8942, 0.89]


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

# 画出三条线
plt.plot(eta, Ma_F1_1, marker='o', label='K=5', linewidth=3)
plt.plot(eta, Ma_F1_2, marker='s', label='K=10', linewidth=3)
plt.plot(eta, Ma_F1_3, marker='^', label='K=20', linewidth=3)

# 添加标签和标题
plt.xlabel('Eta', fontsize=12)
plt.ylabel('Macro-F1', fontsize=12)
plt.title('HERO-DBLP', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("./result/graph/HERO-DBLP.svg", dpi=300, bbox_inches='tight')# 显示图像
plt.show()
