import subprocess
import os

# 设置编号（0~24）
choice = 1  # 例如 hgt + aminer（3 * 5 + 2）

# 模型和数据集列表（顺序要对）
models = ['gcn', 'gat', 'm2v', 'hgt', 'han']
datasets = ['acm', 'freebase', 'dblp', 'aminer', 'yelp']

# 构造路径映射
file_map = []
for dataset in datasets:
    for model in models:
        filepath = os.path.join(f"_{dataset}", f"train_{model}_{dataset}.py")
        file_map.append(filepath)

# 可选：打印映射关系（调试用）
for idx, path in enumerate(file_map):
    print(f"{idx:02d}: {path}")

# 运行对应脚本
if 0 <= choice < len(file_map):
    script_path = file_map[choice]
    print(f"运行脚本: {script_path}")
    subprocess.run(["python", script_path])
else:
    print("无效编号，请选择 0~24 的整数")
