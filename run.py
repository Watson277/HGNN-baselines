import subprocess
import time

# 设置路径列表文件和运行次数
txt_file = "./result/files2.txt"  # 你保存路径列表的文件
n = 20  # 每个脚本运行 n 遍

# 读取 txt 文件中的脚本路径
with open(txt_file, 'r', encoding='utf-8') as f:
    script_paths = [line.strip() for line in f if line.strip()]

# 依次运行每个脚本 n 遍
for script in script_paths:
    for i in range(n):
        print(f"\n>>> 正在运行：{script}（第 {i+1}/{n} 次）")
        try:
            subprocess.run(["python", script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 运行出错：{script}，错误信息：{e}")

