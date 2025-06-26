import torch
from torch_geometric.datasets import HGBDataset

# 加载 ACM 数据集
dataset = HGBDataset(root='/tmp/HGB', name='DBLP')
data = dataset[0]
print(data)