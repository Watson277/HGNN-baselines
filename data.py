import torch
from torch_geometric.datasets import HGBDataset

# 加载 ACM 数据集
dataset = HGBDataset(root='/tmp/HGB', name='ACM')
data = dataset[0]
if 'term' not in data.x_dict:
    data['term'].x = torch.randn(data['term'].num_nodes, 1902).float()
print(data.metadata())