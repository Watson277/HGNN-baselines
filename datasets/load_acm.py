# datasets/load_acm.py
from torch_geometric.datasets import HGBDataset
import torch

def load_acm():
    dataset = HGBDataset(root='/tmp/HGB', name='ACM')
    data = dataset[0]
    # 获取 term 节点数量
    num_terms = data['term'].num_nodes
    # 为 term 添加 one-hot 特征
    data['term'].x = torch.eye(num_terms)
    return data


