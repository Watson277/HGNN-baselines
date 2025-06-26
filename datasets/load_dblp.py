# datasets/load_dblp.py
from torch_geometric.datasets import HGBDataset
import torch

def load_dblp():
    dataset = HGBDataset(root='/tmp/HGB', name='DBLP')
    data = dataset[0]
    # 为venue节点添加one-hot编码特征0
    num_venues = 20
    data['venue'].x = torch.eye(num_venues)
    return data
