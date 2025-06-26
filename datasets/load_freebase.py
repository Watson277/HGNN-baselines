# datasets/load_freebase.py
from torch_geometric.datasets import HGBDataset
import torch

def load_freebase():
    dataset = HGBDataset(root='/tmp/HGB', name='freebase')
    data = dataset[0]
    # 为venue节点添加one-hot编码特征0
    num_venues = 20
    data['venue'].x = torch.eye(num_venues)
    return data

def add_node_features(data, feature_dim=128):
    for node_type in data.node_types:
        if 'x' not in data[node_type]:
            num_nodes = data[node_type].num_nodes
            data[node_type].x = torch.randn(num_nodes, feature_dim).float()
    return data

