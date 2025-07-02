# datasets/load_freebase.py
from torch_geometric.datasets import HGBDataset
import torch

def load_freebase():
    dataset = HGBDataset(root='/tmp/HGB', name='freebase')
    data = dataset[0]
    return data

def add_node_features(data, feature_dim=128):
    for node_type in data.node_types:
        if 'x' not in data[node_type]:
            num_nodes = data[node_type].num_nodes
            data[node_type].x = torch.randn(num_nodes, feature_dim).float()
    return data

