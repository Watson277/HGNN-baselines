# datasets/load_acm.py

from torch_geometric.datasets import HGBDataset

def load_acm():
    dataset = HGBDataset(root='/tmp/HGB', name='ACM')
    data = dataset[0]
    return data



