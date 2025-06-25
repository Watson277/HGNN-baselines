# datasets/load_dblp.py
import torch
from torch_geometric.datasets import AcademicGraphDataset
from torch_geometric.datasets import HGBDataset

def load_dblp(root='data/DBLP'):
    dataset = AcademicGraphDataset(root=root, name='dblp')
    data = dataset[0]
    return data

if __name__ == '__main__':
    dataset = HGBDataset(root='/tmp/HGB', name='DBLP')
    print(dataset)