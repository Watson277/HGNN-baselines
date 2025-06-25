# datasets/load_yelp.py
import torch
from torch_geometric.datasets import YelpDataset

def load_yelp(root='data/Yelp'):
    dataset = YelpDataset(root=root)
    data = dataset[0]
    return data
