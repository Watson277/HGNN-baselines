import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import IMDB


def load_imdb():
    metapaths =[[('movie','actor'),('actor','movie')],
                [('movie','director'),('director','movie')]]
    transform =T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True, drop_unconnected_node_types=True)
    dataset =IMDB(root='/tmp/HGB',transform=transform)
    data = dataset[0]
    return data

if __name__ == "__main__":
    data = load_imdb()
    print(data)