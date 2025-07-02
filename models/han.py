import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import HANConv

import torch.nn.functional as F
from torch.nn import Linear

class HAN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata, heads=2, dropout=0.6):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)
        self.conv1 = HANConv(hidden_channels, hidden_channels, metadata, heads=heads, dropout=dropout)
        self.conv2 = HANConv(hidden_channels, out_channels, metadata, heads=1, dropout=dropout)

    def forward(self, x_dict, edge_index_dict):
        for k,v in x_dict.items():
            if v is None:
                print(f"Before lin_dict: x_dict[{k}] is None")
        x_dict = {k: F.elu(self.lin_dict[k](v)) for k, v in x_dict.items()}
        for k,v in x_dict.items():
            if v is None:
                print(f"After lin_dict: x_dict[{k}] is None")
        x_dict = self.conv1(x_dict, edge_index_dict)
        for k,v in x_dict.items():
            if v is None:
                print(f"After conv1: x_dict[{k}] is None")
        x_dict = self.conv2(x_dict, edge_index_dict)
        for k,v in x_dict.items():
            if v is None:
                print(f"After conv2: x_dict[{k}] is None")
        return x_dict


class HAN2(torch.nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, out_channels, metadata, heads=2, dropout=0.6):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            in_dim = in_channels_dict[node_type]
            self.lin_dict[node_type] = Linear(in_dim, hidden_channels)

        self.conv1 = HANConv(hidden_channels, hidden_channels, metadata, heads=heads, dropout=dropout)
        self.conv2 = HANConv(hidden_channels, out_channels, metadata, heads=1, dropout=dropout)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: F.elu(self.lin_dict[k](v)) for k, v in x_dict.items()}
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

