import torch
from torch.nn import Linear
from torch_geometric.nn import HGTConv
import torch.nn.functional as F

class HGT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata, num_heads):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)

        self.conv1 = HGTConv(hidden_channels, hidden_channels, metadata, heads=num_heads)
        self.conv2 = HGTConv(hidden_channels, out_channels, metadata, heads=num_heads)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: self.lin_dict[k](x) for k, x in x_dict.items()}
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

class HGT2(torch.nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, out_channels, metadata, num_heads=2):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type, in_dim in in_channels_dict.items():
            self.lin_dict[node_type] = Linear(in_dim, hidden_channels)

        self.conv1 = HGTConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=num_heads
        )
        self.conv2 = HGTConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            metadata=metadata,
            heads=num_heads
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            key: self.lin_dict[key](x) for key, x in x_dict.items()
        }
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


