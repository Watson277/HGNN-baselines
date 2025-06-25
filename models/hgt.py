import torch
from torch.nn import Linear
from torch_geometric.nn import HGTConv

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

