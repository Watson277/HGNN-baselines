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



import torch.nn as nn
import torch.nn.functional as F

class SimilarityMLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, src_feat, dst_feat):
        h = torch.cat([src_feat, dst_feat], dim=1)
        h = F.relu(self.fc1(h))
        score = torch.sigmoid(self.fc2(h))  # ∈ [0, 1]
        return score.squeeze(-1)



from torch_geometric.nn import HGTConv
from torch.nn import Linear
from torch_geometric.data import HeteroData

class HGTWithEdgeWeight(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata, num_heads):
        super().__init__()
        self.metadata = metadata
        self.lin_dict = torch.nn.ModuleDict()
        self.sim_mlp_dict = torch.nn.ModuleDict()  # 每种边一个 MLP

        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(in_channels, hidden_channels)

        for edge_type in metadata[1]:
            self.sim_mlp_dict[str(edge_type)] = SimilarityMLP(hidden_channels)

        self.conv1 = HGTConv(hidden_channels, hidden_channels, metadata, heads=num_heads)
        self.conv2 = HGTConv(hidden_channels, out_channels, metadata, heads=num_heads)

    def forward(self, x_dict, edge_index_dict, thresh=0.44):
        # 1. 映射节点特征
        x_dict = {k: self.lin_dict[k](x) for k, x in x_dict.items()}

        # 2. 删除边：筛选出高相似度边
        new_edge_index_dict = {}
        num_edges_removed = {}

        for edge_type in edge_index_dict:
            src_type, _, dst_type = edge_type
            edge_index = edge_index_dict[edge_type]  # [2, num_edges]
            src_x = x_dict[src_type][edge_index[0]]
            dst_x = x_dict[dst_type][edge_index[1]]

            sim_score = self.sim_mlp_dict[str(edge_type)](src_x, dst_x)  # [num_edges]
            keep_mask = sim_score >= thresh
            new_edge_index = edge_index[:, keep_mask]

            new_edge_index_dict[edge_type] = new_edge_index
            num_edges_removed[edge_type] = edge_index.size(1) - new_edge_index.size(1)

        # 3. 打印每种边删除数量
        for etype, num_removed in num_edges_removed.items():
            print(f"{etype} removed {num_removed} edges.")

        # 4. HGTConv 正常传播
        x_dict = self.conv1(x_dict, new_edge_index_dict)
        x_dict = self.conv2(x_dict, new_edge_index_dict)
        return x_dict




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch.nn import Linear


class SimilarityMLP2(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, src_feat, dst_feat):
        h = torch.cat([src_feat, dst_feat], dim=1)
        h = F.relu(self.fc1(h))
        score = torch.sigmoid(self.fc2(h))  # ∈ [0, 1]
        return score.squeeze(-1)


class HGTWithEdgeWeight2(nn.Module):
    def __init__(self, in_dims: dict, hidden_channels: int, out_channels: int, metadata, num_heads: int):
        """
        in_dims: dict，表示每种节点类型的输入特征维度，如 {'author': 256, 'paper': 100}
        """
        super().__init__()
        self.metadata = metadata
        self.hidden_channels = hidden_channels

        # 节点特征变换模块：不同类型节点维度不同
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(in_dims[node_type], hidden_channels)

        # 相似度 MLP 模块：每种边一个
        self.sim_mlp_dict = nn.ModuleDict()
        for edge_type in metadata[1]:
            self.sim_mlp_dict[str(edge_type)] = SimilarityMLP2(hidden_channels)

        # HGT 层
        self.conv1 = HGTConv(hidden_channels, hidden_channels, metadata, heads=num_heads)
        self.conv2 = HGTConv(hidden_channels, out_channels, metadata, heads=num_heads)

    def forward(self, x_dict, edge_index_dict, thresh=0.44):
        # 1. 统一映射节点特征
        x_dict = {k: self.lin_dict[k](x) for k, x in x_dict.items()}

        # 2. 使用 MLP 判断边是否保留
        new_edge_index_dict = {}
        num_edges_removed = {}

        for edge_type in edge_index_dict:
            src_type, _, dst_type = edge_type
            edge_index = edge_index_dict[edge_type]  # [2, num_edges]
            src_x = x_dict[src_type][edge_index[0]]
            dst_x = x_dict[dst_type][edge_index[1]]

            sim_score = self.sim_mlp_dict[str(edge_type)](src_x, dst_x)  # [num_edges]
            keep_mask = sim_score >= thresh
            new_edge_index = edge_index[:, keep_mask]

            new_edge_index_dict[edge_type] = new_edge_index
            num_edges_removed[edge_type] = edge_index.size(1) - new_edge_index.size(1)

        # 3. 可选：打印被删除的边数量
        for etype, num_removed in num_edges_removed.items():
            print(f"{etype} removed {num_removed} edges.")

        # 4. HGTConv 两层传播
        x_dict = self.conv1(x_dict, new_edge_index_dict)
        x_dict = self.conv2(x_dict, new_edge_index_dict)
        return x_dict


