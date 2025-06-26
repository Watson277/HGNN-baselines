# dblp_to_homo.py
import torch
from torch.nn import Linear
from torch_geometric.datasets import HGBDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def convert_dblp_to_homogeneous(proj_dim=128):
    dataset = HGBDataset(root='/tmp/HGB', name='DBLP')
    data = dataset[0]

    # 如果某些节点没有特征（如 venue），构造 one-hot 或随机特征
    for node_type in data.node_types:
        if 'x' not in data[node_type]:
            num_nodes = data[node_type].num_nodes
            data[node_type].x = torch.eye(num_nodes) if num_nodes < proj_dim else torch.randn(num_nodes, proj_dim)

    # 1. 初始化每种节点的线性映射层
    proj_layers = {
        node_type: Linear(data[node_type].num_features, proj_dim, bias=False)
        for node_type in data.node_types
    }

    # 2. 投影所有节点特征，并记录全局编号偏移
    x_all = []
    node_id_offset = {}
    start = 0

    for node_type in data.node_types:
        x = proj_layers[node_type](data[node_type].x).detach()
        x_all.append(x)
        node_id_offset[node_type] = start
        start += data[node_type].num_nodes

    x_all = torch.cat(x_all, dim=0)

    # 3. 构造同构图的 edge_index
    edge_index_all = []

    for (src_type, _, dst_type), edge_index in data.edge_index_dict.items():
        src_offset = node_id_offset[src_type]
        dst_offset = node_id_offset[dst_type]

        edge_index = edge_index.clone()
        edge_index[0] += src_offset
        edge_index[1] += dst_offset
        edge_index_all.append(edge_index)

    edge_index_all = torch.cat(edge_index_all, dim=1)
    edge_index_all = to_undirected(edge_index_all)

    # 4. 构建 Data 对象，只保留 author 节点的标签和 mask
    author_offset = node_id_offset['author']
    num_author = data['author'].num_nodes
    author_idx = torch.arange(num_author) + author_offset

    new_data = Data(
        x=x_all,
        edge_index=edge_index_all,
        y=data['author'].y,
        train_mask=data['author'].train_mask,
        test_mask=data['author'].test_mask,
        author_idx=author_idx
    )

    return new_data
