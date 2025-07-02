# freebase_to_homo.py
import torch
from torch.nn import Linear
from torch_geometric.datasets import HGBDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def convert_freebase_to_homogeneous(proj_dim=128):
    dataset = HGBDataset(root='/tmp/HGB', name='Freebase')
    data = dataset[0]

    # 1. 给没有特征的节点添加随机初始化的特征
    for node_type in data.node_types:
        if 'x' not in data[node_type]:
            num_nodes = data[node_type].num_nodes
            data[node_type].x = torch.randn(num_nodes, proj_dim)

    # 2. 每类节点一个投影层
    proj_layers = {
        node_type: Linear(data[node_type].num_features, proj_dim, bias=False)
        for node_type in data.node_types
    }

    # 3. 统一特征并计算 offset
    x_all = []
    node_id_offset = {}
    start = 0

    for node_type in data.node_types:
        x = proj_layers[node_type](data[node_type].x).detach()
        x_all.append(x)
        node_id_offset[node_type] = start
        start += data[node_type].num_nodes

    x_all = torch.cat(x_all, dim=0)

    # 4. 合并所有边，添加 offset
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

    # 5. 构造同构图 Data 对象，保留 book 的标签
    book_offset = node_id_offset['book']
    num_book = data['book'].num_nodes
    book_idx = torch.arange(num_book) + book_offset

    new_data = Data(
        x=x_all,
        edge_index=edge_index_all,
        y=data['book'].y,
        train_mask=data['book'].train_mask,
        test_mask=data['book'].test_mask,
        book_idx=book_idx
    )

    return new_data
