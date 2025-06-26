# acm_to_homo.py
import torch
from torch.nn import Linear
from torch_geometric.datasets import HGBDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def convert_acm_to_homogeneous(proj_dim=128):
    dataset = HGBDataset(root='/tmp/HGB', name='ACM')
    data = dataset[0]

    if 'term' not in data.x_dict:
        data['term'].x = torch.randn(data['term'].num_nodes, 1902).float()

    # 1. 初始化投影层
    proj_layers = {
        node_type: Linear(data[node_type].num_features, proj_dim, bias=False)
        for node_type in data.node_types
    }

    # 2. 投影特征 + 记录偏移
    x_all = []
    node_id_offset = {}
    start = 0

    for node_type in data.node_types:
        x = proj_layers[node_type](data[node_type].x).detach() 
        x_all.append(x)
        node_id_offset[node_type] = start
        start += data[node_type].num_nodes

    x_all = torch.cat(x_all, dim=0)

    # 3. 合并所有边
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

    # 4. 创建新的同构图
    paper_offset = node_id_offset['paper']
    num_paper = data['paper'].num_nodes
    paper_idx = torch.arange(num_paper) + paper_offset

    new_data = Data(
        x=x_all,
        edge_index=edge_index_all,
        y=data['paper'].y,
        train_mask=data['paper'].train_mask,
        test_mask=data['paper'].test_mask,
        paper_idx=paper_idx
    )

    return new_data
