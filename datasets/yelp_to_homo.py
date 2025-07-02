import torch
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def convert_yelp_to_homogeneous(path='./datasets/Yelp JSON/yelp.pt', proj_dim=64):
    # 加载异构图
    data = torch.load(path)

    # 1. 每类节点一个投影层
    proj_layers = {
        ntype: Linear(data[ntype].num_features, proj_dim, bias=False)
        for ntype in data.node_types
    }

    # 2. 特征拼接 & 节点 ID 偏移计算
    x_all = []
    node_id_offset = {}
    start = 0

    for ntype in data.node_types:
        x = proj_layers[ntype](data[ntype].x).detach()
        x_all.append(x)
        node_id_offset[ntype] = start
        start += data[ntype].num_nodes

    x_all = torch.cat(x_all, dim=0)

    # 3. 合并边（加上偏移），并转为无向图
    edge_index_all = []

    for (src, _, dst), edge_index in data.edge_index_dict.items():
        src_offset = node_id_offset[src]
        dst_offset = node_id_offset[dst]

        edge_index = edge_index.clone()
        edge_index[0] += src_offset
        edge_index[1] += dst_offset
        edge_index_all.append(edge_index)

    edge_index_all = torch.cat(edge_index_all, dim=1)
    edge_index_all = to_undirected(edge_index_all)

    # 4. 指定目标节点类型（如 business），保留其标签与 mask
    target_type = 'business'
    target_offset = node_id_offset[target_type]
    num_target = data[target_type].num_nodes
    business_idx = torch.arange(num_target) + target_offset

    new_data = Data(
        x=x_all,
        edge_index=edge_index_all,
        y=data[target_type].y,
        train_mask=data[target_type].train_mask,
        val_mask=data[target_type].val_mask,
        test_mask=data[target_type].test_mask,
        business_idx=business_idx
    )

    return new_data


if __name__ == '__main__':
    homo_data = convert_yelp_to_homogeneous()
    torch.save(homo_data, './datasets/Yelp JSON/yelp_homo.pt')
    print("✅ 成功保存同构图数据到 yelp_homo.pt")
    print(homo_data)
