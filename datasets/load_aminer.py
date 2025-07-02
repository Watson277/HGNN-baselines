from torch_geometric.datasets import AMiner
import torch


def load_aminer(hidden_dim=128, train_ratio=0.6, val_ratio=0.2):
    dataset = AMiner(root='/tmp/Aminer')
    data = dataset[0]  # HeteroData 对象

    # === 1. 初始化节点特征（随机） ===
    for node_type in data.node_types:
        if 'x' not in data[node_type] or data[node_type].x is None:
            num_nodes = data[node_type].num_nodes
            data[node_type].x = torch.randn(num_nodes, hidden_dim)

    # === 2. 为有标签的节点添加 mask（如 author、venue） ===
    for node_type in data.node_types:
        if 'y' in data[node_type] and 'y_index' in data[node_type]:
            y = data[node_type].y
            y_index = data[node_type].y_index
            num_total = y_index.size(0)

            # 随机划分 index
            perm = torch.randperm(num_total)
            num_train = int(train_ratio * num_total)
            num_val = int(val_ratio * num_total)
            num_test = num_total - num_train - num_val

            train_idx = y_index[perm[:num_train]]
            val_idx = y_index[perm[num_train:num_train + num_val]]
            test_idx = y_index[perm[num_train + num_val:]]

            mask_shape = (data[node_type].num_nodes,)
            data[node_type].train_mask = torch.zeros(mask_shape, dtype=torch.bool)
            data[node_type].val_mask = torch.zeros(mask_shape, dtype=torch.bool)
            data[node_type].test_mask = torch.zeros(mask_shape, dtype=torch.bool)

            data[node_type].train_mask[train_idx] = True
            data[node_type].val_mask[val_idx] = True
            data[node_type].test_mask[test_idx] = True

    print(data)
    return data

