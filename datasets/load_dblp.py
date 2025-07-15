# datasets/load_dblp.py
from torch_geometric.datasets import HGBDataset
import torch
import random
from torch_geometric.data import HeteroData

node_type = 'author'

def load_dblp():
    dataset = HGBDataset(root='/tmp/HGB', name='DBLP')
    data = dataset[0]
    # 为venue节点添加one-hot编码特征0
    num_venues = 20
    data['venue'].x = torch.eye(num_venues)
    return data


def sample_train_mask_for_target_class(data: HeteroData, num_train_per_class=20, node_type='author'):
    y = data[node_type].y
    num_classes = int(y.max().item()) + 1
    train_idx = []

    for c in range(num_classes):
        # 找到所有属于类别c的节点索引
        class_idx = (y == c).nonzero(as_tuple=True)[0]
        if len(class_idx) < num_train_per_class:
            print(f"[Warning] Class {c} has only {len(class_idx)} nodes, using all.")
            sampled = class_idx
        else:
            perm = torch.randperm(len(class_idx))
            sampled = class_idx[perm[:num_train_per_class]]
        train_idx.append(sampled)

    # 合并所有采样的训练节点索引
    train_idx = torch.cat(train_idx, dim=0)

    # 创建新的 train_mask，其他全为 False
    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    train_mask[train_idx] = True

    data[node_type].train_mask = train_mask
    print(f"[INFO] Set {train_mask.sum().item()} training nodes for '{node_type}'")
    return data


if __name__ == '__main__':
    data = load_dblp()
    data = sample_train_mask_for_target_class(data, num_train_per_class=10, node_type='author')
    print("Train mask sum:", data['author'].train_mask.sum().item())  # 应该是 类别数 × 10

    print(data)