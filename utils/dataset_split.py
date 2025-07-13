from torch_geometric.utils import index_to_mask
import torch

def split_paper_nodes_by_class(data, target_node_type, train_per_class=20, val_per_class=30, seed=42, ):
    """
    为 ACM 数据集的 'paper' 节点生成 train/val/test mask。
    每个类别划分固定数量的训练和验证节点，其余为测试节点。
    """
    torch.manual_seed(seed)
    y = data[target_node_type].y
    num_classes = int(y.max().item()) + 1
    indices = []

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))]  # shuffle
        indices.append(idx)

    train_idx = torch.cat([i[:train_per_class] for i in indices])
    val_idx = torch.cat([i[train_per_class:train_per_class+val_per_class] for i in indices])
    all_used = torch.cat([train_idx, val_idx])
    test_mask = torch.ones(y.size(0), dtype=torch.bool)
    test_mask[all_used] = False

    data[target_node_type].train_mask = index_to_mask(train_idx, size=y.size(0))
    data[target_node_type].val_mask = index_to_mask(val_idx, size=y.size(0))
    data[target_node_type].test_mask = test_mask

    print(f"划分完成：train={train_idx.size(0)}, val={val_idx.size(0)}, test={test_mask.sum().item()}")