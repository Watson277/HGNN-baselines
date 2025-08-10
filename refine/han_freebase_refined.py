import sys
import os
# 获取当前文件的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


import os.path as osp
from typing import Dict, List, Union
import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import HGBDataset
from torch_geometric.nn import HANConv
from utils.dataset_split import split_paper_nodes_by_class
from datasets.load_freebase import  add_node_features
from refine.han_refine import prune_edges_by_cosine_similarity


# ✅ 加载 ACM 数据集

metapaths = [
    [('book', 'and', 'book'), ('book', 'and', 'book')]
]
target_node_type = 'book'

print(metapaths)

# ✅ 使用 AddMetaPaths 创建新边类型（paper -> paper）
transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=False)
dataset = HGBDataset(root='/tmp/HGB', name='freebase', transform=transform)
data = dataset[0]
data = add_node_features(data)

split_paper_nodes_by_class(data, train_per_class=20, val_per_class=30, target_node_type=target_node_type)

num_classes = int(data[target_node_type].y.max()) + 1

# 同构图结构优化
metapath_names = [f'metapath_{i}' for i in range(1)]
prune_edges_by_cosine_similarity(data, metapath_names, node_type=target_node_type, threshold=0.2)

print(data)



class HAN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=64, heads=8):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.6, metadata=data.metadata())
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out[target_node_type])
        return out


model = HAN(in_channels=-1, out_channels=num_classes)

# ✅ 设备选择
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

data, model = data.to(device), model.to(device)

# ✅ Lazy init
with torch.no_grad():
    out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)


def train() -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data[target_node_type].train_mask
    loss = F.cross_entropy(out[mask], data[target_node_type].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from typing import List, Tuple

@torch.no_grad()
def test() -> Tuple[List[float], List[float]]:
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)  # shape: [num_nodes, num_classes]
    pred = out.argmax(dim=-1)
    y_true = data[target_node_type].y

    accs, aucs = [], []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data[target_node_type][split]
        acc = (pred[mask] == y_true[mask]).sum() / mask.sum()
        accs.append(float(acc))

        # 多分类 AUC，使用 one-hot 和 softmax 概率
        try:
            probs = F.softmax(out[mask], dim=1)
            y_onehot = F.one_hot(y_true[mask], num_classes=out.size(-1)).cpu()
            auc = roc_auc_score(y_onehot, probs.cpu(), average='macro', multi_class='ovo')
        except ValueError:
            auc = 0.0  # 如果某个 split 中类别不全（例如 val/test 集中某一类）
        aucs.append(auc)
    
    return accs, aucs  # ([train_acc, val_acc, test_acc], [train_auc, val_auc, test_auc])


best_val_acc = 0
start_patience = patience = 100
for epoch in range(1, 201):
    loss = train()
    accs, aucs = test()
    train_acc, val_acc, test_acc = accs
    train_auc, val_auc, test_auc = aucs

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, '
          f'Test AUC: {test_auc:.4f}')

    if best_val_acc <= val_acc:
        best_val_acc = val_acc
        patience = start_patience
    else:
        patience -= 1

    if patience <= 0:
        print(f"Early stopping at epoch {epoch}")
        break

