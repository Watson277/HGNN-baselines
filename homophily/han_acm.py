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


# ✅ 加载 ACM 数据集
metapaths = [
    # [('paper', 'cite', 'paper'), ('paper', 'cite', 'paper')],
    # [('paper', 'cite', 'paper'), ('paper', 'ref', 'paper')],
    # [('paper', 'ref', 'paper'), ('paper', 'cite', 'paper')],
    # [('paper', 'ref', 'paper'), ('paper', 'ref', 'paper')],
    # [('paper', 'to', 'author'), ('author', 'to', 'paper')],
    [('paper', 'to', 'subject'), ('subject', 'to', 'paper')]
]
target_node_type = 'paper'

print(metapaths)

# ✅ 使用 AddMetaPaths 创建新边类型（paper -> paper）
transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=False)
dataset = HGBDataset(root='/tmp/HGB', name='ACM', transform=transform)
data = dataset[0]
split_paper_nodes_by_class(data, train_per_class=200, val_per_class=30, target_node_type=target_node_type)
print(data)

# ✅ 特征补全（ACM 数据集中 term 节点无 x）
for node_type in data.node_types:
    if 'x' not in data[node_type]:
        in_dim = 1902 if node_type == 'term' else 128
        data[node_type].x = torch.randn(data[node_type].num_nodes, in_dim).float()

# ✅ 获取分类任务的类别数
num_classes = int(data[target_node_type].y.max()) + 1


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


from typing import Tuple, List
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F

@torch.no_grad()
def test() -> Tuple[List[float], List[float], List[float], List[float]]:
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)  # shape: [num_nodes, num_classes]
    pred = out.argmax(dim=-1)
    y_true = data[target_node_type].y

    accs, aucs, f1_micros, f1_macros = [], [], [], []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data[target_node_type][split]

        acc = (pred[mask] == y_true[mask]).sum() / mask.sum()
        accs.append(float(acc))

        try:
            probs = F.softmax(out[mask], dim=1)
            y_onehot = F.one_hot(y_true[mask], num_classes=out.size(-1)).cpu()
            auc = roc_auc_score(y_onehot, probs.cpu(), average='macro', multi_class='ovo')
        except ValueError:
            auc = 0.0
        aucs.append(auc)

        # F1-micro & F1-macro
        f1_micro = f1_score(y_true[mask].cpu(), pred[mask].cpu(), average='micro', zero_division=0)
        f1_macro = f1_score(y_true[mask].cpu(), pred[mask].cpu(), average='macro', zero_division=0)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)

    return accs, aucs, f1_micros, f1_macros  # 分别为 [train, val, test] 上的四个指标列表



best_val_acc = 0
start_patience = patience = 100

for epoch in range(1, 101):
    loss = train()
    accs, aucs, f1_micros, f1_macros = test()
    train_acc, val_acc, test_acc = accs
    train_auc, val_auc, test_auc = aucs
    train_f1_micro, val_f1_micro, test_f1_micro = f1_micros
    train_f1_macro, val_f1_macro, test_f1_macro = f1_macros

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, '
          f'Test AUC: {test_auc:.4f}, '
          f'Test F1 Micro: {test_f1_micro:.4f}, Test F1 Macro: {test_f1_macro:.4f}')

    if best_val_acc <= val_acc:
        best_val_acc = val_acc
        patience = start_patience
    else:
        patience -= 1

    if patience <= 0:
        print(f"Early stopping at epoch {epoch}")
        break


