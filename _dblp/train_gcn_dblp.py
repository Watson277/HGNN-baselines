# train_gcn_on_dblp.py
import sys
import os
# 获取当前文件的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from models.gcn import GCN
from datasets.dblp_to_homo import convert_dblp_to_homogeneous
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 1. 加载数据 + 提前“冷冻”索引/标签张量 ===
data = convert_dblp_to_homogeneous()
print(data)

author_idx = data.author_idx.clone().detach().cpu()
train_mask = data.train_mask.clone().detach().cpu()
test_mask = data.test_mask.clone().detach().cpu()
y = data.y.clone().detach().cpu()

data = data.to(device)
author_idx = author_idx.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)
y = y.to(device)

# === 2. 初始化模型 ===
num_classes = int(y.max().item()) + 1
model = GCN(128, 64, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# === 3. 训练 / 测试 ===
def train():
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    out_author = out[author_idx]
    loss = F.cross_entropy(out_author[train_mask], y[train_mask])

    loss.backward()
    optimizer.step()

    pred = out_author.argmax(dim=1)
    train_acc = (pred[train_mask] == y[train_mask]).float().mean()

    return loss.item(), train_acc.item()


from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)  # [num_all_nodes, num_classes]
    out_author = out[author_idx]          # 只取 author 节点
    pred = out_author.argmax(dim=1)

    # Test set
    y_true = y[test_mask].cpu()
    y_pred = pred[test_mask].cpu()
    logits = out_author[test_mask].cpu()  # 获取 logits 做 softmax

    acc = (y_pred == y_true).float().mean().item()
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # AUC
    num_classes = logits.size(1)
    y_prob = F.softmax(logits, dim=1)
    y_true_onehot = F.one_hot(y_true, num_classes=num_classes)

    try:
        auc = roc_auc_score(y_true_onehot, y_prob, average='macro', multi_class='ovr')
    except ValueError:
        auc = float('nan')

    return acc, f1_micro, f1_macro, auc


for epoch in range(1, 201):
    loss, train_acc = train()
    test_acc, test_f1_micro, test_f1_macro, test_auc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Acc: {test_acc:.4f}, Test F1-Mi: {test_f1_micro:.4f}, "
          f"Test F1-Ma: {test_f1_macro:.4f}, Test AUC: {test_auc:.4f}")


