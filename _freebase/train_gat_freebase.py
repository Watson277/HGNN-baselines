import sys
import os
# 获取当前文件的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
from models.gat import GAT
from datasets.freebase_to_homo import convert_freebase_to_homogeneous

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 1. 加载数据 ===
data = convert_freebase_to_homogeneous()
print(data)

book_idx = data.book_idx.clone().detach().cpu()
train_mask = data.train_mask.clone().detach().cpu()
test_mask = data.test_mask.clone().detach().cpu()
y = data.y.clone().detach().cpu()

data = data.to(device)
book_idx = book_idx.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)
y = y.to(device)

# === 2. 初始化模型 ===
num_classes = int(y.max().item()) + 1
model = GAT(128, 64, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# === 3. 训练 / 测试 ===
def train():
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    out_book = out[book_idx]
    loss = F.cross_entropy(out_book[train_mask], y[train_mask])

    loss.backward()
    optimizer.step()

    pred = out_book.argmax(dim=1)
    train_acc = (pred[train_mask] == y[train_mask]).float().mean()

    return loss.item(), train_acc.item()


from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)  # logits
    out_book = out[book_idx]              # 只取 book 节点
    pred = out_book.argmax(dim=1)

    y_true = y[test_mask].cpu()
    y_pred = pred[test_mask].cpu()

    acc = (y_pred == y_true).float().mean().item()
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # AUC（One-vs-Rest）
    y_score = F.softmax(out_book[test_mask], dim=1).cpu()
    y_true_onehot = F.one_hot(y_true, num_classes=y_score.size(1))

    try:
        auc = roc_auc_score(y_true_onehot, y_score, average='macro', multi_class='ovr')
    except ValueError:
        auc = float('nan')  # 防止 test set 中某些类别未出现

    return acc, f1_micro, f1_macro, auc


for epoch in range(1, 101):
    loss, train_acc = train()
    test_acc, test_f1_micro, test_f1_macro, test_auc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Acc: {test_acc:.4f}, Test F1-Mi: {test_f1_micro:.4f}, "
          f"Test F1-Ma: {test_f1_macro:.4f}, Test AUC: {test_auc:.4f}")
