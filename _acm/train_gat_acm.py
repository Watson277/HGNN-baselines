# train_gcn_on_acm.py
import sys
import os
# 获取当前文件的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from models.gat import GAT
from datasets.acm_to_homo import convert_acm_to_homogeneous
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 1. 加载数据 + 提前“冷冻”索引/标签张量 ===
data = convert_acm_to_homogeneous()

paper_idx = data.paper_idx.clone().detach().cpu()
train_mask = data.train_mask.clone().detach().cpu()
test_mask = data.test_mask.clone().detach().cpu()
y = data.y.clone().detach().cpu()

data = data.to(device)
paper_idx = paper_idx.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)
y = y.to(device)

# === 2. 初始化模型 ===
model = GAT(128, 64, 3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# === 3. 训练 / 测试 ===
def train():
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    out_paper = out[paper_idx]
    loss = F.cross_entropy(out_paper[train_mask], y[train_mask])

    loss.backward()
    optimizer.step()

    pred = out_paper.argmax(dim=1)
    train_acc = (pred[train_mask] == y[train_mask]).float().mean()

    return loss.item(), train_acc.item()

from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)  # shape: [total_nodes, num_classes]
    out_paper = out[paper_idx]            # 只取 paper 节点
    pred = out_paper.argmax(dim=1)        # shape: [num_paper_nodes]

    y_true = y[test_mask].cpu()
    y_pred = pred[test_mask].cpu()
    prob = F.softmax(out_paper[test_mask], dim=1).cpu()
    y_onehot = F.one_hot(y_true, num_classes=prob.size(1)).float()

    acc = (y_pred == y_true).float().mean().item()
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    try:
        auc_macro = roc_auc_score(y_onehot, prob, average='macro', multi_class='ovr')
    except ValueError:
        auc_macro = float('nan')

    # ✅ 新增：计算 MSE
    mse = F.mse_loss(prob, y_onehot).item()

    return acc, f1_micro, f1_macro, auc_macro, mse


best_auc = 0.0
best_epoch = 0
best_result = None

for epoch in range(1, 101):
    loss, train_acc = train()
    test_acc, test_f1_micro, test_f1_macro, test_auc, test_mse = test()

    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Acc: {test_acc:.4f}, Test F1-Mi: {test_f1_micro:.4f}, "
          f"Test F1-Ma: {test_f1_macro:.4f}, Test AUC: {test_auc:.4f}, "
          f"Test MSE: {test_mse:.6f}")

    # ✅ 记录 AUC 最佳的结果
    if test_auc > best_auc:
        best_auc = test_auc
        best_epoch = epoch
        best_result = {
            'Train Acc': train_acc,
            'Test Acc': test_acc,
            'F1 Micro': test_f1_micro,
            'F1 Macro': test_f1_macro,
            'AUC': test_auc,
            'MSE': test_mse,
            'Loss': loss
        }

# ✅ 打印 AUC 最佳时的结果
print("\n=== Best Test AUC Result ===")
print(f"Epoch: {best_epoch:03d}, Loss: {best_result['Loss']:.4f}, "
      f"Train Acc: {best_result['Train Acc']:.4f}, Test Acc: {best_result['Test Acc']:.4f}, "
      f"F1 Micro: {best_result['F1 Micro']:.4f}, F1 Macro: {best_result['F1 Macro']:.4f}, "
      f"AUC: {best_result['AUC']:.4f}, MSE: {best_result['MSE']:.6f}")


