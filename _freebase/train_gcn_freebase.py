import sys
import os
# 获取当前文件的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
from models.gcn import GCN
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
model = GCN(128, 64, num_classes).to(device)
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
    y_true_onehot = F.one_hot(y_true, num_classes=y_score.size(1)).float()

    try:
        auc = roc_auc_score(y_true_onehot, y_score, average='macro', multi_class='ovr')
    except ValueError:
        auc = float('nan')

    # ✅ MSE
    mse = F.mse_loss(y_score, y_true_onehot).item()

    return acc, f1_micro, f1_macro, auc, mse

best_acc = 0.0
best_epoch = 0
best_result = None

for epoch in range(1, 101):
    loss, train_acc = train()
    test_acc, test_f1_micro, test_f1_macro, test_auc, test_mse = test()
    
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Acc: {test_acc:.4f}, Test F1-Mi: {test_f1_micro:.4f}, "
          f"Test F1-Ma: {test_f1_macro:.4f}, Test AUC: {test_auc:.4f}, Test MSE: {test_mse:.6f}")

    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch
        best_result = {
            'Loss': loss,
            'Train Acc': train_acc,
            'Test Acc': test_acc,
            'F1 Micro': test_f1_micro,
            'F1 Macro': test_f1_macro,
            'AUC': test_auc,
            'MSE': test_mse
        }

# ✅ 最后输出最佳一轮结果
print("\n=== Best Test Accuracy Result ===")
print(f"Epoch: {best_epoch:03d}, Loss: {best_result['Loss']:.4f}, "
      f"Train Acc: {best_result['Train Acc']:.4f}, Test Acc: {best_result['Test Acc']:.4f}, "
      f"F1 Micro: {best_result['F1 Micro']:.4f}, F1 Macro: {best_result['F1 Macro']:.4f}, "
      f"AUC: {best_result['AUC']:.4f}, MSE: {best_result['MSE']:.6f}")

log = (f"Epoch: {best_epoch:03d}, Loss: {best_result['Loss']:.4f}, "
      f"Train Acc: {best_result['Train Acc']:.4f}, Test Acc: {best_result['Test Acc']:.4f}, "
      f"F1 Micro: {best_result['F1 Micro']:.4f}, F1 Macro: {best_result['F1 Macro']:.4f}, "
      f"AUC: {best_result['AUC']:.4f}, MSE: {best_result['MSE']:.6f}")

# 获取当前脚本名并构造同名 txt 文件
py_file = sys.argv[0]
base_name = os.path.splitext(os.path.basename(py_file))[0]
txt_filename = "./result/freebase/" + base_name + ".txt"
print(txt_filename)
print(log)
with open(txt_filename, 'a', encoding='utf-8') as f:
    print("write")
    f.write(log)
    f.write("\n")