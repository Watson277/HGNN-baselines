import sys
import os
# 获取当前文件的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from models.hgt import HGT
import torch.nn.functional as F
from datasets.load_yelp import sample_train_mask_for_target_class


# 加载 HeteroData 对象
data = torch.load('./datasets/Yelp JSON/yelp.pt')
print(data)
data = sample_train_mask_for_target_class(data)

target_node_type = 'business'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)


model = HGT(
    in_channels=64,
    hidden_channels=64,
    out_channels=10,  # DBLP 中 author 的标签有 4 类
    metadata=data.metadata(),
    num_heads=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

author_y = data[target_node_type].y
train_mask = data[target_node_type].train_mask
test_mask = ~train_mask  # 没有 test_mask 字段，用反向来划分


def train():
    model.train()
    optimizer.zero_grad()
    out_dict = model(data.x_dict, data.edge_index_dict)
    out = out_dict[target_node_type]
    loss = F.cross_entropy(out[train_mask], author_y[train_mask])
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    train_acc = (pred[train_mask] == author_y[train_mask]).float().mean()
    return loss.item(), train_acc.item()

from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F
import torch

@torch.no_grad()
def test():
    model.eval()
    out_dict = model(data.x_dict, data.edge_index_dict)
    out = out_dict[target_node_type]  # logits
    pred = out.argmax(dim=1)

    y_true = author_y[test_mask].cpu()
    y_pred = pred[test_mask].cpu()

    acc = (y_pred == y_true).float().mean().item()
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # AUC（多分类 One-vs-Rest）
    try:
        probs = F.softmax(out[test_mask], dim=1).cpu()
        y_true_onehot = F.one_hot(y_true, num_classes=probs.size(1))
        auc = roc_auc_score(y_true_onehot, probs, average='macro', multi_class='ovr')
    except ValueError:
        auc = float('nan')

    return acc, f1_micro, f1_macro, auc


for epoch in range(1, 101):
    loss, train_acc = train()
    test_acc, test_f1_micro, test_f1_macro, test_auc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
          f"Test F1-Mi: {test_f1_micro:.4f}, Test F1-Ma: {test_f1_macro:.4f}, "
          f"Test AUC: {test_auc:.4f}")



