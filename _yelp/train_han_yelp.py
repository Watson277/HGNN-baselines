import sys
import os
# 获取当前文件的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from models.han import HAN
from datasets.load_yelp import sample_train_mask_for_target_class
from utils.homophily import generate_meta_path_edge_index_from_rel, generate_metapaths, compute_homophily

# 加载 HeteroData 对象
data = torch.load('./datasets/Yelp JSON/yelp.pt')
print(data)
data = sample_train_mask_for_target_class(data)
target_node_type = 'business'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# 计算同配率
meta_paths = generate_metapaths(data.metadata(), center_type=target_node_type, max_hops=2)
for path in meta_paths:
    try:
        edge_index = generate_meta_path_edge_index_from_rel(data, path)
        homophily = compute_homophily(edge_index, data[target_node_type].y)
        print(f"{path}: 同配率 = {homophily:.4f}")
    except Exception as e:
        print(f"{path}: 计算失败 -> {e}")

# 获取类别数（paper的标签）
num_classes = int(data[target_node_type].y.max()) + 1

# HAN 需要输入每种元路径对应的边类型
model = HAN(
    in_channels=64,
    out_channels=num_classes,  # 类别数
    metadata=data.metadata(),
    hidden_channels=64,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    out = model(data.x_dict, data.edge_index_dict)
    out = out[target_node_type]
    loss = loss_fn(out[data[target_node_type].train_mask], data[target_node_type].y[data[target_node_type].train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)[target_node_type]
    pred = out.argmax(dim=1)

    accs = []
    for split in ['train_mask', 'test_mask']:
        mask = data[target_node_type][split]
        acc = (pred[mask] == data[target_node_type].y[mask]).sum() / mask.sum()
        accs.append(acc.item())

    # 测试集 F1
    test_mask = data[target_node_type]['test_mask']
    y_true = data[target_node_type].y[test_mask].cpu()
    y_pred = pred[test_mask].cpu()

    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # AUC（多分类 One-vs-Rest）
    try:
        probs = F.softmax(out[test_mask], dim=1).cpu()
        y_true_onehot = F.one_hot(y_true, num_classes=probs.size(1))
        auc = roc_auc_score(y_true_onehot, probs, average='macro', multi_class='ovr')
    except ValueError:
        auc = float('nan')  # 某些类别在 test_mask 中没出现会报错

    return accs[0], accs[1], f1_micro, f1_macro, auc


for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc, test_f1_micro, test_f1_macro, test_auc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
          f"Test F1-Micro: {test_f1_micro:.4f}, Test F1-Macro: {test_f1_macro:.4f}, "
          f"Test AUC: {test_auc:.4f}")

