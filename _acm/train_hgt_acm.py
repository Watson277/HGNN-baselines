import sys
import os
# 获取当前文件的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from models.hgt import HGT
from datasets.load_acm import load_acm, sample_train_mask_for_target_class, node_type

# 加载 ACM 数据集
data = load_acm()
# 没类选取10个节点
data = sample_train_mask_for_target_class(data)

print(data)

target_node_type = node_type

# 获取类别数（paper的标签）
num_classes = int(data['paper'].y.max()) + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HGT(
    in_channels=1902,
    hidden_channels=64,
    out_channels=num_classes,
    metadata=data.metadata(),
    num_heads=1
).to(device)

for node_type in data.x_dict:
    data[node_type].x = data[node_type].x.float()


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()

data = data.to(device)

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
    out = model(data.x_dict, data.edge_index_dict)[target_node_type]  # [num_nodes, num_classes]
    pred = out.argmax(dim=1)

    # 计算 train/test accuracy
    accs = []
    for split in ['train_mask', 'test_mask']:
        mask = data[target_node_type][split]
        acc = (pred[mask] == data[target_node_type].y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)

    # 只计算 test F1 和 AUC
    test_mask = data[target_node_type]['test_mask']
    y_true = data[target_node_type].y[test_mask].cpu()
    y_pred = pred[test_mask].cpu()

    test_f1_micro = f1_score(y_true, y_pred, average='micro')
    test_f1_macro = f1_score(y_true, y_pred, average='macro')

    # 计算 AUC：需要 one-hot label 和 softmax 概率
    y_prob = F.softmax(out[test_mask], dim=1).cpu()  # [num_test_samples, num_classes]
    y_true_onehot = F.one_hot(y_true, num_classes=y_prob.size(1)).float()

    try:
        test_auc_macro = roc_auc_score(y_true_onehot, y_prob, average='macro', multi_class='ovr')
    except ValueError:
        test_auc_macro = float('nan')  # 如果只有一个类，AUC 无法计算

    return accs[0], accs[1], test_f1_micro, test_f1_macro, test_auc_macro

# 主训练循环
if __name__ == '__main__':
    for epoch in range(1, 201):
        loss = train()
        train_acc, test_acc, test_f1_micro, test_f1_macro, test_auc = test()
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
              f"Test F1-Mi: {test_f1_micro:.4f}, Test F1-Ma: {test_f1_macro:.4f}, "
              f"Test AUC: {test_auc:.4f}")





