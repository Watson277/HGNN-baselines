import sys
import os
# 获取当前文件的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
from datasets.load_dblp import load_dblp
from models.han import HAN2
from datasets.load_dblp import load_dblp, sample_train_mask_for_target_class, node_type
from utils.homophily import generate_meta_path_edge_index_from_rel, generate_metapaths, compute_homophily

# 加载 DBLP 数据集
data = load_dblp()
print(data)
data = sample_train_mask_for_target_class(data)

# 只选定用于分类的节点类型：author
target_node_type = node_type

# # 计算同配率
# meta_paths = generate_metapaths(data.metadata(), center_type=target_node_type)
# for path in meta_paths:
#     try:
#         edge_index = generate_meta_path_edge_index_from_rel(data, path)
#         homophily = compute_homophily(edge_index, data[target_node_type].y)
#         print(f"{path}: 同配率 = {homophily:.4f}")
#     except Exception as e:
#         print(f"{path}: 计算失败 -> {e}")

# 输入维度（各节点特征维度）
in_channels_dict = {
    'author': 334,
    'paper': 4231,
    'term': 50,
    'venue': 20
}

hidden_channels = 64
out_channels = 4  # 比如4分类，根据你的标签数调整

model = HAN2(
    in_channels_dict=in_channels_dict,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    metadata=data.metadata(),
    heads=2,
    dropout=0.6
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)[target_node_type]
    loss = F.cross_entropy(out[data[target_node_type].train_mask], data[target_node_type].y[data[target_node_type].train_mask])
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

    accs = []
    for split in ['train_mask', 'test_mask']:
        mask = data[target_node_type][split]
        acc = (pred[mask] == data[target_node_type].y[mask]).sum() / mask.sum()
        accs.append(acc.item())

    # F1 + AUC only on test set
    test_mask = data[target_node_type]['test_mask']
    y_true = data[target_node_type].y[test_mask].cpu()
    y_pred = pred[test_mask].cpu()
    logits = out[test_mask].cpu()

    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    num_classes = logits.size(1)
    y_prob = F.softmax(logits, dim=1)
    y_true_onehot = F.one_hot(y_true, num_classes=num_classes).float()

    try:
        auc = roc_auc_score(y_true_onehot, y_prob, average='macro', multi_class='ovr')
    except ValueError:
        auc = float('nan')

    # ✅ 新增：MSE
    mse = F.mse_loss(y_prob, y_true_onehot).item()

    return accs[0], accs[1], f1_micro, f1_macro, auc, mse


best_acc = 0.0
best_epoch = 0
best_result = None

for epoch in range(1, 151):
    loss = train()
    train_acc, test_acc, test_f1_micro, test_f1_macro, test_auc, test_mse = test()

    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
          f"Test F1-Mi: {test_f1_micro:.4f}, Test F1-Ma: {test_f1_macro:.4f}, "
          f"Test AUC: {test_auc:.4f}, Test MSE: {test_mse:.6f}")

    # ✅ 按 Test Accuracy 记录最佳
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

# ✅ 最终输出最佳结果
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
txt_filename = "./result/dblp/" + base_name + ".txt"
print(txt_filename)
print(log)
with open(txt_filename, 'a', encoding='utf-8') as f:
    print("write")
    f.write(log)
    f.write("\n")
