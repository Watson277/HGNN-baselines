import sys
import os
# 获取当前文件的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from datasets.load_freebase import load_freebase, add_node_features, sample_train_mask_for_target_class, node_type
from models.han import HAN2
import torch.nn.functional as F
from utils.homophily import generate_meta_path_edge_index_from_rel, generate_metapaths, compute_homophily

data = load_freebase()
data = add_node_features(data, feature_dim=128)
data = sample_train_mask_for_target_class(data)
print(data)

# 只选定用于分类的节点类型：book
target_node_type = node_type

# 计算同配率
meta_paths = generate_metapaths(data.metadata(), center_type=target_node_type, max_hops=2)
for path in meta_paths:
    try:
        edge_index = generate_meta_path_edge_index_from_rel(data, path)
        homophily = compute_homophily(edge_index, data[target_node_type].y)
        print(f"{path}: 同配率 = {homophily:.4f}")
    except Exception as e:
        print(f"{path}: 计算失败 -> {e}")

# 获取类别数（book的标签）
num_classes = int(data[target_node_type].y.max()) + 1

in_channels_dict = {
    node_type: data[node_type].num_features
    for node_type in data.node_types
}

model = HAN2(
    in_channels_dict=in_channels_dict,
    hidden_channels=64,
    out_channels=num_classes,
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
    out = model(data.x_dict, data.edge_index_dict)[target_node_type]
    pred = out.argmax(dim=1)

    accs = []
    for split in ['train_mask', 'test_mask']:
        mask = data[target_node_type][split]
        acc = (pred[mask] == data[target_node_type].y[mask]).sum() / mask.sum()
        accs.append(acc.item())

    # F1 分数
    test_mask = data[target_node_type]['test_mask']
    y_true = data[target_node_type].y[test_mask].cpu()
    y_pred = pred[test_mask].cpu()

    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # AUC 分数（多分类 One-vs-Rest）
    y_score = F.softmax(out[test_mask], dim=1).cpu()
    y_true_one_hot = F.one_hot(y_true, num_classes=y_score.size(1))

    try:
        auc = roc_auc_score(y_true_one_hot, y_score, average='macro', multi_class='ovr')
    except ValueError:
        auc = float('nan')  # 防止某些类别在test中没有出现时抛错

    return accs[0], accs[1], f1_micro, f1_macro, auc


if __name__ == '__main__':
    for epoch in range(1, 201):
        loss = train()
        train_acc, test_acc, test_f1_micro, test_f1_macro, test_auc = test()
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
              f"Test F1-Mi: {test_f1_micro:.4f}, Test F1-Ma: {test_f1_macro:.4f}, "
              f"Test AUC: {test_auc:.4f}")



    