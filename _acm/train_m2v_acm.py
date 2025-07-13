import sys
import os
# 获取当前文件的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from torch_geometric.nn.models import MetaPath2Vec
import torch
from datasets.load_acm import load_acm, sample_train_mask_for_target_class, node_type
import torch.nn.functional as F

target_node_type = node_type

# 加载 ACM 数据集
data = load_acm()
# 没类选取10个节点
data = sample_train_mask_for_target_class(data)
print(data)

metapath = [
    ('paper', 'to', 'author'),
    ('author', 'to', 'paper'),
]

model = MetaPath2Vec(
    edge_index_dict=data.edge_index_dict,
    embedding_dim=128,
    metapath=metapath,
    walk_length=50,
    context_size=5,
    walks_per_node=3,
    num_negative_samples=5,
    num_nodes_dict={k: data[k].num_nodes for k in data.node_types},
    sparse=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
data = data.to(device)

# 获取模型内置的随机游走采样器
loader = model.loader(batch_size=128, shuffle=True, num_workers=0)

# 优化器（只优化嵌入参数）
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

from sklearn.metrics import f1_score

# 训练函数
def train():
    model.train()
    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        pos_rw, neg_rw = pos_rw.to(device), neg_rw.to(device)
        optimizer.zero_grad()
        loss = model.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / (i + 1)


from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F

@torch.no_grad()
def test():
    model.eval()
    z = model(target_node_type)  # 获取 paper 节点嵌入
    y = data[target_node_type].y
    train_mask, test_mask = data[target_node_type].train_mask, data[target_node_type].test_mask
    num_classes = y.max().item() + 1

    def evaluate(mask):
        clf = torch.nn.Linear(z.size(1), num_classes).to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=5e-4)

        best_acc = 0.0
        best_pred = None
        best_logits = None

        for _ in range(50):
            clf.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(clf(z[mask]), y[mask])

            clf.eval()
            with torch.no_grad():
                logits = clf(z[mask])
                pred = logits.argmax(dim=1)
                acc = (pred == y[mask]).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_pred = pred.cpu()
                    best_logits = logits.cpu()

        return best_acc, best_pred, y[mask].cpu(), best_logits

    train_acc, _, _, _ = evaluate(train_mask)
    test_acc, y_pred_test, y_true_test, test_logits = evaluate(test_mask)

    # 计算 F1
    f1_micro = f1_score(y_true_test, y_pred_test, average='micro')
    f1_macro = f1_score(y_true_test, y_pred_test, average='macro')

    # AUC 计算（多分类支持 one-vs-rest）
    y_true_onehot = F.one_hot(y_true_test, num_classes=num_classes)
    y_prob = F.softmax(test_logits, dim=1)
    try:
        test_auc = roc_auc_score(y_true_onehot, y_prob, average='macro', multi_class='ovr')
    except ValueError:
        test_auc = float('nan')  # 若计算失败则返回 nan

    return train_acc, test_acc, f1_micro, f1_macro, test_auc

for epoch in range(1, 51):
    loss = train()
    train_acc, test_acc, test_f1_micro, test_f1_macro, test_auc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
          f"Test F1-Mi: {test_f1_micro:.4f}, Test F1-Ma: {test_f1_macro:.4f}, "
          f"Test AUC: {test_auc:.4f}")





