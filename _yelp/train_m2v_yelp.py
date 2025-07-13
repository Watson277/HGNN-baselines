import sys
import os
# 获取当前文件的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from torch_geometric.nn.models import MetaPath2Vec
import torch
import torch.nn.functional as F

# 加载 HeteroData 对象
data = torch.load('./datasets/Yelp JSON/yelp.pt')
print(data)

target_node_type = 'business'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)


metapaths = [
    ('user', 'friends', 'user'), 
    ('user', 'writes', 'review'), 
    ('review', 'about', 'business')
]

model = MetaPath2Vec(
    edge_index_dict=data.edge_index_dict,
    embedding_dim=64,
    metapath=metapaths,  # 👈 传入一个元组列表（不是列表的列表）
    walk_length=3,
    context_size=3,
    walks_per_node=5,
    num_negative_samples=5,
    num_nodes_dict={key: data[key].num_nodes for key in data.node_types},
    sparse=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

# 获取模型内置的随机游走采样器
loader = model.loader(batch_size=128, shuffle=True, num_workers=0)

# 优化器（只优化嵌入参数）
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

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

def test():
    model.eval()
    with torch.no_grad():
        z = model(target_node_type)  # 节点嵌入

    y = data[target_node_type].y
    train_mask = data[target_node_type].train_mask
    test_mask = data[target_node_type].test_mask

    clf = torch.nn.Linear(z.size(1), y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=5e-4)

    # 训练分类器（只用训练集）
    clf.train()
    for _ in range(50):
        optimizer.zero_grad()
        loss = F.cross_entropy(clf(z[train_mask]), y[train_mask])
        loss.backward()
        optimizer.step()

    # 测试分类器
    clf.eval()
    with torch.no_grad():
        logits_test = clf(z[test_mask])
        probs_test = F.softmax(logits_test, dim=1).cpu()
        pred_test = logits_test.argmax(dim=1)

    train_acc = (clf(z[train_mask]).argmax(dim=1) == y[train_mask]).float().mean().item()
    test_acc = (pred_test == y[test_mask]).float().mean().item()

    y_true_test = y[test_mask].cpu()
    test_f1_micro = f1_score(y_true_test, pred_test.cpu(), average='micro')
    test_f1_macro = f1_score(y_true_test, pred_test.cpu(), average='macro')

    # 计算多分类 AUC（One-vs-Rest）
    try:
        y_true_onehot = F.one_hot(y_true_test, num_classes=probs_test.size(1))
        test_auc = roc_auc_score(y_true_onehot, probs_test, average='macro', multi_class='ovr')
    except ValueError:
        test_auc = float('nan')  # 某些类别在测试集中缺失时

    return train_acc, test_acc, test_f1_micro, test_f1_macro, test_auc


for epoch in range(1, 51):
    loss = train()
    train_acc, test_acc, test_f1_micro, test_f1_macro, test_auc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
          f"Test F1-Micro: {test_f1_micro:.4f}, Test F1-Macro: {test_f1_macro:.4f}, "
          f"Test AUC: {test_auc:.4f}")


