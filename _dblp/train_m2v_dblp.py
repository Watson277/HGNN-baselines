from torch_geometric.nn.models import MetaPath2Vec
import torch
from datasets.load_dblp import load_dblp
import torch.nn.functional as F

# 加载 DBLP 数据集
data = load_dblp()

# 定义 metapath，例如：author-paper-author（APA）
metapath = [
    ('author', 'to', 'paper'),
    ('paper', 'to', 'author'),
]

# 初始化模型
model = MetaPath2Vec(
    edge_index_dict=data.edge_index_dict,
    embedding_dim=128,
    metapath=metapath,
    walk_length=10,
    context_size=5,
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

# 嵌入函数（只针对某一类节点）
@torch.no_grad()
def test():
    model.eval()
    z = model('author')  # 取出 author 的嵌入
    y = data['author'].y
    split = data['author'].train_mask, data['author'].test_mask

    accs = []
    for mask in split:
        clf = torch.nn.Linear(z.size(1), y.max().item() + 1).to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=5e-4)

        best_acc = 0
        for _ in range(50):  # 用 Logistic Regression 分类
            clf.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(clf(z[mask], ), y[mask])


            clf.eval()
            pred = clf(z[mask]).argmax(dim=1)
            acc = (pred == y[mask]).float().mean().item()
            best_acc = max(best_acc, acc)
        accs.append(best_acc)
    return accs

# 训练主循环
for epoch in range(1, 51):
    loss = train()
    train_acc, test_acc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


train_acc, test_acc = test()
print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

