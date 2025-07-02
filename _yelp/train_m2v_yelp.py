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

# 嵌入函数（只针对某一类节点）
@torch.no_grad()
def test():
    model.eval()
    z = model(target_node_type)  # 取出 author 的嵌入
    y = data[target_node_type].y
    split = data[target_node_type].train_mask, data[target_node_type].test_mask

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