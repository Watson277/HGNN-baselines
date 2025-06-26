# train_gcn_on_dblp.py
import torch
from models.gcn import GCN
from datasets.dblp_to_homo import convert_dblp_to_homogeneous
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 1. 加载数据 + 提前“冷冻”索引/标签张量 ===
data = convert_dblp_to_homogeneous()
print(data)

author_idx = data.author_idx.clone().detach().cpu()
train_mask = data.train_mask.clone().detach().cpu()
test_mask = data.test_mask.clone().detach().cpu()
y = data.y.clone().detach().cpu()

data = data.to(device)
author_idx = author_idx.to(device)
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
    out_author = out[author_idx]
    loss = F.cross_entropy(out_author[train_mask], y[train_mask])

    loss.backward()
    optimizer.step()

    pred = out_author.argmax(dim=1)
    train_acc = (pred[train_mask] == y[train_mask]).float().mean()

    return loss.item(), train_acc.item()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    out_author = out[author_idx]
    pred = out_author.argmax(dim=1)
    acc = (pred[test_mask] == y[test_mask]).float().mean()
    return acc.item()

# === 4. 训练主循环 ===
for epoch in range(1, 101):
    loss, train_acc = train()
    test_acc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
