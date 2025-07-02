import torch
import torch.nn.functional as F
from models.gat import GAT
from datasets.freebase_to_homo import convert_freebase_to_homogeneous

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 1. 加载数据 ===
data = convert_freebase_to_homogeneous()
print(data)

book_idx = data.book_idx.clone().detach().cpu()
train_mask = data.train_mask.clone().detach().cpu()
test_mask = data.test_mask.clone().detach().cpu()
y = data.y.clone().detach().cpu()

data = data.to(device)
book_idx = book_idx.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)
y = y.to(device)

# === 2. 初始化模型 ===
num_classes = int(y.max().item()) + 1
model = GAT(128, 64, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# === 3. 训练 / 测试 ===
def train():
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    out_book = out[book_idx]
    loss = F.cross_entropy(out_book[train_mask], y[train_mask])

    loss.backward()
    optimizer.step()

    pred = out_book.argmax(dim=1)
    train_acc = (pred[train_mask] == y[train_mask]).float().mean()

    return loss.item(), train_acc.item()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    out_book = out[book_idx]
    pred = out_book.argmax(dim=1)
    acc = (pred[test_mask] == y[test_mask]).float().mean()
    return acc.item()

# === 4. 训练主循环 ===
for epoch in range(1, 101):
    loss, train_acc = train()
    test_acc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
