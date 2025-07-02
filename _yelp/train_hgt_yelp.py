import torch
from models.hgt import HGT
import torch.nn.functional as F


# 加载 HeteroData 对象
data = torch.load('./datasets/Yelp JSON/yelp.pt')
print(data)

target_node_type = 'business'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)


model = HGT(
    in_channels=64,
    hidden_channels=64,
    out_channels=10,  # DBLP 中 author 的标签有 4 类
    metadata=data.metadata(),
    num_heads=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

author_y = data[target_node_type].y
train_mask = data[target_node_type].train_mask
test_mask = ~train_mask  # 没有 test_mask 字段，用反向来划分


def train():
    model.train()
    optimizer.zero_grad()
    out_dict = model(data.x_dict, data.edge_index_dict)
    out = out_dict[target_node_type]
    loss = F.cross_entropy(out[train_mask], author_y[train_mask])
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    train_acc = (pred[train_mask] == author_y[train_mask]).float().mean()
    return loss.item(), train_acc.item()

@torch.no_grad()
def test():
    model.eval()
    out_dict = model(data.x_dict, data.edge_index_dict)
    out = out_dict[target_node_type]
    pred = out.argmax(dim=1)
    acc = (pred[test_mask] == author_y[test_mask]).float().mean()
    return acc.item()

for epoch in range(1, 101):
    loss, train_acc = train()
    test_acc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")