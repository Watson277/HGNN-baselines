import torch
from models.hgt import HGT2
import torch.nn.functional as F
from datasets.load_dblp import load_dblp

# 加载 DBLP 数据集
data = load_dblp()
print(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

in_channels_dict = {
    node_type: data[node_type].num_features
    for node_type in data.node_types
}


model = HGT2(
    in_channels_dict=in_channels_dict,
    hidden_channels=128,
    out_channels=4,  # DBLP 中 author 的标签有 4 类
    metadata=data.metadata(),
    num_heads=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

author_y = data['author'].y
train_mask = data['author'].train_mask
test_mask = ~train_mask  # 没有 test_mask 字段，用反向来划分

def train():
    model.train()
    optimizer.zero_grad()
    out_dict = model(data.x_dict, data.edge_index_dict)
    out = out_dict['author']
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
    out = out_dict['author']
    pred = out.argmax(dim=1)
    acc = (pred[test_mask] == author_y[test_mask]).float().mean()
    return acc.item()

for epoch in range(1, 101):
    loss, train_acc = train()
    test_acc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

