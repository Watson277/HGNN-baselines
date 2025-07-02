import torch
from models.han import HAN

# 加载 HeteroData 对象
data = torch.load('./datasets/Yelp JSON/yelp.pt')
print(data)

target_node_type = 'business'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)


# 获取类别数（paper的标签）
num_classes = int(data[target_node_type].y.max()) + 1

# HAN 需要输入每种元路径对应的边类型
model = HAN(
    in_channels=64,
    out_channels=num_classes,  # 类别数
    metadata=data.metadata(),
    hidden_channels=64,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    out = model(data.x_dict, data.edge_index_dict)
    out = out[target_node_type]
    loss = loss_fn(out[data[target_node_type].train_mask], data[target_node_type].y[data[target_node_type].train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

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
    return accs

for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')