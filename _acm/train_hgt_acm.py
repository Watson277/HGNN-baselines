import torch
from models.hgt import HGT
from datasets.load_acm import load_acm

# 加载 ACM 数据集
data = load_acm()

# 获取类别数（paper的标签）
num_classes = int(data['paper'].y.max()) + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HGT(
    in_channels=1902,
    hidden_channels=64,
    out_channels=num_classes,
    metadata=data.metadata(),
    num_heads=1
).to(device)

for node_type in data.x_dict:
    data[node_type].x = data[node_type].x.float()


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()

data = data.to(device)

def train():
    model.train()
    out = model(data.x_dict, data.edge_index_dict)
    out = out['paper']
    loss = loss_fn(out[data['paper'].train_mask], data['paper'].y[data['paper'].train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)['paper']
    pred = out.argmax(dim=1)

    accs = []
    for split in ['train_mask', 'test_mask']:
        mask = data['paper'][split]
        acc = (pred[mask] == data['paper'].y[mask]).sum() / mask.sum()
        accs.append(acc.item())
    return accs

if __name__ == '__main__':
    for epoch in range(1, 201):
        loss = train()
        train_acc, test_acc = test()
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


