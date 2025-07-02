import torch
from datasets.load_aminer import load_aminer
from models.hgt import HGT

data = load_aminer()
print(data)

target_node_type = 'author'

# 获取类别数（paper的标签）
num_classes = int(data[target_node_type].y.max()) + 1
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HGT(
    in_channels=128,
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

if __name__ == '__main__':
    for epoch in range(1, 201):
        loss = train()
        train_acc, test_acc = test()
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")