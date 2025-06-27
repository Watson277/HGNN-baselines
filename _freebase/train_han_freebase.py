import torch
from datasets.load_freebase import load_freebase, add_node_features
from models.han import HAN2
import torch.nn.functional as F

data = load_freebase()
data = add_node_features(data, feature_dim=128)
print(data)

# 获取类别数（book的标签）
num_classes = int(data['book'].y.max()) + 1

# 只选定用于分类的节点类型：book
target_node_type = 'book'

in_channels_dict = {
    node_type: data[node_type].num_features
    for node_type in data.node_types
}

model = HAN2(
    in_channels_dict=in_channels_dict,
    hidden_channels=64,
    out_channels=num_classes,
    metadata=data.metadata(),
    heads=2,
    dropout=0.6
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)[target_node_type]
    loss = F.cross_entropy(out[data[target_node_type].train_mask], data[target_node_type].y[data[target_node_type].train_mask])
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

for epoch in range(1, 101):
    loss = train()
    train_acc, test_acc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


    