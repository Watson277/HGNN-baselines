import torch
import torch.nn.functional as F
from datasets.load_dblp import load_dblp
from models.han import HAN2

# 加载 DBLP 数据集
data = load_dblp()
print(data)

# 只选定用于分类的节点类型：author
target_node_type = 'author'

# 输入维度（各节点特征维度）
in_channels_dict = {
    'author': 334,
    'paper': 4231,
    'term': 50,
    'venue': 20
}

hidden_channels = 64
out_channels = 4  # 比如4分类，根据你的标签数调整

model = HAN2(
    in_channels_dict=in_channels_dict,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
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
