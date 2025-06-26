import torch
from models.han import HAN
from datasets.load_acm import load_acm

# 加载 ACM 数据集
data = load_acm()

# 给没有特征的节点补上 x
if 'term' not in data.x_dict:
    data['term'].x = torch.randn(data['term'].num_nodes, 1902).float()



# Meta-path: paper → author → paper, paper → subject → paper
metapaths = [
    [('paper', 'to', 'author'), ('author', 'to', 'paper')],
    [('paper', 'to', 'subject'), ('subject', 'to', 'paper')],
]

# 获取类别数（paper的标签）
num_classes = int(data['paper'].y.max()) + 1

# 为所有节点加 self-loop + 转为无向图 + 添加元路径
# transform = Compose([
#     ToUndirected(),
#     AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=False),  # 保留原始边
# ])

# print(data)
for rel_type, edge_index in data.edge_index_dict.items():
    print(f"{rel_type}: edges={edge_index.size(1)}")


# HAN 需要输入每种元路径对应的边类型
model = HAN(
    in_channels=1902,
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

for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
