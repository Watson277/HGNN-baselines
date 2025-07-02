from torch_geometric.nn.models import MetaPath2Vec
import torch
from datasets.load_acm import load_acm

# 加载 ACM 数据集
data = load_acm()

metapath = [
    ('paper', 'to', 'author'),
    ('author', 'to', 'paper'),
]

model = MetaPath2Vec(
    edge_index_dict=data.edge_index_dict,
    embedding_dim=128,
    metapath=metapath,
    walk_length=50,
    context_size=5,
    walks_per_node=3,
    num_negative_samples=5,
    num_nodes_dict={k: data[k].num_nodes for k in data.node_types},
    sparse=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
data = data.to(device)

def train():
    model.train()
    total_loss = 0
    loader = model.loader(batch_size=128, shuffle=True, num_workers=2)

    for pos_rw, neg_rw in loader:
        pos_rw, neg_rw = pos_rw.to(device), neg_rw.to(device)
        optimizer.zero_grad()
        loss = model.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@torch.no_grad()
def test():
    model.eval()

    # 获取 paper 节点的嵌入（假设我们要分类 paper 节点）
    z = model('paper')
    z = z.cpu().numpy()

    y = data['paper'].y.cpu().numpy()
    train_mask = data['paper'].train_mask.cpu().numpy()
    test_mask = data['paper'].test_mask.cpu().numpy()

    z_train, y_train = z[train_mask], y[train_mask]
    z_test, y_test = z[test_mask], y[test_mask]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(z_train, y_train)

    y_pred_train = clf.predict(z_train)
    y_pred_test = clf.predict(z_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    return train_acc, test_acc


if __name__ == '__main__':
    for epoch in range(1, 101):
        loss = train()
        train_acc, test_acc = test()  # 确保 test() 返回两个准确率
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


