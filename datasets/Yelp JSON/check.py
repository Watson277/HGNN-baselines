import torch
from torch_geometric.data import HeteroData

# 加载 HeteroData 对象
data = torch.load('yelp_sampled_hetero2.pt')
# === 过滤掉未参与边连接的节点类型 ===
connected_node_types = set()
for src, _, dst in data.edge_types:
    connected_node_types.add(src)
    connected_node_types.add(dst)

# 删除未被连接的节点类型及其特征
for node_type in list(data.node_types):
    if node_type not in connected_node_types:
        del data[node_type]
torch.save(data, 'yelp_sampled_hetero2.pt')
print(data)

