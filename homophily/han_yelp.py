import torch
from torch_geometric.transforms import AddMetaPaths

# 加载数据
data = torch.load('./datasets/Yelp JSON/yelp.pt')

# 查看原始边类型
print("原始边类型:", data.edge_index_dict.keys())

# 构造 metapath
metapaths = [
    [('user', 'rates', 'business'), ('business', 'rev_by', 'user')]
]

# 添加 metapath 边
transform = AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=False)
data = transform(data)

# 查看变换后的边类型
print("添加元路径后的边类型:", data.edge_index_dict.keys())
