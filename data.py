from torch_geometric.datasets import Yelp
import torch_geometric.transforms as T

# 设置数据集存储路径
path = './data/Yelp'

# 加载数据集
dataset = Yelp(root='/tmp/Yelp')
# 取出图对象
data = dataset[0]

print(data)




