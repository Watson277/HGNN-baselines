import json
import torch
import random
from tqdm import tqdm
from collections import Counter
from torch.nn import Linear
from torch_geometric.data import HeteroData

ROOT = './datasets/Yelp JSON/'
files = {
    'business': 'yelp_academic_dataset_business.json',
    'user': 'yelp_academic_dataset_user.json',
    'review': 'yelp_academic_dataset_review.json',
    'checkin': 'yelp_academic_dataset_checkin.json',  # 保留原路径，但不再使用
    'tip': 'yelp_academic_dataset_tip.json'
}

data = HeteroData()
embed_size = 64
MAX_NODES = 50000  # 每类节点最多保留数量

# === 1. business 节点 & 标签 ===
business_map = {}
business_feats = []
business_categories = []

with open(ROOT + files['business'], 'r', encoding='utf8') as f:
    for i, line in enumerate(tqdm(f, desc='Business')):
        b = json.loads(line)
        business_map[b['business_id']] = i
        business_feats.append([
            b['stars'], b['review_count'], len(b['categories'] or "")
        ])
        if b['categories']:
            main_cat = b['categories'].split(',')[0].strip()
        else:
            main_cat = 'Unknown'
        business_categories.append(main_cat)

business_feats = torch.tensor(business_feats, dtype=torch.float)
data['business'].x = business_feats

# ✅ 标签处理：确保最终只有10类（0~9），'other' 为最后一类
N = 10
top_cats = [c for c, _ in Counter(business_categories).most_common(N - 1)]  # 前9类
cat2label = {c: i for i, c in enumerate(top_cats)}
cat2label['other'] = N - 1  # 'other' 类编号为9

labels = [cat2label.get(cat, N - 1) for cat in business_categories]  # unknown 映射为9
y = torch.tensor(labels, dtype=torch.long)
data['business'].y = y


# 划分训练集
num_nodes = y.size(0)
perm = torch.randperm(num_nodes)
train_end = int(0.7 * num_nodes)
val_end = int(0.85 * num_nodes)
data['business'].train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data['business'].val_mask = torch.zeros(num_nodes, dtype=torch.bool)
data['business'].test_mask = torch.zeros(num_nodes, dtype=torch.bool)
data['business'].train_mask[perm[:train_end]] = True
data['business'].val_mask[perm[train_end:val_end]] = True
data['business'].test_mask[perm[val_end:]] = True

# === 2. user 节点 ===
user_map = {}
user_feats = []
with open(ROOT + files['user'], 'r', encoding='utf8') as f:
    for i, line in enumerate(tqdm(f, desc='User')):
        u = json.loads(line)
        user_map[u['user_id']] = i
        user_feats.append([
            u['review_count'], u['average_stars'], u['useful'], u['funny'], u['cool']
        ])
data['user'].x = torch.tensor(user_feats, dtype=torch.float)

# === 3. review 节点与边 user -> review -> business ===
review_map = {}
review_feats = []
u2r = []
r2b = []
with open(ROOT + files['review'], 'r', encoding='utf8') as f:
    for i, line in enumerate(tqdm(f, desc='Review')):
        r = json.loads(line)
        if r['user_id'] in user_map and r['business_id'] in business_map:
            rid = len(review_map)
            review_map[r['review_id']] = rid
            review_feats.append([r['stars'], r['useful'], r['funny'], r['cool']])
            u2r.append([user_map[r['user_id']], rid])
            r2b.append([rid, business_map[r['business_id']]])
data['review'].x = torch.tensor(review_feats, dtype=torch.float)
data['user', 'writes', 'review'].edge_index = torch.tensor(u2r).t().contiguous()
data['review', 'about', 'business'].edge_index = torch.tensor(r2b).t().contiguous()

# === 4. user -> user 好友边 ===
user_edges = []
with open(ROOT + files['user'], 'r', encoding='utf8') as f:
    for line in tqdm(f, desc='User-Friends'):
        u = json.loads(line)
        uid = user_map.get(u['user_id'])
        if uid is not None:
            for fid in u['friends'].split(', '):
                if fid in user_map:
                    user_edges.append([uid, user_map[fid]])
data['user', 'friends', 'user'].edge_index = torch.tensor(user_edges, dtype=torch.long).t().contiguous()

# === 5. tip 节点与边 user -> tip -> business ===
tip_map = {}
tip_feats = []
u2tip = []
tip2b = []
with open(ROOT + files['tip'], 'r', encoding='utf8') as f:
    for line in tqdm(f, desc='Tip'):
        t = json.loads(line)
        if t['user_id'] in user_map and t['business_id'] in business_map:
            tid = len(tip_map)
            tip_map[f"{t['user_id']}_{t['business_id']}_{t['date']}"] = tid
            tip_feats.append([t['compliment_count'], len(t['text'])])
            u2tip.append([user_map[t['user_id']], tid])
            tip2b.append([tid, business_map[t['business_id']]])
data['tip'].x = torch.tensor(tip_feats, dtype=torch.float)
data['user', 'tips', 'tip'].edge_index = torch.tensor(u2tip, dtype=torch.long).t().contiguous()
data['tip', 'about', 'business'].edge_index = torch.tensor(tip2b, dtype=torch.long).t().contiguous()

# ✅ === 删除 checkin 节点及其边 ===
if 'checkin' in data.node_types:
    del data['checkin']
if ('checkin', 'at', 'business') in data.edge_types:
    del data[('checkin', 'at', 'business')]

# === 截断每类节点为 MAX_NODES，并保留相关边 ===
keep_nodes = {}
old2new = {}

for ntype in data.node_types:
    num_nodes = data[ntype].num_nodes
    keep_idx = torch.randperm(num_nodes)[:MAX_NODES]
    keep_nodes[ntype] = keep_idx
    old2new[ntype] = {int(old): i for i, old in enumerate(keep_idx)}
    
    data[ntype].x = data[ntype].x[keep_idx]
    if 'y' in data[ntype]:
        data[ntype].y = data[ntype].y[keep_idx]
    for mask in ['train_mask', 'val_mask', 'test_mask']:
        if mask in data[ntype]:
            data[ntype][mask] = data[ntype][mask][keep_idx]

# 过滤并重映射边
for src, rel, dst in list(data.edge_types):  # list 防止在循环中边被删
    edge_index = data[(src, rel, dst)].edge_index
    src_mask = torch.tensor([int(s) in old2new[src] for s in edge_index[0]])
    dst_mask = torch.tensor([int(d) in old2new[dst] for d in edge_index[1]])
    keep_mask = src_mask & dst_mask

    ei = edge_index[:, keep_mask]
    if ei.size(1) == 0:
        del data[(src, rel, dst)]
        continue

    src_new = [old2new[src][int(i)] for i in ei[0]]
    dst_new = [old2new[dst][int(i)] for i in ei[1]]
    data[(src, rel, dst)].edge_index = torch.tensor([src_new, dst_new], dtype=torch.long)

# === 所有节点统一映射为 64 维特征 ===
for ntype in data.node_types:
    in_dim = data[ntype].x.size(-1)
    data[ntype].x = Linear(in_dim, embed_size)(data[ntype].x)

# === 保存 ===
torch.save(data, './datasets/Yelp JSON/yelp.pt')
print("✅ 已保存为 yelp.pt")
print(data)
