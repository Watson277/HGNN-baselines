import json
from tqdm import tqdm
import torch
from collections import Counter
from torch_geometric.data import HeteroData
from torch.nn import Linear

ROOT = './datasets/Yelp JSON/'  # 数据目录
files = {
    'business': 'yelp_academic_dataset_business.json',
    'user': 'yelp_academic_dataset_user.json',
    'review': 'yelp_academic_dataset_review.json',
    'checkin': 'yelp_academic_dataset_checkin.json',
    'tip': 'yelp_academic_dataset_tip.json'
}

data = HeteroData()
embed_size = 64  # 所有节点最终嵌入维度

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
        # 主类标签
        if b['categories']:
            main_cat = b['categories'].split(',')[0].strip()
        else:
            main_cat = 'Unknown'
        business_categories.append(main_cat)

business_feats = torch.tensor(business_feats, dtype=torch.float)
data['business'].x = business_feats

# === 构建分类标签 ===
N = 10
top_cats = [c for c, _ in Counter(business_categories).most_common(N)]
cat2label = {c: i for i, c in enumerate(top_cats)}
cat2label['other'] = N

labels = [cat2label.get(cat, N) for cat in business_categories]
y = torch.tensor(labels, dtype=torch.long)
data['business'].y = y

# === 训练集划分 ===
num_nodes = y.size(0)
perm = torch.randperm(num_nodes)
train_end = int(0.7 * num_nodes)
val_end = int(0.85 * num_nodes)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[perm[:train_end]] = True
val_mask[perm[train_end:val_end]] = True
test_mask[perm[val_end:]] = True

data['business'].train_mask = train_mask
data['business'].val_mask = val_mask
data['business'].test_mask = test_mask

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
if user_edges:
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
if tip_feats:
    data['tip'].x = torch.tensor(tip_feats, dtype=torch.float)
    data['user', 'tips', 'tip'].edge_index = torch.tensor(u2tip, dtype=torch.long).t().contiguous()
    data['tip', 'about', 'business'].edge_index = torch.tensor(tip2b, dtype=torch.long).t().contiguous()

# === 6. checkin 节点与边 checkin -> business ===
checkin_map = {}
checkin_feats = []
c2b = []
with open(ROOT + files['checkin'], 'r', encoding='utf8') as f:
    for line in tqdm(f, desc='Checkin'):
        c = json.loads(line)
        bid = c['business_id']
        if bid in business_map:
            cid = len(checkin_map)
            checkin_map[bid] = cid
            date_str = c.get('date', '')
            total_checkins = len(date_str.split(', ')) if date_str else 0
            checkin_feats.append([total_checkins])
            c2b.append([cid, business_map[bid]])
if checkin_feats:
    data['checkin'].x = torch.tensor(checkin_feats, dtype=torch.float)
    data['checkin', 'at', 'business'].edge_index = torch.tensor(c2b, dtype=torch.long).t().contiguous()

# === 特征统一映射为 64 维嵌入 ===
for node_type in data.node_types:
    in_dim = data[node_type].x.size(-1)
    proj = Linear(in_dim, embed_size)
    data[node_type].x = proj(data[node_type].x)

# === 保存 ===
torch.save(data, './datasets/Yelp JSON/yelp_full_hetero_labeled_64.pt')
print('✅ 保存完成: yelp_full_hetero_labeled_64.pt')
print(data)
