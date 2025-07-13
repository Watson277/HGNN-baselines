import sys
import os
# 获取当前文件的上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from datasets.load_acm import load_acm
import torch
from torch_geometric.utils import to_undirected

def generate_meta_path_edge_index_from_rel(data, meta_path):
    start_type = meta_path[0][0]
    end_type = meta_path[-1][-1]
    assert start_type == end_type, "元路径首尾节点类型必须一致"

    edge_index = data.edge_index_dict[meta_path[0]]
    cur_src, cur_dst = edge_index
    mapping = {}
    for i in range(cur_src.size(0)):
        s = cur_src[i].item()
        d = cur_dst[i].item()
        mapping.setdefault(d, []).append(s)

    for triple in meta_path[1:]:
        edge_index = data.edge_index_dict[triple]
        cur_src, cur_dst = edge_index
        new_mapping = {}
        for i in range(cur_src.size(0)):
            s = cur_src[i].item()
            d = cur_dst[i].item()
            if s in mapping:
                for start_node in mapping[s]:
                    new_mapping.setdefault(d, []).append(start_node)
        mapping = new_mapping

    final_src = []
    final_dst = []
    for dst_node, src_nodes in mapping.items():
        for src_node in src_nodes:
            final_src.append(src_node)
            final_dst.append(dst_node)

    edge_index = torch.tensor([final_src, final_dst], dtype=torch.long)
    return to_undirected(edge_index)

def compute_homophily(edge_index, y):
    src, dst = edge_index
    same_label = (y[src] == y[dst])
    return same_label.sum().item() / edge_index.size(1)

def generate_metapaths(metadata, center_type, max_hops=2):
    """
    自动生成从 center_type 出发并返回 center_type 的所有合法 n 跳元路径
    n 必须为偶数（比如 2, 4）
    """
    _, edge_types = metadata
    meta_paths = []

    # 从起点出发的边
    start_edges = [e for e in edge_types if e[0] == center_type]

    def dfs(path, hops_left):
        if hops_left == 0:
            if path[-1][2] == center_type:  # 最终回到起点
                meta_paths.append(path)
            return
        last_dst = path[-1][2]
        next_edges = [e for e in edge_types if e[0] == last_dst]
        for e in next_edges:
            dfs(path + [e], hops_left - 1)

    for edge in start_edges:
        dfs([edge], max_hops - 1)

    return meta_paths


if __name__ == '__main__':
    data = load_acm()
    # data = torch.load('./datasets/Yelp JSON/yelp.pt')
    print(data.metadata())
    target_node_type = 'paper'

    meta_paths = generate_metapaths(data.metadata(), center_type=target_node_type)
    if not meta_paths:
        meta_paths = data.metadata()
        print(type(meta_paths))

    for path in meta_paths:
        # print(path)
        try:
            edge_index = generate_meta_path_edge_index_from_rel(data, path)
            homophily = compute_homophily(edge_index, data[target_node_type].y)
            print(f"{path}: 同配率 = {homophily:.4f}")
        except Exception as e:
            print(f"{path}: 计算失败 -> {e}")


