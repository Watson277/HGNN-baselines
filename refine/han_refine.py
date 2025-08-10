
import torch
import torch.nn.functional as F

def prune_edges_by_cosine_similarity(data, metapath_names, node_type, threshold=0.2, batch_size=100000):
    x = data[node_type].x  # [num_nodes, feat_dim]

    for metapath in metapath_names:
        edge_type = (node_type, metapath, node_type)
        edge_index = data[edge_type].edge_index  # [2, num_edges]

        src_all, dst_all = edge_index
        num_edges = edge_index.size(1)

        keep_src = []
        keep_dst = []

        print(f"[{metapath}] 总边数: {num_edges}")

        for i in range(0, num_edges, batch_size):
            end = min(i + batch_size, num_edges)
            src_batch = src_all[i:end]
            dst_batch = dst_all[i:end]

            x_src = F.normalize(x[src_batch], dim=1)
            x_dst = F.normalize(x[dst_batch], dim=1)
            sim = (x_src * x_dst).sum(dim=1)

            mask = sim >= threshold
            keep_src.append(src_batch[mask])
            keep_dst.append(dst_batch[mask])

        new_src = torch.cat(keep_src, dim=0)
        new_dst = torch.cat(keep_dst, dim=0)
        new_edge_index = torch.stack([new_src, new_dst], dim=0)

        print(f"[{metapath}] 保留边数: {new_edge_index.size(1)}")

        data[edge_type].edge_index = new_edge_index