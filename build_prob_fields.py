# build_prob_fields.py — 基于距离的奇异点/PSL 连续概率场生成
# =============================================================================
# 输入: merged_data CSV（含 is_singularity 和 is_psl_1/2）
# 输出: 新的 .pt 数据集（y_node = 奇异点概率场, y_edge = PSL 概率场）
#
# 公式: P(v) = max_{s in sources} exp(-d(v,s)^2 / (2*sigma^2))
#   d = Dijkstra 图最短路径距离（近似测地线距离）
#   sigma_sing ≈ 5% 归一化边长, sigma_psl ≈ 2%
# =============================================================================

import os, sys, torch, numpy as np, pandas as pd
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import heapq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- 参数 ----
SIGMA_SING = 0.04       # 奇异点衰减距离（归一化坐标）
SIGMA_PSL = 0.015       # PSL 边衰减距离
MAX_DIST_SING = 0.15    # 奇异点最大影响距离（超过截断为 0）
MAX_DIST_PSL = 0.08     # PSL 最大影响距离


def build_graph_adj(node_df, edge_df):
    """构建图邻接表 + 边权（欧氏距离）"""
    node_xy = node_df.set_index('node_id')[['x', 'y']]
    adj = defaultdict(list)  # node_id → [(neighbor_id, distance)]
    edge_midpoints = {}      # edge_idx → (mx, my)
    edge_node_pairs = []     # [(start_id, end_id)]

    for _, row in edge_df.iterrows():
        s, e = int(row['start_id']), int(row['end_id'])
        sx, sy = node_xy.loc[s, 'x'], node_xy.loc[s, 'y']
        ex, ey = node_xy.loc[e, 'x'], node_xy.loc[e, 'y']
        dist = np.sqrt((sx - ex)**2 + (sy - ey)**2)
        adj[s].append((e, dist))
        adj[e].append((s, dist))
        edge_midpoints[len(edge_node_pairs)] = ((sx + ex) / 2, (sy + ey) / 2)
        edge_node_pairs.append((s, e))

    return adj, edge_midpoints, edge_node_pairs


def dijkstra_field(adj, source_nodes, sigma, max_dist):
    """
    Dijkstra 从多个源点出发，计算每个节点的最短距离，
    然后转为概率场: P = exp(-d² / 2σ²)，超过 max_dist 截断为 0。
    """
    dist = {}
    pq = []
    for src in source_nodes:
        if src in adj:
            dist[src] = 0.0
            heapq.heappush(pq, (0.0, src))

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd <= max_dist and nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    prob = {}
    for node_id, d in dist.items():
        if d <= max_dist:
            prob[node_id] = float(np.exp(-d**2 / (2 * sigma**2)))
    return prob


def compute_singularity_field(node_df, edge_df):
    """对每个确定奇异点，计算距离衰减概率场"""
    adj, _, _ = build_graph_adj(node_df, edge_df)
    sing_nodes = node_df[node_df['is_singularity'] == 1]['node_id'].tolist()

    if not sing_nodes:
        # 无奇异点 → 全零
        return np.zeros(len(node_df))

    prob = dijkstra_field(adj, sing_nodes, SIGMA_SING, MAX_DIST_SING)

    node_ids = node_df['node_id'].values
    result = np.array([prob.get(nid, 0.0) for nid in node_ids])
    return result


def compute_psl_field(node_df, edge_df, psl_edge_col='is_psl_1', sigma=SIGMA_PSL):
    """
    对确定 PSL 边的中点，计算每条边到最近 PSL 边中点的距离 → 边概率场。
    """
    psl_edges = edge_df[edge_df[psl_edge_col] == 1]
    if len(psl_edges) == 0:
        return np.zeros(len(edge_df))

    adj, edge_midpoints, edge_node_pairs = build_graph_adj(node_df, edge_df)
    n_edges = len(edge_node_pairs)

    # PSL 源边中点坐标
    psl_midpoints = []
    for _, row in psl_edges.iterrows():
        s, e = int(row['start_id']), int(row['end_id'])
        mx = (node_df.loc[node_df['node_id'] == s, 'x'].values[0] +
              node_df.loc[node_df['node_id'] == e, 'x'].values[0]) / 2
        my = (node_df.loc[node_df['node_id'] == s, 'y'].values[0] +
              node_df.loc[node_df['node_id'] == e, 'y'].values[0]) / 2
        psl_midpoints.append((mx, my))

    psl_midpoints = np.array(psl_midpoints)

    # 计算每条边中点到最近 PSL 边中点的欧氏距离
    all_midpoints = np.array([edge_midpoints[i] for i in range(n_edges)])
    result = np.zeros(n_edges)

    # 分批计算距离（避免大矩阵）
    batch_size = 5000
    for start in range(0, n_edges, batch_size):
        end = min(start + batch_size, n_edges)
        batch = all_midpoints[start:end]  # [B, 2]
        # [B, 1, 2] - [1, P, 2] → [B, P]
        diffs = batch[:, None, :] - psl_midpoints[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        min_dists = dists.min(axis=1)  # [B]
        result[start:end] = np.exp(-min_dists ** 2 / (2 * sigma ** 2))
        # 截断
        result[start:end][min_dists > MAX_DIST_PSL] = 0.0

    return result


def process_merged_data(merged_dir, output_path):
    """批量处理 merged_data 生成含概率场的 .pt"""
    from build_graph_dataset import read_merged_graph, remove_z_and_normalize

    node_files = sorted(glob(os.path.join(merged_dir, '*_merged_nodes.csv')))
    graphs = []

    for nf in tqdm(node_files, desc='Building prob fields'):
        case = os.path.basename(nf).replace('_merged_nodes.csv', '')
        ef = os.path.join(merged_dir, f'{case}_merged_edges.csv')

        node_df = pd.read_csv(nf)
        edge_df = pd.read_csv(ef)

        # 计算概率场
        sing_prob = compute_singularity_field(node_df, edge_df)
        psl1_prob = compute_psl_field(node_df, edge_df, 'is_psl_1')

        # 转为 PyG Data + 替换标签
        data = read_merged_graph(nf, ef)
        data.y_node = torch.tensor(sing_prob, dtype=torch.float).unsqueeze(1)
        # y_edge: 用 PSL1 概率场（单列，兼容当前模型）
        # 双向边：每边概率复制一份
        n_edges = len(edge_df)
        y_edge_full = np.zeros(n_edges * 2)
        for i in range(n_edges):
            y_edge_full[i * 2] = psl1_prob[i]
            y_edge_full[i * 2 + 1] = psl1_prob[i]
        data.y_edge = torch.tensor(y_edge_full, dtype=torch.float).unsqueeze(1)

        graphs.append(data)

    graphs = remove_z_and_normalize(graphs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graphs, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'\nSaved: {output_path} ({len(graphs)} graphs, {size_mb:.1f} MB)')

    # 统计
    total_nnz = sum((d.y_node > 0.01).sum().item() for d in graphs)
    total_enz = sum((d.y_edge > 0.01).sum().item() for d in graphs)
    print(f'  Nodes with prob > 0.01: {total_nnz} (avg {total_nnz/len(graphs):.0f}/graph)')
    print(f'  Edges with prob > 0.01: {total_enz} (avg {total_enz/len(graphs):.0f}/graph)')

    return graphs


if __name__ == '__main__':
    merged_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', '02_merged')
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'datasets', '03_graph', 'merged_25cases_prob.pt')
    process_merged_data(merged_dir, output_path)
