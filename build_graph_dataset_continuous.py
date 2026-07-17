# build_graph_dataset_continuous.py
# =============================================================================
# 基于距离衰减 + m1/m2 PSL 分组的概率分布数据集构建
#
# 策略：
#   1. 奇异点：二值种子 → 节点间欧氏距离 → 高斯衰减 → [N,1]
#   2. PSL 边：二值种子 → 按边方向与 m1/m2 向量对齐度分为两组
#      → 分别计算中点距离场 → 高斯衰减 → [2E,2] (m1_psl, m2_psl)
#
# PSL 分组逻辑：
#   - 对每条 PSL 种子边，计算边方向向量
#   - 取边中点处 m1/m2 向量（两端点平均）
#   - 比较 |dot(edge_dir, m1)| 与 |dot(edge_dir, m2)|
#   - 对齐度更高的归入对应组
# =============================================================================

import os, sys, torch, numpy as np, pandas as pd
from glob import glob
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================= 配置 =============================

SIGMA_SING = 0.03
SIGMA_EDGE = 0.05

# ============================= 工具函数 =============================

def sf(val):
    if pd.isna(val): return 0.0
    s = str(val).strip()
    if s in ('/', '', 'nan', 'NaN'): return 0.0
    return float(s)


def gaussian_kernel(dists, sigma):
    return np.exp(-(dists ** 2) / (2 * sigma ** 2))


def compute_distance_field(points, seed_points):
    """points [M,2], seed_points [K,2] → probs [M]"""
    if len(seed_points) == 0:
        return np.zeros(len(points), dtype=np.float32)

    batch_size = 4096
    all_min = []
    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        chunk = points[start:end]
        diff = chunk[:, None, :] - seed_points[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        all_min.append(dists.min(axis=1))
    return gaussian_kernel(np.concatenate(all_min), SIGMA_EDGE)




# ============================= 图构建 =============================

def read_merged_graph(node_path, edge_path):
    node_df = pd.read_csv(node_path)
    edge_df = pd.read_csv(edge_path)

    node_ids = node_df['node_id'].tolist()
    id_map = {orig_id: local_idx for local_idx, orig_id in enumerate(node_ids)}

    # One-hot 编码
    type_categories = [['+t', '-c']]
    encoder = OneHotEncoder(categories=type_categories, sparse_output=False, handle_unknown='ignore')
    m1_types = node_df['m1_type'].astype(str).str.strip().values.reshape(-1, 1)
    m2_types = node_df['m2_type'].astype(str).str.strip().values.reshape(-1, 1)
    m1_type_oh = encoder.fit_transform(m1_types)
    m2_type_oh = encoder.fit_transform(m2_types)

    # 节点特征 [x, y, z, m1_vx, m1_vy, m1_abs, m1_oh(2), m2_vx, m2_vy, m2_abs, m2_oh(2)] = 13
    feats = []
    coords_list = []

    for idx, row in node_df.iterrows():
        x, y, z = float(row['x']), float(row['y']), float(row['z'])
        coords_list.append([x, y])
        m1vx = sf(row['m1_vx']); m1vy = sf(row['m1_vy']); m1abs = sf(row['m1_abs'])
        m2vx = sf(row['m2_vx']); m2vy = sf(row['m2_vy']); m2abs = sf(row['m2_abs'])
        m1_oh = m1_type_oh[idx].tolist()
        m2_oh = m2_type_oh[idx].tolist()
        feats.append([x, y, z, m1vx, m1vy, m1abs] + m1_oh +
                     [m2vx, m2vy, m2abs] + m2_oh)

    X = torch.tensor(feats, dtype=torch.float)
    coords = np.array(coords_list, dtype=np.float32)

    # 奇异点种子
    sing_binary = node_df['is_singularity'].values.astype(int)
    sing_seeds = np.where(sing_binary == 1)[0].tolist()

    # 边构建 + PSL 分组
    edge_indices = []
    undirected_edges = []
    edge_midpoints_list = []
    m1_seeds = []
    m2_seeds = []

    has_psl1 = 'is_psl_1' in edge_df.columns
    has_psl2 = 'is_psl_2' in edge_df.columns

    for e_idx, (_, row) in enumerate(edge_df.iterrows()):
        u = id_map[int(row['start_id'])]
        v = id_map[int(row['end_id'])]

        edge_indices.append([u, v])
        edge_indices.append([v, u])

        undirected_edges.append([u, v])
        mid_x = (coords[u, 0] + coords[v, 0]) / 2.0
        mid_y = (coords[u, 1] + coords[v, 1]) / 2.0
        edge_midpoints_list.append([mid_x, mid_y])

        # PSL 种子：is_psl_1 → m1 族, is_psl_2 → m2 族
        # 两个标注员分别标注了不同的 PSL 族（m1 主应力线 / m2 主应力线）
        if has_psl1 and int(row.get('is_psl_1', 0)) == 1:
            m1_seeds.append(e_idx)
        if has_psl2 and int(row.get('is_psl_2', 0)) == 1:
            m2_seeds.append(e_idx)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    undirected_edges = np.array(undirected_edges, dtype=np.int64)
    edge_midpoints = np.array(edge_midpoints_list, dtype=np.float32)

    data = Data(x=X, edge_index=edge_index)
    meta = {
        'coords': coords,
        'sing_seeds': sing_seeds,
        'm1_seeds': m1_seeds,
        'm2_seeds': m2_seeds,
        'edge_midpoints': edge_midpoints,
    }
    return data, meta


# ============================= 归一化 =============================

def remove_z_and_normalize(graph_list, meta_list):
    if not graph_list:
        return [], []

    graphs_no_z = []
    for data in graph_list:
        x_no_z = torch.cat([data.x[:, :2], data.x[:, 3:]], dim=1)  # 12维
        graphs_no_z.append(Data(x=x_no_z, edge_index=data.edge_index))

    # 特征归一化
    norm_cols = [2, 3, 4, 7, 8, 9]
    col_min = [float('inf')] * len(norm_cols)
    col_max = [float('-inf')] * len(norm_cols)
    for data in graphs_no_z:
        for i, col in enumerate(norm_cols):
            c = data.x[:, col]
            col_min[i] = min(col_min[i], c.min().item())
            col_max[i] = max(col_max[i], c.max().item())

    for data in graphs_no_z:
        for i, col in enumerate(norm_cols):
            rng = col_max[i] - col_min[i]
            if rng > 1e-8:
                data.x[:, col] = (data.x[:, col] - col_min[i]) / rng
            else:
                data.x[:, col] = 0.0

    # 坐标同步归一化
    all_x = np.concatenate([m['coords'][:, 0] for m in meta_list])
    all_y = np.concatenate([m['coords'][:, 1] for m in meta_list])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    for meta in meta_list:
        c = meta['coords']
        c[:, 0] = (c[:, 0] - x_min) / (x_max - x_min) if x_max - x_min > 1e-8 else 0
        c[:, 1] = (c[:, 1] - y_min) / (y_max - y_min) if y_max - y_min > 1e-8 else 0

        em = meta['edge_midpoints']
        em[:, 0] = (em[:, 0] - x_min) / (x_max - x_min) if x_max - x_min > 1e-8 else 0
        em[:, 1] = (em[:, 1] - y_min) / (y_max - y_min) if y_max - y_min > 1e-8 else 0

    for data in graphs_no_z:
        data.x[:, 0] = (data.x[:, 0] - x_min) / (x_max - x_min) if x_max - x_min > 1e-8 else 0
        data.x[:, 1] = (data.x[:, 1] - y_min) / (y_max - y_min) if y_max - y_min > 1e-8 else 0

    return graphs_no_z, meta_list


# ============================= 入口 =============================

def main():
    merged_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', '02_merged')
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'datasets', '03_graph', 'merged_25cases_continuous.pt')

    node_files = sorted(glob(os.path.join(merged_dir, '*_merged_nodes.csv')))
    print(f'Found {len(node_files)} merged cases')
    print(f'σ_sing={SIGMA_SING}, σ_edge={SIGMA_EDGE}\n')

    # 读取
    graph_structs = []
    total_sing = 0; total_m1 = 0; total_m2 = 0
    for nf in tqdm(node_files, desc='Reading'):
        case = os.path.basename(nf).replace('_merged_nodes.csv', '')
        ef = os.path.join(merged_dir, f'{case}_merged_edges.csv')
        if not os.path.exists(ef):
            print(f'  Missing edges for {case}, skip')
            continue
        try:
            data, meta = read_merged_graph(nf, ef)
            graph_structs.append((data, meta))
            total_sing += len(meta['sing_seeds'])
            total_m1 += len(meta['m1_seeds'])
            total_m2 += len(meta['m2_seeds'])
        except Exception as e:
            print(f'  Error {case}: {e}')

    print(f'\nRead {len(graph_structs)} graphs')
    print(f'Sing seeds: {total_sing} ({total_sing/len(graph_structs):.1f}/graph)')
    print(f'M1-PSL seeds: {total_m1} ({total_m1/len(graph_structs):.0f}/graph)')
    print(f'M2-PSL seeds: {total_m2} ({total_m2/len(graph_structs):.0f}/graph)')

    # 归一化
    data_list = [g[0] for g in graph_structs]
    meta_list = [g[1] for g in graph_structs]
    graphs_no_z, meta_list = remove_z_and_normalize(data_list, meta_list)
    print(f'\nNormalized {len(graphs_no_z)} graphs')

    # 赋值连续概率
    print(f'\nAssigning continuous probabilities...')
    final_graphs = []

    for i, data in enumerate(tqdm(graphs_no_z, desc='Probability')):
        meta = meta_list[i]
        coords_norm = meta['coords']
        edge_mids_norm = meta['edge_midpoints']

        # 节点：奇异点距离场
        sing_probs = compute_distance_field(
            coords_norm,
            coords_norm[meta['sing_seeds']] if meta['sing_seeds'] else np.zeros((0, 2), dtype=np.float32)
        )
        # 对于奇异点用 SIGMA_SING
        if meta['sing_seeds']:
            data.y_node = torch.tensor(sing_probs, dtype=torch.float).unsqueeze(1)
        else:
            data.y_node = torch.zeros((len(coords_norm), 1), dtype=torch.float)

        # 边：m1-PSL 和 m2-PSL 分别计算距离场
        m1_probs = compute_distance_field(
            edge_mids_norm,
            edge_mids_norm[meta['m1_seeds']] if meta['m1_seeds'] else np.zeros((0, 2), dtype=np.float32)
        )
        m2_probs = compute_distance_field(
            edge_mids_norm,
            edge_mids_norm[meta['m2_seeds']] if meta['m2_seeds'] else np.zeros((0, 2), dtype=np.float32)
        )

        # 扩展为双向边 [2E, 2]
        E = len(edge_mids_norm)
        edge_probs = np.stack([
            np.repeat(m1_probs, 2),
            np.repeat(m2_probs, 2),
        ], axis=1)  # [2E, 2]

        data.y_edge = torch.tensor(edge_probs, dtype=torch.float)
        final_graphs.append(data)

    # 统计
    all_node = torch.cat([d.y_node for d in final_graphs])
    all_edge = torch.cat([d.y_edge for d in final_graphs])
    print(f'\nProbability statistics:')
    print(f'  y_node (sing): min={all_node.min():.4f} max={all_node.max():.4f} '
          f'mean={all_node.mean():.4f} >0.01={(all_node>0.01).sum()/all_node.numel()*100:.1f}%')
    print(f'  y_edge[:,0] (m1-PSL): min={all_edge[:,0].min():.4f} max={all_edge[:,0].max():.4f} '
          f'mean={all_edge[:,0].mean():.4f} >0.01={(all_edge[:,0]>0.01).sum()/all_edge[:,0].numel()*100:.1f}%')
    print(f'  y_edge[:,1] (m2-PSL): min={all_edge[:,1].min():.4f} max={all_edge[:,1].max():.4f} '
          f'mean={all_edge[:,1].mean():.4f} >0.01={(all_edge[:,1]>0.01).sum()/all_edge[:,1].numel()*100:.1f}%')

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(final_graphs, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'\nSaved: {output_path} ({size_mb:.1f} MB)')
    print(f'y_node shape: {final_graphs[0].y_node.shape} (singularity prob)')
    print(f'y_edge shape: {final_graphs[0].y_edge.shape} ([m1-PSL, m2-PSL])')


if __name__ == '__main__':
    main()
