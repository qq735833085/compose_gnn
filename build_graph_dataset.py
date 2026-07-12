# build_graph_dataset.py — 将 merged_data/ CSV 批量转为 .pt 图数据集
import os, sys, torch, numpy as np, pandas as pd
from glob import glob
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_merged_graph(node_path, edge_path):
    """读取合并后的节点/边 CSV，转为 PyG Data 对象"""
    node_df = pd.read_csv(node_path)
    edge_df = pd.read_csv(edge_path)

    node_ids = node_df['node_id'].tolist()
    id_map = {orig_id: local_idx for local_idx, orig_id in enumerate(node_ids)}

    # One-hot 编码 m1_type / m2_type
    type_categories = [['+t', '-c']]
    encoder = OneHotEncoder(categories=type_categories, sparse_output=False, handle_unknown='ignore')

    # 清洗类型列（去空格）
    m1_types = node_df['m1_type'].astype(str).str.strip().values.reshape(-1, 1)
    m2_types = node_df['m2_type'].astype(str).str.strip().values.reshape(-1, 1)
    m1_type_oh = encoder.fit_transform(m1_types)
    m2_type_oh = encoder.fit_transform(m2_types)

    # 辅助：安全 float
    def sf(val):
        if pd.isna(val):
            return 0.0
        s = str(val).strip()
        if s in ('/', '', 'nan', 'NaN'):
            return 0.0
        return float(s)

    # 构建节点特征 [x, y, z, m1_vx, m1_vy, m1_abs, m1_oh(2), m2_vx, m2_vy, m2_abs, m2_oh(2)] = 13 维
    feats = []
    for idx, row in node_df.iterrows():
        x, y, z = float(row['x']), float(row['y']), float(row['z'])
        m1_vx = sf(row['m1_vx']); m1_vy = sf(row['m1_vy']); m1_abs = sf(row['m1_abs'])
        m2_vx = sf(row['m2_vx']); m2_vy = sf(row['m2_vy']); m2_abs = sf(row['m2_abs'])
        m1_oh = m1_type_oh[idx].tolist()
        m2_oh = m2_type_oh[idx].tolist()
        feats.append([x, y, z, m1_vx, m1_vy, m1_abs] + m1_oh +
                     [m2_vx, m2_vy, m2_abs] + m2_oh)

    X = torch.tensor(feats, dtype=torch.float)

    # 节点标签: is_singularity (DBSCAN 结果, 0/1 二值) → 回归目标
    y_sing = torch.tensor(node_df['is_singularity'].values, dtype=torch.float).unsqueeze(1)

    # 边拓扑 + 标签（双列：is_psl_1 + is_psl_2）
    edge_indices = []
    edge_labels = []

    has_col1 = 'is_psl_1' in edge_df.columns
    has_col2 = 'is_psl_2' in edge_df.columns

    for _, row in edge_df.iterrows():
        u = id_map[int(row['start_id'])]
        v = id_map[int(row['end_id'])]
        edge_indices.append([u, v])
        edge_indices.append([v, u])  # 双向

        psl1 = float(row['is_psl_1']) if has_col1 else 0.0
        psl2 = float(row['is_psl_2']) if has_col2 else 0.0
        edge_labels.append([psl1, psl2])
        edge_labels.append([psl1, psl2])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    Y_edge = torch.tensor(edge_labels, dtype=torch.float)  # [2E, 2]

    data = Data(x=X, edge_index=edge_index, y_node=y_sing, y_edge=Y_edge)
    return data


def remove_z_and_normalize(graph_list):
    """去 z 列 + 全局 Min-Max 归一化"""
    if not graph_list:
        return []

    # 去 z
    graphs_no_z = []
    for data in graph_list:
        x_no_z = torch.cat([data.x[:, :2], data.x[:, 3:]], dim=1)  # 去掉 idx=2 的 z
        graphs_no_z.append(Data(x=x_no_z, edge_index=data.edge_index,
                                y_node=data.y_node.clone(), y_edge=data.y_edge))

    # 全局归一化 m1/m2 方向+量值列 (新索引: 2,3,4,7,8,9)
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

    # y_node 归一化
    y_min = min(d.y_node.min().item() for d in graphs_no_z if d.y_node.numel() > 0)
    y_max = max(d.y_node.max().item() for d in graphs_no_z if d.y_node.numel() > 0)
    if y_max - y_min > 1e-8:
        for d in graphs_no_z:
            d.y_node = (d.y_node - y_min) / (y_max - y_min)
    else:
        for d in graphs_no_z:
            d.y_node = torch.zeros_like(d.y_node)

    return graphs_no_z


def main():
    merged_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', '02_merged')
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'datasets', '03_graph', 'merged_25cases.pt')

    node_files = sorted(glob(os.path.join(merged_dir, '*_merged_nodes.csv')))
    n_cases = len(node_files)
    print(f'Found {n_cases} merged cases')

    graphs = []
    for nf in tqdm(node_files, desc='Converting'):
        case = os.path.basename(nf).replace('_merged_nodes.csv', '')
        ef = os.path.join(merged_dir, f'{case}_merged_edges.csv')
        if not os.path.exists(ef):
            print(f'  Missing edges for {case}, skip')
            continue
        try:
            data = read_merged_graph(nf, ef)
            graphs.append(data)
        except Exception as e:
            print(f'  Error {case}: {e}')

    print(f'\nConverted {len(graphs)} graphs')

    # 去 z + 归一化
    graphs = remove_z_and_normalize(graphs)
    print(f'After normalization: {len(graphs)} graphs, input_dim={graphs[0].x.shape[1]}')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graphs, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'\nSaved: {output_path} ({size_mb:.1f} MB)')
    print(f'Features: {graphs[0].x.shape[1]} dim, nodes: {graphs[0].x.shape[0]}, edges: {graphs[0].edge_index.shape[1]//2}')


if __name__ == '__main__':
    main()
