# merge_dual_annotators.py
# =============================================================================
# 双标注员数据合并脚本（重新设计版）
#
# 功能：
#   1. 按 node_id / (start_id,end_id) 合并两位标注员的数据
#   2. 互补填充 m1/m2 方向数据（标注员 1 标 m2，标注员 2 标 m1）
#   3. 奇异点：DBSCAN 聚类 + 最大值筛选
#   4. 应力线 (is_on_psl)：保留两列，分别对应两套应力线
#   5. 边应力线 (close_to_psl)：自适应阈值 + 连通分量 → 二值判断
#
# 用法：
#   python merge_dual_annotators.py                     # 处理全部案例
#   python merge_dual_annotators.py --case 1            # 仅处理案例 1
#   python merge_dual_annotators.py --visualize         # 处理后自动可视化
# =============================================================================

import os, sys, re, argparse
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# 配置
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'valid_dataset')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'merged_data')

# DBSCAN 参数
DBSCAN_EPS = 0.05        # 聚类半径（归一化坐标）
DBSCAN_MIN_SAMPLES = 1   # 最小样本数

# PSL 脊线检测参数（基于节点概率场）
PSL_STRICT_PTILE = 99.5   # 核心脊线节点：概率最高的前 N% 节点
PSL_LOOSE_PTILE = 98.0    # 候选脊线节点：概率最高的前 N% 节点
PSL_MIN_EDGES = 3          # PSL 最少边数（连通分量过滤）
PSL_BRIDGE_HOPS = 3        # 桥接最大跳数：允许在原始图中 ≤N 步内连接碎片
PSL_BRIDGE_MIN_PROB = 0.3  # 桥接路径上节点的最低平均概率
BOUNDARY_EPS = 0.005       # 边界判定阈值：|x|≤eps 或 |x-1|≤eps 或 |y|≤eps 或 |y-1|≤eps


# =============================================================================
# 辅助函数
# =============================================================================

def extract_group_name(filename):
    """
    从文件名提取组名。
    '1.1 nodes.csv' → '1'
    'jia1-s3-1 edges.csv' → 'jia1-s3'
    'mingli-s1-01-1 nodes.csv' → 'mingli-s1-01'
    """
    base = os.path.splitext(filename)[0].split(' ')[0]
    if '-' in base:
        group = '-'.join(base.split('-')[:-1])
    else:
        group = base.split('.')[0]
    return group


def parse_file_pairs(files):
    """将文件按组名配对，返回 {group: {1: path, 2: path}}"""
    pairs = defaultdict(dict)
    errors = []
    for f in files:
        group = extract_group_name(f)
        basename = f.split(' ')[0]
        if '-' in basename:
            version = basename.split('-')[-1]
        else:
            version = basename.split('.')[-1]
        pairs[group][version] = f

    valid = {}
    for group, versions in pairs.items():
        if '1' in versions and '2' in versions:
            valid[group] = versions
        else:
            errors.append(f"Group '{group}': missing version, have {list(versions.keys())}")
    return valid, errors


def safe_float(val):
    """安全转换为 float，'/' 或空值返回 NaN"""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ('/', '', 'nan', 'NaN'):
        return np.nan
    return float(s.replace('.', '', 1) if s.endswith('.') else s)  # 处理 '-c.' 等情况


def clean_type(val):
    """清洗 m1_type/m2_type：去掉尾部句点，统一为 '+t' 或 '-c'"""
    s = str(val).strip().replace('.', '').lower()
    if 't' in s:
        return '+t'
    elif 'c' in s:
        return '-c'
    return s


# =============================================================================
# 节点合并
# =============================================================================

def merge_nodes(file1, file2, group_name):
    """
    合并两位标注员的节点数据。

    策略：
    - m1 数据：优先取标注员 2（通常 m1 有值），缺失时取标注员 1
    - m2 数据：优先取标注员 1（通常 m2 有值），缺失时取标注员 2
    - is_singularity：DBSCAN 聚类后取每簇最大概率值
    - is_on_psl：保留两列 (is_on_psl_1, is_on_psl_2)
    """
    df1 = pd.read_csv(file1, keep_default_na=False, na_values=[''])
    df2 = pd.read_csv(file2, keep_default_na=False, na_values=[''])

    # 确保 node_id 一致
    ids1 = set(df1['node_id'])
    ids2 = set(df2['node_id'])
    common = ids1 & ids2
    if len(common) < len(ids1) or len(common) < len(ids2):
        print(f"  [!] node_id mismatch: {len(ids1)} vs {len(ids2)}, common={len(common)}")

    # 按 node_id 对齐
    df1 = df1.set_index('node_id').reindex(sorted(common)).reset_index()
    df2 = df2.set_index('node_id').reindex(sorted(common)).reset_index()

    result = pd.DataFrame()
    result['node_id'] = df1['node_id']
    result['x'] = df1['x']
    result['y'] = df1['y']
    result['z'] = df1['z']

    # ---- m1 数据：优先取 2（通常 2 有 m1，1 的 m1 是 / ）----
    for col in ['m1_vx', 'm1_vy', 'm1_type', 'm1_abs']:
        vals1 = df1[col].apply(safe_float if col in ('m1_vx', 'm1_vy', 'm1_abs') else clean_type)
        vals2 = df2[col].apply(safe_float if col in ('m1_vx', 'm1_vy', 'm1_abs') else clean_type)
        # 优先取 2（非 NaN），回退到 1
        result[col] = np.where(vals2.notna(), vals2, vals1)
        # 仍未填充的设为 0.0（数值列）或保持（类型列）
        if col in ('m1_vx', 'm1_vy', 'm1_abs'):
            result[col] = result[col].fillna(0.0)

    # ---- m2 数据：优先取 1（通常 1 有 m2，2 的 m2 是 / ）----
    for col in ['m2_vx', 'm2_vy', 'm2_type', 'm2_abs']:
        vals1 = df1[col].apply(safe_float if col in ('m2_vx', 'm2_vy', 'm2_abs') else clean_type)
        vals2 = df2[col].apply(safe_float if col in ('m2_vx', 'm2_vy', 'm2_abs') else clean_type)
        result[col] = np.where(vals1.notna(), vals1, vals2)
        if col in ('m2_vx', 'm2_vy', 'm2_abs'):
            result[col] = result[col].fillna(0.0)

    # ---- is_singularity：DBSCAN 聚类最大值 ----
    result['is_singularity_raw_1'] = pd.to_numeric(df1['is_singularity'], errors='coerce').fillna(0)
    result['is_singularity_raw_2'] = pd.to_numeric(df2['is_singularity'], errors='coerce').fillna(0)
    # 均值作为聚类输入
    prob_avg = (result['is_singularity_raw_1'] + result['is_singularity_raw_2']) / 2
    result['_prob_avg'] = prob_avg
    result['is_singularity'] = assign_final_singularities(result, '_prob_avg')
    result.drop(columns=['_prob_avg'], inplace=True)

    # ---- is_on_psl：保留原始概率值（后续用于 PSL 脊线检测） ----
    result['is_on_psl_raw_1'] = pd.to_numeric(df1['is_on_psl'], errors='coerce').fillna(0)
    result['is_on_psl_raw_2'] = pd.to_numeric(df2['is_on_psl'], errors='coerce').fillna(0)
    # 初始化二值列（稍后由 PSL 检测结果填充）
    result['is_on_psl_1'] = 0
    result['is_on_psl_2'] = 0

    # ---- 检查完整性 ----
    check_completeness(result, group_name)

    return result


def assign_final_singularities(df, prob_col):
    """
    DBSCAN 聚类 → 每簇保留最大概率节点为奇异点。
    """
    prob = df[prob_col].values
    node_ids = df['node_id'].values
    x = df['x'].values
    y = df['y'].values

    for threshold in [0.5, 0.3, 0.1]:
        mask = prob > threshold
        if mask.sum() > 0:
            break

    if mask.sum() == 0:
        return pd.Series(0, index=df.index, dtype=int)

    cand_ids = node_ids[mask]
    cand_prob = prob[mask]
    cand_coords = np.column_stack([x[mask], y[mask]])

    if len(cand_ids) == 1:
        final = set(cand_ids)
        return pd.Series(df['node_id'].isin(final).astype(int).values, index=df.index)

    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(cand_coords)
    labels = clustering.labels_

    final_ids = set()
    for cluster_id in set(labels):
        c_mask = labels == cluster_id
        c_probs = cand_prob[c_mask]
        best_local_idx = np.argmax(c_probs)
        # 找到原始索引
        local_indices = np.where(mask)[0]
        cluster_local_indices = local_indices[c_mask]
        best_node_id = node_ids[cluster_local_indices[best_local_idx]]
        final_ids.add(best_node_id)

    result = np.isin(node_ids, list(final_ids)).astype(int)
    return pd.Series(result, index=df.index)


def check_completeness(df, group_name):
    """检查合并后是否仍有空值"""
    missing = {}
    for col in ['m1_vx', 'm1_vy', 'm2_vx', 'm2_vy']:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            missing[col] = n_missing
    if missing:
        print(f"  [!] {group_name}: still has NaN in {missing}")


# =============================================================================
# 边合并 + PSL 准确判断
# =============================================================================

def merge_edges(file1, file2, group_name, node_df):
    """
    合并边数据，并对 close_to_psl 做二值判断。

    边匹配：通过 (start_id, end_id) 元组配对。

    PSL 准确判断算法：
    1. 取 close_to_psl > PSL_HIGH_PROB 的边作为候选
    2. 计算候选边的中点坐标
    3. DBSCAN 空间聚类 → 识别不同的 PSL 路径
    4. 每簇内做连通分量分析 → 过滤孤立噪声边（< PSL_MIN_CLUSTER）
    5. 保留的边标记为 is_psl = 1
    """
    df1 = pd.read_csv(file1, keep_default_na=False, na_values=[''])
    df2 = pd.read_csv(file2, keep_default_na=False, na_values=[''])

    # 创建 (start, end) 键用于对齐
    keys1 = list(zip(df1['start_id'].astype(int), df1['end_id'].astype(int)))
    keys2 = list(zip(df2['start_id'].astype(int), df2['end_id'].astype(int)))
    df1 = df1.copy()
    df2 = df2.copy()
    df1['_key'] = keys1
    df2['_key'] = keys2

    # 检查拓扑一致性
    keys1 = set(df1['_key'])
    keys2 = set(df2['_key'])
    if keys1 != keys2:
        only1 = len(keys1 - keys2)
        only2 = len(keys2 - keys1)
        print(f"  [!] {group_name} edges: {only1} only in 1, {only2} only in 2")

    # 按 key 对齐
    df1 = df1.set_index('_key')
    df2 = df2.set_index('_key')
    common_keys = sorted(keys1 & keys2)
    df1 = df1.reindex(common_keys)
    df2 = df2.reindex(common_keys)

    result = pd.DataFrame()
    result['edge_id'] = range(len(common_keys))
    result['start_id'] = [k[0] for k in common_keys]
    result['end_id'] = [k[1] for k in common_keys]

    # 保留原始概率值
    result['close_to_psl_1'] = pd.to_numeric(df1['close_to_psl'], errors='coerce').fillna(0).values
    result['close_to_psl_2'] = pd.to_numeric(df2['close_to_psl'], errors='coerce').fillna(0).values

    # PSL 二值判断暂不在此处进行（稍后在 process_case 中用节点概率场检测）
    result['is_psl_1'] = 0
    result['is_psl_2'] = 0

    return result


def determine_psl_edges_from_nodes(node_df, edge_df, node_prob_col):
    """
    基于节点 is_on_psl 概率场 + 边 close_to_psl 概率，通过脊线检测识别 PSL 边。

    算法（节点定位 + 边筛选 两步法）：
    ┌─────────────────────────────────────────────────────┐
    │ Step 1-2: 节点概率 → 确定应力线所在区域（"哪里"）    │
    │ Step 3:   边概率   → 区域中筛选应力线边（"哪些边"）  │
    │ Step 4:   连通分量 → 保证连贯性                      │
    │ Step 5:   桥接     → 填补小间隙                      │
    └─────────────────────────────────────────────────────┘
    """
    node_prob = pd.to_numeric(node_df[node_prob_col], errors='coerce').fillna(0).values
    nonzero_prob = node_prob[node_prob > 1e-6]

    if len(nonzero_prob) < PSL_MIN_EDGES:
        return np.zeros(len(edge_df), dtype=int)

    # ---- Step 1: 节点双阈值 → 应力线区域 ----
    T_strict = np.percentile(node_prob, PSL_STRICT_PTILE)
    T_loose = np.percentile(node_prob, PSL_LOOSE_PTILE)
    if T_loose < 1e-6:
        T_loose = np.percentile(nonzero_prob, 80)
    if T_strict < 1e-6:
        T_strict = np.percentile(nonzero_prob, 95)

    core_nodes = set(node_df['node_id'][node_prob >= T_strict].values)
    region_nodes = set(node_df['node_id'][node_prob >= T_loose].values)

    if len(region_nodes) < PSL_MIN_EDGES:
        return np.zeros(len(edge_df), dtype=int)

    # ---- Step 2: 在区域子图内用边概率筛选 PSL 边 ----
    starts = edge_df['start_id'].values
    ends = edge_df['end_id'].values

    # 确定是对应哪个边概率列
    edge_prob_col = 'close_to_psl_1' if '1' in node_prob_col else 'close_to_psl_2'
    edge_prob = edge_df[edge_prob_col].values if edge_prob_col in edge_df.columns else np.zeros(len(edge_df))

    # 收集区域内的边及其边概率
    region_edge_indices = []
    region_edge_probs = []
    for ei in range(len(edge_df)):
        s, e = starts[ei], ends[ei]
        if s in region_nodes and e in region_nodes:
            region_edge_indices.append(ei)
            region_edge_probs.append(edge_prob[ei])

    if len(region_edge_indices) < PSL_MIN_EDGES:
        return np.zeros(len(edge_df), dtype=int)

    # 自适应边概率阈值：取区域内边概率的中位数
    region_edge_probs = np.array(region_edge_probs)
    edge_threshold = np.median(region_edge_probs[region_edge_probs > 1e-6]) if (region_edge_probs > 1e-6).any() else 0.0
    # 至少保证一定数量的边被选中
    if edge_threshold < 1e-6:
        edge_threshold = np.percentile(region_edge_probs, 50)

    # 选出 PSL 候选边：边概率 >= edge_threshold
    candidate_edges = []
    adj = defaultdict(set)
    for ei in region_edge_indices:
        if edge_prob[ei] >= edge_threshold:
            s, e = starts[ei], ends[ei]
            adj[s].add(e)
            adj[e].add(s)
            candidate_edges.append(ei)

    if len(candidate_edges) < PSL_MIN_EDGES:
        return np.zeros(len(edge_df), dtype=int)

    # ---- Step 3: 连通分量分析 ----
    visited = set()
    components = []

    for node in region_nodes:
        if node not in visited and node in adj:
            comp_nodes = set()
            queue = [node]
            visited.add(node)
            while queue:
                u = queue.pop(0)
                comp_nodes.add(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)

            comp_edges = []
            for ei in candidate_edges:
                s, e = starts[ei], ends[ei]
                if s in comp_nodes and e in comp_nodes:
                    comp_edges.append(ei)

            if comp_edges:
                components.append((comp_nodes, comp_edges))

    # ---- Step 4: 过滤 + 标记 PSL ----
    is_psl = np.zeros(len(edge_df), dtype=int)
    kept_components = []

    for comp_nodes, comp_edges in components:
        has_core = bool(comp_nodes & core_nodes)
        is_large_enough = len(comp_edges) >= PSL_MIN_EDGES

        if has_core and is_large_enough:
            for ei in comp_edges:
                is_psl[ei] = 1
            kept_components.append(comp_nodes)

    # ---- Step 5: 桥接碎片 ----
    if PSL_BRIDGE_HOPS > 0 and len(kept_components) > 1:
        full_adj = defaultdict(list)
        for ei in range(len(edge_df)):
            s, e = starts[ei], ends[ei]
            full_adj[s].append((e, ei))
            full_adj[e].append((s, ei))

        merged = list(kept_components)
        changed = True
        while changed:
            changed = False
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    if i >= len(merged) or j >= len(merged):
                        break
                    path_edges, avg_prob = find_bridge_path(
                        merged[i], merged[j], full_adj, node_prob,
                        node_df['node_id'].values, PSL_BRIDGE_HOPS
                    )
                    if path_edges is not None and avg_prob >= PSL_BRIDGE_MIN_PROB:
                        for ei in path_edges:
                            is_psl[ei] = 1
                        merged[i] = merged[i] | merged[j]
                        merged.pop(j)
                        changed = True
                        break
                if changed:
                    break

    return is_psl


def find_bridge_path(comp_a, comp_b, full_adj, node_prob, node_ids, max_hops):
    """
    在原始图中寻找两个连通分量之间的最短桥接路径。

    使用 BFS，路径代价 = 途径节点的 (1-P) 之和。
    仅返回平均节点概率 ≥ PSL_BRIDGE_MIN_PROB 的路径。

    Returns:
        (path_edges, avg_prob) 或 (None, 0) 如果不可达
    """
    from collections import deque

    prob_map = {nid: p for nid, p in zip(node_ids, node_prob)}

    # BFS 从分量 A 的所有节点出发
    # queue: (node, path_edges, path_nodes, sum_1_minus_p)
    queue = deque()
    for n in comp_a:
        queue.append((n, [], [n], 1.0 - prob_map.get(n, 0)))

    visited_nodes = set(comp_a)
    best_result = None
    best_avg = 0.0

    while queue:
        u, edges, nodes_on_path, cost = queue.popleft()

        if len(edges) >= max_hops:
            continue

        for v, ei in full_adj.get(u, []):
            if v in visited_nodes:
                continue

            new_edges = edges + [ei]
            new_nodes = nodes_on_path + [v]
            new_cost = cost + (1.0 - prob_map.get(v, 0))

            if v in comp_b:
                # 到达 B 分量
                avg = 1.0 - new_cost / len(new_nodes)
                if avg > best_avg:
                    best_avg = avg
                    best_result = new_edges
                continue

            visited_nodes.add(v)
            queue.append((v, new_edges, new_nodes, new_cost))

    return best_result, best_avg


def prune_dead_branches(edge_df, is_psl, node_df, boundary_eps=BOUNDARY_EPS):
    """
    迭代剥离 PSL 图中的死分支。

    规则：应力线必须（满足其一即可）：
      A) 端点落在楼板边界上 (x≈0, x≈1, y≈0, y≈1)
      B) 端点落在奇异点上 (is_singularity=1)
      C) 形成闭环（分量内所有节点度≥2）

    算法：迭代移除不满足以上条件的度=1 节点及其关联边，
    直到所有度=1 节点均为有效端点或图中无度=1 节点。
    """
    is_psl = is_psl.copy()
    starts = edge_df['start_id'].values
    ends = edge_df['end_id'].values
    x = node_df['x'].values
    y = node_df['y'].values
    is_sing = node_df['is_singularity'].values if 'is_singularity' in node_df.columns else np.zeros(len(node_df))
    node_id_to_idx = {nid: i for i, nid in enumerate(node_df['node_id'].values)}
    node_idx_to_id = {i: nid for nid, i in node_id_to_idx.items()}

    def is_boundary(node_id):
        idx = node_id_to_idx.get(node_id)
        if idx is None:
            return False
        return (x[idx] <= boundary_eps or x[idx] >= 1.0 - boundary_eps or
                y[idx] <= boundary_eps or y[idx] >= 1.0 - boundary_eps)

    def is_singularity(node_id):
        idx = node_id_to_idx.get(node_id)
        if idx is None:
            return False
        return is_sing[idx] == 1

    def is_valid_endpoint(node_id):
        return is_boundary(node_id) or is_singularity(node_id)

    # 迭代剥离
    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1

        # 构建当前 PSL 子图
        adj = defaultdict(set)
        edge_of = {}  # (u,v) → edge_idx in edge_df
        for ei in np.where(is_psl == 1)[0]:
            s, e = starts[ei], ends[ei]
            adj[s].add(e)
            adj[e].add(s)
            edge_of[(min(s, e), max(s, e))] = ei

        if not adj:
            break

        # 收集要移除的边
        edges_to_remove = set()

        # ① 移除无效的度=1 端点
        for node, neighbors in list(adj.items()):
            if len(neighbors) == 1:
                if not is_valid_endpoint(node):
                    # 移除该端点及其唯一边
                    nbr = list(neighbors)[0]
                    ei = edge_of.get((min(node, nbr), max(node, nbr)))
                    if ei is not None:
                        edges_to_remove.add(ei)

        # ② 移除没有有效端点的孤立连通分量
        visited = set()
        for node in list(adj.keys()):
            if node in visited:
                continue
            # BFS 收集分量
            comp = set()
            q = [node]
            visited.add(node)
            while q:
                u = q.pop(0)
                comp.add(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)

            # 判定分量有效性
            has_valid = any(is_valid_endpoint(n) for n in comp)
            is_loop = all(len(adj[n]) >= 2 for n in comp)

            if not has_valid and not is_loop:
                # 无效分量 → 移除所有边
                for n in comp:
                    for nbr in adj[n]:
                        ei = edge_of.get((min(n, nbr), max(n, nbr)))
                        if ei is not None:
                            edges_to_remove.add(ei)

        if edges_to_remove:
            for ei in edges_to_remove:
                is_psl[ei] = 0
            changed = True

    return is_psl


def convert_node_psl_to_binary(node_df, edge_df, psl_edge_col, psl_node_col_out):
    """
    根据已确定的 PSL 边，将节点 is_on_psl 从概率值转为二值。

    规则：若节点是任意 is_psl=1 的边的端点，则该节点在应力线上。
    """
    psl_edges = edge_df[edge_df[psl_edge_col] == 1]
    psl_nodes = set(psl_edges['start_id'].values) | set(psl_edges['end_id'].values)
    node_df[psl_node_col_out] = node_df['node_id'].isin(psl_nodes).astype(int)
    return node_df


# =============================================================================
# 主流程
# =============================================================================

def process_case(group_name, versions, output_dir):
    """处理单个案例的合并"""
    file1_nodes = os.path.join(DATA_DIR, versions['1'] + ' nodes.csv'
                                if 'nodes' not in versions['1']
                                else versions['1'])
    file2_nodes = os.path.join(DATA_DIR, versions['2'] + ' nodes.csv'
                                if 'nodes' not in versions['2']
                                else versions['2'])

    # 自动补全文件名
    if not os.path.exists(file1_nodes):
        if 'nodes' not in versions['1']:
            file1_nodes = os.path.join(DATA_DIR, versions['1'])

    # 构建实际文件路径
    f1_nodes = os.path.join(DATA_DIR,
                             next(f for f in [versions['1'] + ' nodes.csv',
                                              versions['1']]
                                  if os.path.exists(os.path.join(DATA_DIR, f))))
    f2_nodes = os.path.join(DATA_DIR,
                             next(f for f in [versions['2'] + ' nodes.csv',
                                              versions['2']]
                                  if os.path.exists(os.path.join(DATA_DIR, f))))
    # 边文件同理
    f1_nodes_candidates = [versions['1'] + ' nodes.csv', versions['1']]
    f2_nodes_candidates = [versions['2'] + ' nodes.csv', versions['2']]
    f1_nodes = os.path.join(DATA_DIR, next(f for f in f1_nodes_candidates
                                            if os.path.exists(os.path.join(DATA_DIR, f))))
    f2_nodes = os.path.join(DATA_DIR, next(f for f in f2_nodes_candidates
                                            if os.path.exists(os.path.join(DATA_DIR, f))))

    f1_edges = f1_nodes.replace(' nodes', ' edges')
    f2_edges = f2_nodes.replace(' nodes', ' edges')

    print(f'\n{"="*60}')
    print(f'Processing: {group_name}')
    print(f'  Nodes: {os.path.basename(f1_nodes)} + {os.path.basename(f2_nodes)}')
    print(f'  Edges: {os.path.basename(f1_edges)} + {os.path.basename(f2_edges)}')

    # 合并节点（保留 is_on_psl 原始概率值 + 奇异点原始概率值）
    node_df = merge_nodes(f1_nodes, f2_nodes, group_name)

    # 合并边（仅保留 close_to_psl 原始概率值）
    edge_df = merge_edges(f1_edges, f2_edges, group_name, node_df)

    # ★ 基于节点 is_on_psl 概率场检测 PSL 边（图脊线追踪）
    print(f'  PSL detection (node probability field):')
    for set_name, node_col, edge_col in [
        ('Set1', 'is_on_psl_raw_1', 'is_psl_1'),
        ('Set2', 'is_on_psl_raw_2', 'is_psl_2')
    ]:
        # 计算自适应阈值
        prob = node_df[node_col].values
        nonzero = prob[prob > 1e-6]
        T_s = np.percentile(prob, PSL_STRICT_PTILE)
        T_l = np.percentile(prob, PSL_LOOSE_PTILE)
        n_core = (prob >= T_s).sum()
        n_cand = (prob >= T_l).sum()
        print(f'    {set_name}: T_strict(P{PSL_STRICT_PTILE})={T_s:.4f} ({n_core} core nodes), '
              f'T_loose(P{PSL_LOOSE_PTILE})={T_l:.4f} ({n_cand} cand nodes)')

        is_psl = determine_psl_edges_from_nodes(node_df, edge_df, node_col)
        n_before = is_psl.sum()

        # ★ 修剪死分支：移除未到达边界/奇异点/未闭环的游离分支
        edge_df[edge_col] = is_psl  # 暂存以便 pruning 使用
        is_psl_pruned = prune_dead_branches(edge_df, is_psl, node_df)
        edge_df[edge_col] = is_psl_pruned
        n_after = is_psl_pruned.sum()
        print(f'      → {n_before} edges detected, {n_before - n_after} pruned ({n_after} final)')

    # ★ 根据已确定的 PSL 边，将节点 is_on_psl 从概率值转为二值
    node_df = convert_node_psl_to_binary(node_df, edge_df, 'is_psl_1', 'is_on_psl_1')
    node_df = convert_node_psl_to_binary(node_df, edge_df, 'is_psl_2', 'is_on_psl_2')

    # 统计
    n_psl1 = edge_df['is_psl_1'].sum()
    n_psl2 = edge_df['is_psl_2'].sum()
    n_sing = node_df['is_singularity'].sum()
    n_node_psl1 = node_df['is_on_psl_1'].sum()
    n_node_psl2 = node_df['is_on_psl_2'].sum()
    print(f'  Result: {len(node_df)} nodes, {len(edge_df)} edges')
    print(f'  Singularities: {n_sing}')
    print(f'  PSL edges: {n_psl1} (set1) + {n_psl2} (set2)')
    print(f'  PSL nodes (binary): {n_node_psl1} (set1) + {n_node_psl2} (set2)')

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    node_out = os.path.join(output_dir, f'{group_name}_merged_nodes.csv')
    edge_out = os.path.join(output_dir, f'{group_name}_merged_edges.csv')
    node_df.to_csv(node_out, index=False)
    edge_df.to_csv(edge_out, index=False)
    print(f'  Saved: {os.path.basename(node_out)}, {os.path.basename(edge_out)}')

    return node_df, edge_df


def main():
    parser = argparse.ArgumentParser(description='Merge dual-annotator CSV data')
    parser.add_argument('--case', type=str, default=None,
                        help='Process single case (e.g. "1" or "jia1-s3")')
    parser.add_argument('--visualize', action='store_true',
                        help='Run visualization after merge')
    parser.add_argument('--input-dir', type=str, default=DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    print(f'Input:  {args.input_dir}')
    print(f'Output: {args.output_dir}')

    # 收集文件
    all_files = glob(os.path.join(args.input_dir, '*.csv'))
    nodes_files = [os.path.basename(f) for f in all_files if 'nodes' in os.path.basename(f)]

    pairs, errors = parse_file_pairs(nodes_files)
    if errors:
        print('\nPairing errors:')
        for e in errors:
            print(f'  {e}')

    print(f'\nFound {len(pairs)} valid case pairs')

    if args.case:
        # 单案例模式
        if args.case in pairs:
            process_case(args.case, pairs[args.case], args.output_dir)
        else:
            print(f'Case "{args.case}" not found. Available: {sorted(pairs.keys())[:20]}...')
            return
    else:
        # 全量模式
        processed = 0
        for group_name, versions in sorted(pairs.items()):
            try:
                process_case(group_name, versions, args.output_dir)
                processed += 1
            except Exception as e:
                print(f'  [!] Error processing {group_name}: {e}')
                import traceback
                traceback.print_exc()
        print(f'\n{"="*60}')
        print(f'Done! {processed}/{len(pairs)} cases processed.')
        print(f'Output: {os.path.abspath(args.output_dir)}/')

    # 可视化
    if args.visualize:
        from visualize_merge_check import visualize_merge
        # 取第一个案例做可视化
        first_group = args.case or sorted(pairs.keys())[0]
        visualize_merge(first_group, pairs[first_group], args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
