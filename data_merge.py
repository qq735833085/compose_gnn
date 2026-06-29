import os
import re
import pandas as pd
import numpy as np
from glob import glob
from sklearn.cluster import DBSCAN

# ==================== 配置 ====================
data_dir = r"D:\composite_0602\initial_dataset"               # 原始csv文件所在目录
output_dir = "./merged_data"      # 输出目录
os.makedirs(output_dir, exist_ok=True)

# 全局错误记录
errors = []

# ==================== 辅助函数 ====================
def extract_group_name(filename):
    """
    从文件名中提取组名（去除末尾的 -1/-2 或 .1/.2 以及可能的空格）
    例如：
        "1.1 edges.csv" -> "1"
        "1.2 nodes.csv" -> "1"
        "jia1-s3-1 edges.csv" -> "jia1-s3"
        "ztt9-s2-2 nodes.csv" -> "ztt9-s2"
        "mingli-s1-01-1 nodes.csv" -> "mingli-s1-01"
    """
    base = os.path.splitext(filename)[0].split(' ')[0]          # 去掉 .csv
    
    if '-' in base:
        group = '-'.join(base.split('-')[:-1])
    else:
        group = base.split('.')[0]

    return group

def parse_file_pairs(files):
    """根据提取的组名将文件配对"""
    pairs = {}
    for f in files:
        group = extract_group_name(f)
        basename = f.split(' ')[0]
        if '-' in basename:
            version = basename.split('-')[-1]
        else:
            version = basename.split('.')[-1]

        if group not in pairs:
            pairs[group] = {}
        pairs[group][version] = f
    # 只保留完整配对的（同时存在1和2）
    valid_pairs = {}
    for group, versions in pairs.items():
        if '1' in versions and '2' in versions:
            valid_pairs[group] = versions
        else:
            errors.append(f"Pairing failed for group '{group}': missing version {set(['1','2']) - set(versions.keys())}")
    return valid_pairs

def assign_final_singularities(df, prob_col='is_singularity', coord_cols=('x', 'y'),
                               node_id_col='node_id', threshold=0.5, eps=0.05, min_samples=1):
    """根据概率和坐标聚类，标记最终奇异点"""
    df_out = df.copy()
    candidates = df_out[df_out[prob_col] > threshold].copy()
    if len(candidates) == 0:
        df_out['is_singularity_final'] = 0
        return df_out
    coords = candidates[list(coord_cols)].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    candidates['cluster'] = clustering.labels_
    max_prob_idx = candidates.groupby('cluster')[prob_col].idxmax()
    final_nodes = set(candidates.loc[max_prob_idx, node_id_col])
    df_out['is_singularity_final'] = df_out[node_id_col].isin(final_nodes).astype(int)
    return df_out

def assign_final_psl(df, prob_col='is_on_psl', coord_cols=('x', 'y'),
                     node_id_col='node_id', threshold=0.5, eps=0.05, min_samples=1):
    """根据PSL概率和坐标聚类，标记最终PSL点"""
    df_out = df.copy()
    candidates = df_out[df_out[prob_col] > threshold].copy()
    if len(candidates) == 0:
        df_out['is_on_psl_final'] = 0
        return df_out
    coords = candidates[list(coord_cols)].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    candidates['cluster'] = clustering.labels_
    max_prob_idx = candidates.groupby('cluster')[prob_col].idxmax()
    final_nodes = set(candidates.loc[max_prob_idx, node_id_col])
    df_out['is_on_psl_final'] = df_out[node_id_col].isin(final_nodes).astype(int)
    return df_out

def merge_dataframes(df1, df2, id_col, skip_cols=None):
    """合并两个DataFrame，处理占位符'/'，检查一致性"""
    if skip_cols is None:
        skip_cols = []
    merged = pd.merge(df1, df2, on=id_col, how='outer', suffixes=('_1', '_2'))
    result = pd.DataFrame()
    result[id_col] = merged[id_col]
    
    all_cols = set(df1.columns) | set(df2.columns)
    common_cols = all_cols - {id_col} - set(skip_cols)
    
    for col in common_cols:
        col1 = col + '_1'
        col2 = col + '_2'
        if col1 not in merged.columns:
            result[col] = merged[col2]
            continue
        if col2 not in merged.columns:
            result[col] = merged[col1]
            continue
        vals1 = merged[col1].fillna('/').replace('', '/')
        vals2 = merged[col2].fillna('/').replace('', '/')
        def merge_cell(v1, v2):
            if v1 == '/' and v2 == '/':
                return '/'
            if v1 == '/':
                return v2
            if v2 == '/':
                return v1
            if str(v1) != str(v2):
                errors.append(f"Conflict in column '{col}': {v1} vs {v2}")
                return v1
            return v1
        result[col] = [merge_cell(v1, v2) for v1, v2 in zip(vals1, vals2)]
    
    # 处理skip_cols（如 is_singularity, is_on_psl 等）
    for col in skip_cols:
        if col in df1.columns and col in df2.columns:
            # 检查一致性
            temp = pd.merge(df1[[id_col, col]], df2[[id_col, col]], on=id_col, how='outer', suffixes=('_1', '_2'))
            for idx, row in temp.iterrows():
                v1 = row.get(col+'_1', '/')
                v2 = row.get(col+'_2', '/')
                if str(v1) != str(v2) and v1 != '/' and v2 != '/':
                    errors.append(f"Conflict in skip column '{col}': {v1} vs {v2}")
            result[col] = df1.set_index(id_col)[col].reindex(result[id_col]).fillna(df2.set_index(id_col)[col]).fillna('/')
        elif col in df1.columns:
            result[col] = df1.set_index(id_col)[col].reindex(result[id_col]).fillna('/')
        elif col in df2.columns:
            result[col] = df2.set_index(id_col)[col].reindex(result[id_col]).fillna('/')
        else:
            result[col] = '/'
    return result

def check_node_completeness(df, group_name):
    """检查合并后的节点数据是否仍有占位符，记录错误"""
    # 检查 m1_vx, m1_vy, m2_vx, m2_vy 四列
    for col in ['m1_vx', 'm1_vy', 'm2_vx', 'm2_vy']:
        missing = df[df[col] == '/']
        if len(missing) > 0:
            errors.append(f"Group {group_name}: {len(missing)} nodes have '{col}' = '/' after merge. Node IDs: {missing['node_id'].tolist()[:10]}...")

def merge_nodes(file1, file2, group_name):
    """合并节点文件，并添加 final 标记"""
    print(f"Processing nodes for {group_name}...")
    df1 = pd.read_csv(file1, keep_default_na=False, na_values=[''])
    df2 = pd.read_csv(file2, keep_default_na=False, na_values=[''])
    
    # 合并节点
    merged = merge_dataframes(df1, df2, 'node_id', skip_cols=['is_singularity', 'is_on_psl'])
    
    # 添加最终奇异点和PSL点标记
    merged = assign_final_singularities(merged)
    merged = assign_final_psl(merged)
    
    # 检查完整性
    check_node_completeness(merged, group_name)
    
    # 保存
    out_path = os.path.join(output_dir, f"{group_name}_merged_nodes.csv")
    merged.to_csv(out_path, index=False)
    print(f"Saved nodes to {out_path}")
    return merged

def merge_edges(file1, file2, group_name, node_df):
    """合并边文件，并根据节点信息添加 type 列"""
    print(f"Processing edges for {group_name}...")
    df1 = pd.read_csv(file1, keep_default_na=False, na_values=[''])
    df2 = pd.read_csv(file2, keep_default_na=False, na_values=[''])
    
    # 合并边
    merged = merge_dataframes(df1, df2, 'edge_id')
    
    # 添加 type 列：根据边两端节点是否有 m2 值决定
    def get_edge_type(row):
        start = row['start_id']
        end = row['end_id']
        start_node = node_df[node_df['node_id'] == start]
        end_node = node_df[node_df['node_id'] == end]
        if start_node.empty or end_node.empty:
            errors.append(f"Edge {row['edge_id']}: node {start if start_node.empty else end} not found in merged nodes")
            return 'unknown'
        def has_m2(node_series):
            vx = node_series.get('m2_vx', '/')
            vy = node_series.get('m2_vy', '/')
            return vx != '/' or vy != '/'
        def has_m1(node_series):
            vx = node_series.get('m1_vx', '/')
            vy = node_series.get('m1_vy', '/')
            return vx != '/' or vy != '/'
        if has_m2(start_node.iloc[0]) or has_m2(end_node.iloc[0]):
            return 'm2'
        elif has_m1(start_node.iloc[0]) or has_m1(end_node.iloc[0]):
            return 'm1'
        else:
            errors.append(f"Edge {row['edge_id']}: no m1/m2 for nodes {start} or {end}")
            return 'unknown'
    
    merged['type'] = merged.apply(get_edge_type, axis=1)
    
    out_path = os.path.join(output_dir, f"{group_name}_merged_edges.csv")
    merged.to_csv(out_path, index=False)
    print(f"Saved edges to {out_path}")

# ==================== 主流程 ====================
def main():
    # 获取所有csv文件
    all_files = glob(os.path.join(data_dir, "*.csv"))
    nodes_files = [f for f in all_files if "nodes" in os.path.basename(f)]
    edges_files = [f for f in all_files if "edges" in os.path.basename(f)]
    
    # 配对节点文件
    nodes_pairs = parse_file_pairs([os.path.basename(f) for f in nodes_files])
    # 配对边文件
    edges_pairs = parse_file_pairs([os.path.basename(f) for f in edges_files])
    
    # 1. 先处理所有节点，保存合并后的DataFrame供边使用
    node_dfs = {}
    for group, versions in nodes_pairs.items():
        file1 = os.path.join(data_dir, versions['1'])
        file2 = os.path.join(data_dir, versions['2'])
        node_df = merge_nodes(file1, file2, group)
        node_dfs[group] = node_df
    
    # 2. 处理所有边，利用对应的节点数据
    for group, versions in edges_pairs.items():
        if group not in node_dfs:
            errors.append(f"Node file for group {group} not found, skip edges")
            continue
        file1 = os.path.join(data_dir, versions['1'])
        file2 = os.path.join(data_dir, versions['2'])
        merge_edges(file1, file2, group, node_dfs[group])
    
    # 3. 打印错误
    if errors:
        print("\n=== Errors encountered ===")
        for err in set(errors):
            print(err)
    else:
        print("\nNo errors found.")

if __name__ == "__main__":
    main()