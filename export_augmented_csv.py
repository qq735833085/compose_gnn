# export_augmented_csv.py
# =============================================================================
# 从增强 .pt 数据集反向导出 CSV（nodes.csv + edges.csv）
#
# 数据流：
#   initial_dataset/ CSV → data_merge.py → merged CSV → read_oridata.py → .pt
#                                                         ↑
#   build_augmented_dataset.py  ← 增强 .pt ← 本脚本 → augmented_data/ CSV
#
# 输出目录结构：
#   augmented_data/
#     case_000_orig/nodes.csv, edges.csv    ← 原始样本
#     case_000_aug_0/nodes.csv, edges.csv   ← 增强变体 #0
#     case_000_aug_1/nodes.csv, edges.csv   ← 增强变体 #1
#     ...
# =============================================================================

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 特征索引（与 read_oridata.py 一致，12 维去掉 z 列）
IDX_X, IDX_Y = 0, 1
IDX_M1_VX, IDX_M1_VY, IDX_M1_ABS = 2, 3, 4
IDX_M1_OH_0, IDX_M1_OH_1 = 5, 6
IDX_M2_VX, IDX_M2_VY, IDX_M2_ABS = 7, 8, 9
IDX_M2_OH_0, IDX_M2_OH_1 = 10, 11


def data_to_csv(data, nodes_path, edges_path, z_value=0.0):
    """
    将 PyG Data 对象反向导出为 nodes.csv 和 edges.csv。

    PyG Data 特征 (12 维) → CSV 列:
      [0] x  → x
      [1] y  → y
      z     → z (固定值，原始数据中 z=0)
      [2] m1_vx → m1_vx
      [3] m1_vy → m1_vy
      [4] m1_abs → m1_abs
      [5,6] m1_oh → m1_type (+t / -c)
      [7] m2_vx → m2_vx
      [8] m2_vy → m2_vy
      [9] m2_abs → m2_abs
      [10,11] m2_oh → m2_type (+t / -c)
      y_node → is_singularity
      --- is_on_psl (无对应，填 0)

    edge_index + y_edge → edges.csv
    """
    x = data.x.cpu().numpy()
    y_node = data.y_node.cpu().numpy().ravel()
    y_edge = data.y_edge.cpu().numpy()  # 保持原始形状 [2E] 或 [2E, 2]
    edge_index = data.edge_index.cpu().numpy()

    N = x.shape[0]
    E = edge_index.shape[1] // 2  # 无向边数（取一半）

    # ---- 构建 nodes.csv ----
    nodes = pd.DataFrame()
    nodes['node_id'] = np.arange(N)

    # 坐标（z 恢复为 0）
    nodes['x'] = x[:, IDX_X]
    nodes['y'] = x[:, IDX_Y]
    nodes['z'] = z_value

    # m1 方向分量（零向量输出为 '/'）
    m1_vx = x[:, IDX_M1_VX]
    m1_vy = x[:, IDX_M1_VY]
    m1_is_zero = (np.abs(m1_vx) < 1e-8) & (np.abs(m1_vy) < 1e-8)
    nodes['m1_vx'] = np.where(m1_is_zero, '/', m1_vx.round(12))
    nodes['m1_vy'] = np.where(m1_is_zero, '/', m1_vy.round(12))

    # m1 类型（从独热编码反向）
    m1_is_tension = x[:, IDX_M1_OH_0] > 0.5
    nodes['m1_type'] = np.where(m1_is_tension, '+t', '-c')

    # m1 大小
    nodes['m1_abs'] = x[:, IDX_M1_ABS]

    # m2 方向分量
    m2_vx = x[:, IDX_M2_VX]
    m2_vy = x[:, IDX_M2_VY]
    m2_is_zero = (np.abs(m2_vx) < 1e-8) & (np.abs(m2_vy) < 1e-8)
    nodes['m2_vx'] = np.where(m2_is_zero, '/', m2_vx.round(12))
    nodes['m2_vy'] = np.where(m2_is_zero, '/', m2_vy.round(12))

    # m2 类型
    m2_is_tension = x[:, IDX_M2_OH_0] > 0.5
    nodes['m2_type'] = np.where(m2_is_tension, '+t', '-c')

    # m2 大小
    nodes['m2_abs'] = x[:, IDX_M2_ABS]

    # 标签
    nodes['is_singularity'] = y_node
    nodes['is_on_psl'] = 0.0  # 当前数据集中此标签已被移除，填 0

    # ---- 构建 edges.csv ----
    edges = pd.DataFrame()
    edges['edge_id'] = np.arange(E)

    # edge_index 是双向边 [2, 2E]，取正向边 (偶数索引)
    edges['start_id'] = edge_index[0, 0::2]
    edges['end_id'] = edge_index[1, 0::2]

    # PSL 标签（双列：is_psl_1 + is_psl_2）
    if y_edge.ndim == 2 and y_edge.shape[1] >= 2:
        edges['is_psl_1'] = y_edge[0::2, 0].astype(int)
        edges['is_psl_2'] = y_edge[0::2, 1].astype(int)
        edges['close_to_psl_1'] = 0.0  # 增强后概率值不再保留，填 0
        edges['close_to_psl_2'] = 0.0
    else:
        # 兼容旧格式（单列）
        edges['is_psl_1'] = y_edge[0::2].astype(int)
        edges['is_psl_2'] = 0
        edges['close_to_psl_1'] = 0.0
        edges['close_to_psl_2'] = 0.0

    # ---- 保存 ----
    os.makedirs(os.path.dirname(nodes_path) or '.', exist_ok=True)
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)

    return nodes, edges


def main():
    parser = argparse.ArgumentParser(description='Export augmented .pt to CSV files')
    parser.add_argument('--input', type=str,
                        default='graph_dataset/20260602_104_pro_augmented_x3.pt',
                        help='Input augmented .pt file')
    parser.add_argument('--output-dir', type=str, default='augmented_data',
                        help='Output directory for CSV files')
    parser.add_argument('--n-originals', type=int, default=104,
                        help='Number of original graphs (first N in dataset)')
    parser.add_argument('--n-augment', type=int, default=3,
                        help='Number of augmented variants per original')
    parser.add_argument('--max-export', type=int, default=None,
                        help='Max graphs to export (for testing)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'Error: {args.input} not found')
        sys.exit(1)

    print(f'Loading: {args.input}')
    dataset = torch.load(args.input, weights_only=False)
    n_total = len(dataset)
    print(f'  {n_total} graphs loaded')

    n_to_export = min(n_total, args.max_export) if args.max_export else n_total

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'\nExporting {n_to_export} graphs to {args.output_dir}/ ...')

    for i in tqdm(range(n_to_export), desc='Exporting'):
        data = dataset[i]

        # 确定分类：原始 or 增强变体
        if i < args.n_originals:
            case_dir = f'case_{i:03d}_orig'
        else:
            aug_idx = (i - args.n_originals) % args.n_augment
            orig_idx = (i - args.n_originals) // args.n_augment
            case_dir = f'case_{orig_idx:03d}_aug_{aug_idx}'

        case_path = os.path.join(args.output_dir, case_dir)
        nodes_path = os.path.join(case_path, 'nodes.csv')
        edges_path = os.path.join(case_path, 'edges.csv')

        data_to_csv(data, nodes_path, edges_path)

    # 统计
    n_dirs = len(os.listdir(args.output_dir))
    print(f'\nDone! {n_dirs} case directories exported to: {os.path.abspath(args.output_dir)}/')

    # 目录结构示意
    print(f'\nDirectory structure:')
    print(f'{args.output_dir}/')
    sample_dirs = sorted(os.listdir(args.output_dir))[:5]
    for d in sample_dirs:
        files = os.listdir(os.path.join(args.output_dir, d))
        print(f'  {d}/')
        for f in files:
            size_kb = os.path.getsize(os.path.join(args.output_dir, d, f)) / 1024
            print(f'    {f}  ({size_kb:.1f} KB)')
    if n_dirs > 5:
        print(f'  ... ({n_dirs - 5} more)')


if __name__ == '__main__':
    main()
