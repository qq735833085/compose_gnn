# visualize_merge_check.py
# =============================================================================
# 合并前后数据验证可视化
#
# 功能：对单个案例，对比标注员 1、标注员 2 和合并后的数据，
#       帮助验证合并逻辑是否正确。
#
# 布局 (4 行 × 4 列)：
#   ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
#   │ 原始 Ann1: m1    │ 原始 Ann2: m2    │ 原始 Ann1: Sing  │ 原始 Ann2: Sing  │
#   │ (谁填了 m1/m2)   │ (谁填了 m1/m2)   │ (奇异点概率)     │ (奇异点概率)     │
#   ├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
#   │ 合并: m1_abs     │ 合并: m2_abs     │ 合并: Singularity│ 合并: PSL 节点   │
#   │ (填充后)         │ (填充后)         │ (DBSCAN 结果)    │ Red=Set1 Blue=S2 │
#   ├──────────────────┼──────────────────┼──────────────────┼──────────────────┤
#   │ 边 PSL: 概率 Ann1│ 边 PSL: 概率 Ann2│ 边 PSL: 二值 S1 │ 边 PSL: 二值 S2 │
#   │ (原始概率)       │ (原始概率)       │ (连通分量法)     │ (连通分量法)     │
#   ├──────────────────┴──────────────────┴──────────────────┴──────────────────┤
#   │ 分布: 奇异点概率直方图 + PSL概率直方图 + m1/m2直方图   (底层统计行)       │
#   └──────────────────────────────────────────────────────────────────────────┘
#
# 用法：
#   python visualize_merge_check.py --case 1
#   python visualize_merge_check.py --case jia1-s3
# =============================================================================

import os, sys, argparse
from glob import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

from merge_dual_annotators import (DATA_DIR, OUTPUT_DIR,
                                     parse_file_pairs, safe_float, clean_type,
                                     merge_nodes, merge_edges,
                                     PSL_STRICT_PTILE, PSL_LOOSE_PTILE)

# =============================================================================
# 主可视化函数
# =============================================================================

def visualize_merge(case_name, versions, input_dir, output_dir, save_path=None):
    """
    对指定案例生成合并前后对比图。

    Args:
        case_name: 案例组名 (如 '1')
        versions: {'1': '1.1 nodes.csv', '2': '1.2 nodes.csv'}
        input_dir: 原始数据目录
        output_dir: 合并后数据目录
        save_path: 保存路径，自动生成时存在 output_dir 下
    """
    if save_path is None:
        save_path = os.path.join(output_dir, f'{case_name}_merge_check.png')

    # 加载原始数据
    f1_nodes = os.path.join(input_dir, versions['1'])
    f2_nodes = os.path.join(input_dir, versions['2'])
    f1_edges = f1_nodes.replace(' nodes', ' edges')
    f2_edges = f2_nodes.replace(' nodes', ' edges')

    df1_n = pd.read_csv(f1_nodes, keep_default_na=False, na_values=[''])
    df2_n = pd.read_csv(f2_nodes, keep_default_na=False, na_values=[''])
    df1_e = pd.read_csv(f1_edges)
    df2_e = pd.read_csv(f2_edges)

    # 加载/生成合并数据
    merged_n_path = os.path.join(output_dir, f'{case_name}_merged_nodes.csv')
    merged_e_path = os.path.join(output_dir, f'{case_name}_merged_edges.csv')

    if os.path.exists(merged_n_path) and os.path.exists(merged_e_path):
        dfm_n = pd.read_csv(merged_n_path)
        dfm_e = pd.read_csv(merged_e_path)
    else:
        print('Merged files not found, generating...')
        dfm_n = merge_nodes(f1_nodes, f2_nodes, case_name)
        dfm_e = merge_edges(f1_edges, f2_edges, case_name, dfm_n)

    # ---- 绘图 ----
    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(f'Merge Check: Case "{case_name}"\n'
                 f'Annotator 1 ({os.path.basename(f1_nodes)}) + '
                 f'Annotator 2 ({os.path.basename(f2_nodes)})',
                 fontsize=13, fontweight='bold')

    gs = GridSpec(4, 4, figure=fig, hspace=0.40, wspace=0.30,
                  height_ratios=[1, 1, 1, 0.6])

    # ===== Row 1: 原始数据（标注员1 vs 标注员2）=====
    # [0,0] m1 互补填充检查
    ax = fig.add_subplot(gs[0, 0])
    plot_dual_heatmap(ax, df1_n, df2_n, 'm1_abs', 'm1_abs',
                       title='Raw: m1 Fill Check\n(Red=Ann1 has, Blue=Ann2 has)')
    # [0,1] m2 互补填充检查
    ax = fig.add_subplot(gs[0, 1])
    plot_dual_heatmap(ax, df1_n, df2_n, 'm2_abs', 'm2_abs',
                       title='Raw: m2 Fill Check\n(Red=Ann1 has, Blue=Ann2 has)')
    # [0,2] ★ 原始奇异点概率 — 标注员 1
    ax = fig.add_subplot(gs[0, 2])
    plot_scalar_heatmap(ax, df1_n, 'is_singularity', cmap='hot', vmin=0, vmax=1,
                         title=f'Ann1: is_singularity prob\n'
                               f'max={df1_n["is_singularity"].max():.3f}, >0.5:{(df1_n["is_singularity"]>0.5).sum()}')
    # [0,3] ★ 原始奇异点概率 — 标注员 2
    ax = fig.add_subplot(gs[0, 3])
    plot_scalar_heatmap(ax, df2_n, 'is_singularity', cmap='hot', vmin=0, vmax=1,
                         title=f'Ann2: is_singularity prob\n'
                               f'max={df2_n["is_singularity"].max():.3f}, >0.5:{(df2_n["is_singularity"]>0.5).sum()}')

    # ===== Row 2: 合并后节点 =====
    # [1,0] 合并 m1_abs
    ax = fig.add_subplot(gs[1, 0])
    plot_scalar_heatmap(ax, dfm_n, 'm1_abs', cmap='viridis',
                         title='Merged: m1_abs (both filled)')
    # [1,1] 合并 m2_abs
    ax = fig.add_subplot(gs[1, 1])
    plot_scalar_heatmap(ax, dfm_n, 'm2_abs', cmap='viridis',
                         title='Merged: m2_abs (both filled)')
    # [1,2] 合并 Singularity (DBSCAN 最终结果)
    ax = fig.add_subplot(gs[1, 2])
    plot_binary_heatmap(ax, dfm_n, 'is_singularity',
                         title=f'Merged: Singularity (DBSCAN final)\n'
                               f'{dfm_n["is_singularity"].sum()} singularities found')
    # [1,3] ★ 合并 PSL 节点（基于已确定 PSL 边的二值）
    ax = fig.add_subplot(gs[1, 3])
    plot_psl_node_comparison(ax, dfm_n,
                              title='Merged: PSL Nodes (binary)\n'
                                    'Red=Set1, Blue=Set2, Purple=Both')

    # ===== Row 3: 边 PSL =====
    # [2,0] PSL 概率 Ann1（原始，供参考）
    ax = fig.add_subplot(gs[2, 0])
    plot_edge_heatmap(ax, dfm_e, dfm_n, 'close_to_psl_1', cmap='Reds',
                       title=f'Edge close_to_psl (Set1 raw, reference)\n'
                             f'nonzero:{(dfm_e["close_to_psl_1"]>1e-6).sum()}')
    # [2,1] PSL 概率 Ann2（原始，供参考）
    ax = fig.add_subplot(gs[2, 1])
    plot_edge_heatmap(ax, dfm_e, dfm_n, 'close_to_psl_2', cmap='Blues',
                       title=f'Edge close_to_psl (Set2 raw, reference)\n'
                             f'nonzero:{(dfm_e["close_to_psl_2"]>1e-6).sum()}')
    # [2,2] ★ PSL 二值 S1（节点概率场脊线检测）
    ax = fig.add_subplot(gs[2, 2])
    psl1 = dfm_e['is_psl_1'].sum()
    t1_s = np.percentile(dfm_n['is_on_psl_raw_1'], PSL_STRICT_PTILE)
    t1_l = np.percentile(dfm_n['is_on_psl_raw_1'], PSL_LOOSE_PTILE)
    plot_edge_binary(ax, dfm_e, dfm_n, 'is_psl_1', color='red',
                      title=f'PSL edges Set1 (node ridge detection)\n'
                            f'{psl1} edges | T_s={t1_s:.3f} T_l={t1_l:.3f}')
    # [2,3] ★ PSL 二值 S2（节点概率场脊线检测）
    ax = fig.add_subplot(gs[2, 3])
    psl2 = dfm_e['is_psl_2'].sum()
    t2_s = np.percentile(dfm_n['is_on_psl_raw_2'], PSL_STRICT_PTILE)
    t2_l = np.percentile(dfm_n['is_on_psl_raw_2'], PSL_LOOSE_PTILE)
    plot_edge_binary(ax, dfm_e, dfm_n, 'is_psl_2', color='blue',
                      title=f'PSL edges Set2 (node ridge detection)\n'
                            f'{psl2} edges | T_s={t2_s:.3f} T_l={t2_l:.3f}')

    # ===== Row 4: 分布直方图 =====
    # [3,0:2] 奇异点概率分布 (Ann1 + Ann2)
    ax = fig.add_subplot(gs[3, 0:2])
    s1 = pd.to_numeric(df1_n['is_singularity'], errors='coerce').fillna(0)
    s2 = pd.to_numeric(df2_n['is_singularity'], errors='coerce').fillna(0)
    ax.hist(s1, bins=50, alpha=0.5, color='coral', density=True,
            label=f'Ann1 (>{0.5}:{(s1>0.5).sum()})', edgecolor='white', linewidth=0.2)
    ax.hist(s2, bins=50, alpha=0.5, color='steelblue', density=True,
            label=f'Ann2 (>{0.5}:{(s2>0.5).sum()})', edgecolor='white', linewidth=0.2)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1)
    ax.set_title('Singularity Probability Distribution'); ax.legend(fontsize=8)

    # [3,2:4] PSL 概率分布 (Ann1 + Ann2)
    ax = fig.add_subplot(gs[3, 2:4])
    c1 = dfm_e['close_to_psl_1'].values
    c2 = dfm_e['close_to_psl_2'].values
    # 只画 >0 的值（去掉大量零值）
    c1_pos = c1[c1 > 1e-6]
    c2_pos = c2[c2 > 1e-6]
    ax.hist(c1_pos, bins=50, alpha=0.5, color='red', density=True,
            label=f'Set1 nonzero:{len(c1_pos)}', edgecolor='white', linewidth=0.2)
    ax.hist(c2_pos, bins=50, alpha=0.5, color='blue', density=True,
            label=f'Set2 nonzero:{len(c2_pos)}', edgecolor='white', linewidth=0.2)
    ax.set_title('close_to_psl Distribution (nonzero only, for reference)'); ax.legend(fontsize=6)

    # 底部统计
    fig.text(0.5, 0.005,
             f'Nodes: {len(dfm_n)} | Edges: {len(dfm_e)} | '
             f'Singularities: {dfm_n["is_singularity"].sum()} | '
             f'PSL edges: {dfm_e["is_psl_1"].sum()} (S1) + {dfm_e["is_psl_2"].sum()} (S2) | '
             f'PSL nodes: {dfm_n["is_on_psl_1"].sum()} (S1) + {dfm_n["is_on_psl_2"].sum()} (S2)',
             ha='center', fontsize=10, style='italic')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {save_path}')
    plt.close(fig)


# =============================================================================
# 子图绘制函数
# =============================================================================

def plot_scalar_heatmap(ax, df, col, cmap='viridis', title='', point_size=1.5, vmin=None, vmax=None):
    """通用节点标量热力图"""
    x, y = df['x'].values, df['y'].values
    vals = pd.to_numeric(df[col], errors='coerce').fillna(0).values
    idx = _downsample(len(x), 8000)
    sc = ax.scatter(x[idx], y[idx], c=vals[idx], cmap=cmap, s=point_size, marker='.',
                    vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=9)


def plot_binary_heatmap(ax, df, col, title='', point_size=2):
    """二值热力图（0=灰, 1=红）"""
    x, y = df['x'].values, df['y'].values
    vals = df[col].values.astype(int)
    idx = _downsample(len(x), 10000)
    colors = np.where(vals[idx] == 1, 'red', 'lightgray')
    ax.scatter(x[idx], y[idx], c=colors, s=point_size, marker='.', alpha=0.6)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=9)


def plot_dual_heatmap(ax, df1, df2, col1, col2, title='', point_size=2):
    """双标注员对比热力图：显示谁填充了哪个节点"""
    x, y = df1['x'].values, df1['y'].values

    v1 = pd.to_numeric(df1[col1], errors='coerce').fillna(0).values
    v2 = pd.to_numeric(df2[col2], errors='coerce').fillna(0).values

    idx = _downsample(len(x), 8000)
    # 标注员 1 有值 → 红色
    mask1 = v1[idx] > 1e-8
    ax.scatter(x[idx][mask1], y[idx][mask1], c='red', s=point_size, marker='.',
               alpha=0.4, label=f'Ann1 has ({mask1.sum()})')
    # 标注员 2 有值 → 蓝色
    mask2 = v2[idx] > 1e-8
    ax.scatter(x[idx][mask2], y[idx][mask2], c='blue', s=point_size, marker='.',
               alpha=0.4, label=f'Ann2 has ({mask2.sum()})')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=9)


def plot_psl_node_comparison(ax, df, title='', point_size=2):
    """节点级 PSL 两套标注对比"""
    x, y = df['x'].values, df['y'].values
    psl1 = df['is_on_psl_1'].values == 1  # 二值（已从边 PSL 转换）
    psl2 = df['is_on_psl_2'].values == 1

    idx = _downsample(len(x), 10000)
    # 仅标注员 1
    only1 = psl1[idx] & ~psl2[idx]
    ax.scatter(x[idx][only1], y[idx][only1], c='red', s=point_size, marker='.',
               alpha=0.5, label=f'Ann1 only ({only1.sum()})')
    # 仅标注员 2
    only2 = ~psl1[idx] & psl2[idx]
    ax.scatter(x[idx][only2], y[idx][only2], c='blue', s=point_size, marker='.',
               alpha=0.5, label=f'Ann2 only ({only2.sum()})')
    # 两者都有
    both = psl1[idx] & psl2[idx]
    ax.scatter(x[idx][both], y[idx][both], c='purple', s=point_size, marker='.',
               alpha=0.6, label=f'Both ({both.sum()})')

    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=9)


def plot_edge_heatmap(ax, edge_df, node_df, col, cmap='Reds', title='',
                       line_width=0.5):
    """边级热力图（高概率边全部保留，低概率边下采样）"""
    node_xy = node_df.set_index('node_id')[['x', 'y']]
    vals = edge_df[col].values
    starts = edge_df['start_id'].values
    ends = edge_df['end_id'].values

    n = len(starts)
    from matplotlib.collections import LineCollection

    # 背景边（低概率）下采样
    lo_mask = vals < np.percentile(vals, 90)
    lo_idx = np.where(lo_mask)[0]
    if len(lo_idx) > 3000:
        lo_idx = np.random.choice(lo_idx, 3000, replace=False)

    # 高概率边全部保留
    hi_idx = np.where(~lo_mask)[0]

    all_idx = np.concatenate([lo_idx, hi_idx])
    norm = Normalize(vmin=0, vmax=max(vals.max(), 0.01))
    segments = []
    colors = []
    for i in all_idx:
        s_xy = node_xy.loc[starts[i]].values
        e_xy = node_xy.loc[ends[i]].values
        segments.append([(s_xy[0], s_xy[1]), (e_xy[0], e_xy[1])])
        colors.append(vals[i])

    lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.6,
                        linewidths=line_width)
    lc.set_array(np.array(colors))
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=9)


def plot_edge_binary(ax, edge_df, node_df, col, color='red', title='',
                      line_width=1.2):
    """边级二值图（PSL 边全部高亮，不下采样）"""
    node_xy = node_df.set_index('node_id')[['x', 'y']]
    vals = edge_df[col].values
    starts = edge_df['start_id'].values
    ends = edge_df['end_id'].values

    n = len(starts)
    from matplotlib.collections import LineCollection

    # 1. 先画背景边（下采样）
    bg_idx = _downsample(n, 4000)
    bg_segments = []
    for i in bg_idx:
        if vals[i] != 1:  # 跳过 PSL 边
            s_xy = node_xy.loc[starts[i]].values
            e_xy = node_xy.loc[ends[i]].values
            bg_segments.append([(s_xy[0], s_xy[1]), (e_xy[0], e_xy[1])])

    if bg_segments:
        lc_bg = LineCollection(bg_segments, colors='lightgray', alpha=0.12,
                               linewidths=0.2)
        ax.add_collection(lc_bg)

    # 2. PSL 边全部绘制（不下采样），加粗高亮
    psl_idx = np.where(vals == 1)[0]
    fg_segments = []
    for i in psl_idx:
        s_xy = node_xy.loc[starts[i]].values
        e_xy = node_xy.loc[ends[i]].values
        fg_segments.append([(s_xy[0], s_xy[1]), (e_xy[0], e_xy[1])])

    if fg_segments:
        lc_fg = LineCollection(fg_segments, colors=color, alpha=0.85,
                               linewidths=line_width, zorder=10)
        ax.add_collection(lc_fg)

    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=9)


def _downsample(n, max_n):
    """均匀下采样索引"""
    if n <= max_n:
        return np.arange(n)
    step = max(1, n // max_n)
    return np.arange(0, n, step)


# =============================================================================
# 直方图对比（补充检查）
# =============================================================================

def visualize_distributions(case_name, versions, input_dir, output_dir, save_path=None):
    """生成合并前后的分布直方图对比"""
    if save_path is None:
        save_path = os.path.join(output_dir, f'{case_name}_dist_check.png')

    f1_nodes = os.path.join(input_dir, versions['1'])
    f2_nodes = os.path.join(input_dir, versions['2'])
    df1 = pd.read_csv(f1_nodes, keep_default_na=False, na_values=[''])
    df2 = pd.read_csv(f2_nodes, keep_default_na=False, na_values=[''])

    merged_path = os.path.join(output_dir, f'{case_name}_merged_nodes.csv')
    dfm = pd.read_csv(merged_path)

    # 关键的 6 个数值列
    cols = ['m1_vx', 'm1_vy', 'm1_abs', 'm2_vx', 'm2_vy', 'm2_abs']
    fig, axes = plt.subplots(2, 6, figsize=(20, 7))

    for i, col in enumerate(cols):
        # 原始
        ax = axes[0, i]
        v1 = pd.to_numeric(df1[col], errors='coerce').fillna(0)
        v2 = pd.to_numeric(df2[col], errors='coerce').fillna(0)
        ax.hist(v1, bins=40, alpha=0.5, color='red', density=True, label='Ann1',
                edgecolor='white', linewidth=0.2)
        ax.hist(v2, bins=40, alpha=0.5, color='blue', density=True, label='Ann2',
                edgecolor='white', linewidth=0.2)
        ax.set_title(f'Raw: {col}', fontsize=8); ax.tick_params(labelsize=6)

        # 合并后
        ax = axes[1, i]
        vm = pd.to_numeric(dfm[col], errors='coerce').fillna(0)
        ax.hist(vm, bins=40, alpha=0.7, color='green', density=True,
                edgecolor='white', linewidth=0.2)
        # 标注非零比例
        nz = (vm.abs() > 1e-8).mean() * 100
        ax.set_title(f'Merged: {col} (nonzero={nz:.0f}%)', fontsize=8)
        ax.tick_params(labelsize=6)
        if i == 0:
            axes[0, i].legend(fontsize=7)

    fig.suptitle(f'Distribution Check: Raw vs Merged — Case "{case_name}"',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {save_path}')
    plt.close(fig)


# =============================================================================
# CLI 入口
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize merge correctness')
    parser.add_argument('--case', type=str, required=True,
                        help='Case name to visualize (e.g. "1", "jia1-s3")')
    parser.add_argument('--input-dir', type=str, default=DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    # 找到对应的文件对
    all_files = glob(os.path.join(args.input_dir, '*.csv'))
    nodes_basenames = [os.path.basename(f) for f in all_files if 'nodes' in os.path.basename(f)]
    pairs, errors = parse_file_pairs(nodes_basenames)

    if args.case not in pairs:
        print(f'Case "{args.case}" not found. Available: {sorted(pairs.keys())[:20]}...')
        sys.exit(1)

    # 先生成合并数据（如果还没有）
    from merge_dual_annotators import merge_nodes, merge_edges

    versions = pairs[args.case]
    f1_nodes = os.path.join(args.input_dir, versions['1'])
    f2_nodes = os.path.join(args.input_dir, versions['2'])
    f1_edges = f1_nodes.replace(' nodes', ' edges')
    f2_edges = f2_nodes.replace(' nodes', ' edges')

    print(f'Case: {args.case}')
    print(f'  Ann1: {os.path.basename(f1_nodes)} + {os.path.basename(f1_edges)}')
    print(f'  Ann2: {os.path.basename(f2_nodes)} + {os.path.basename(f2_edges)}')

    # 确保合并数据存在
    merged_n = os.path.join(args.output_dir, f'{args.case}_merged_nodes.csv')
    merged_e = os.path.join(args.output_dir, f'{args.case}_merged_edges.csv')
    if not os.path.exists(merged_n) or not os.path.exists(merged_e):
        print('Generating merged files...')
        node_df = merge_nodes(f1_nodes, f2_nodes, args.case)
        edge_df = merge_edges(f1_edges, f2_edges, args.case, node_df)
        os.makedirs(args.output_dir, exist_ok=True)
        node_df.to_csv(merged_n, index=False)
        edge_df.to_csv(merged_e, index=False)
        print(f'  Saved: {merged_n}, {merged_e}')

    # 可视化
    print('\nGenerating merge check visualization...')
    save_path = os.path.join(args.output_dir, f'{args.case}_merge_check.png')
    visualize_merge(args.case, versions, args.input_dir, args.output_dir, save_path)

    print('\nGenerating distribution check...')
    dist_path = os.path.join(args.output_dir, f'{args.case}_dist_check.png')
    visualize_distributions(args.case, versions, args.input_dir, args.output_dir, dist_path)

    print('\nDone!')
