# utils/visualization.py
# =============================================================================
# 楼板应力场数据综合可视化模块
# =============================================================================
# 支持 PyG Data 对象的全方位可视化：
#   1. 网格拓扑 + 应力方向场
#   2. 奇异点 / PSL 概率热力图
#   3. 主应力大小分布
#   4. 特征分布直方图
#   5. 综合仪表板（多图合一）
#   6. 增强前后对比
# =============================================================================

import math
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互后端，支持无 GUI 环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec

# =============================================================================
# 特征索引（与 read_oridata.py / augment.py 一致）
# =============================================================================
IDX_X, IDX_Y = 0, 1
IDX_M1_VX, IDX_M1_VY, IDX_M1_ABS = 2, 3, 4
IDX_M1_OH_0, IDX_M1_OH_1 = 5, 6
IDX_M2_VX, IDX_M2_VY, IDX_M2_ABS = 7, 8, 9
IDX_M2_OH_0, IDX_M2_OH_1 = 10, 11

FEATURE_NAMES = [
    'x', 'y',
    'm1_vx', 'm1_vy', 'm1_abs', 'm1_+t', 'm1_-c',
    'm2_vx', 'm2_vy', 'm2_abs', 'm2_+t', 'm2_-c',
]

# =============================================================================
# 配色方案
# =============================================================================
COLORMAP_SINGULARITY = 'hot'       # 奇异点：红热
COLORMAP_PSL = 'plasma'            # PSL：等离子
COLORMAP_MAGNITUDE = 'viridis'     # 力值：翠绿
COLORMAP_DIVERGING = 'RdBu_r'      # 方向分量：红蓝发散


def _downsample_nodes(x, y, max_points=3000):
    """对节点做均匀下采样，避免散点图过密"""
    n = len(x)
    if n <= max_points:
        return np.arange(n)
    step = max(1, n // max_points)
    return np.arange(0, n, step)


def _downsample_edges(edge_index, max_edges=2000):
    """对边做均匀下采样"""
    num_edges = edge_index.shape[1] // 2  # 双向边，取一半
    if num_edges <= max_edges:
        return edge_index
    step = max(1, num_edges // max_edges)
    idx = []
    for i in range(0, num_edges, step):
        idx.append(i * 2)      # 正向边
        idx.append(i * 2 + 1)  # 反向边
    return edge_index[:, idx]


def _to_numpy(data, field_or_idx):
    """将 Data 的字段转为 numpy，处理不同输入类型"""
    if isinstance(field_or_idx, int):
        # 按索引取 x 的列
        return data.x[:, field_or_idx].cpu().numpy().ravel()
    elif isinstance(field_or_idx, str):
        # 按属性名取
        val = getattr(data, field_or_idx)
        return val.cpu().numpy().ravel() if val.dim() > 1 else val.cpu().numpy()
    else:
        raise ValueError(f'Invalid field spec: {field_or_idx}')


# =============================================================================
# 单图可视化函数
# =============================================================================

def plot_mesh_topology(data, ax=None, title='Mesh Topology', edge_alpha=0.08,
                       edge_color='#333333', node_color='#1f77b4', node_size=2):
    """
    绘制网格拓扑结构（边 + 节点）。

    Args:
        data: PyG Data 对象
        ax: matplotlib Axes，为 None 时自动创建
        title: 图表标题
        edge_alpha: 边透明度（大图建议降低）
        edge_color: 边颜色
        node_color: 节点颜色
        node_size: 节点大小
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    x = _to_numpy(data, IDX_X)
    y = _to_numpy(data, IDX_Y)

    # 边（下采样）
    ei = _downsample_edges(data.edge_index)
    N_edges = ei.shape[1]

    # 批量绘制线段（避免逐条绘制的性能问题）
    segments = np.zeros((N_edges, 2, 2))
    for i in range(N_edges):
        src, dst = ei[0, i].item(), ei[1, i].item()
        segments[i, 0] = [x[src], y[src]]
        segments[i, 1] = [x[dst], y[dst]]

    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, colors=edge_color, alpha=edge_alpha, linewidths=0.3)
    ax.add_collection(lc)

    # 节点（下采样）
    idx = _downsample_nodes(x, y)
    ax.scatter(x[idx], y[idx], c=node_color, s=node_size, alpha=0.6, marker='.')

    ax.set_xlim(x.min() - 0.02, x.max() + 0.02)
    ax.set_ylim(y.min() - 0.02, y.max() + 0.02)
    ax.set_aspect('equal')
    ax.set_xlabel('x (normalized)')
    ax.set_ylabel('y (normalized)')
    ax.set_title(title)
    return ax


def plot_stress_direction_field(data, ax=None, title='Principal Stress Direction Field',
                                 arrow_spacing=500, arrow_scale=0.015, arrow_width=0.002):
    """
    绘制主应力方向场（用箭头表示 m1 和 m2 方向）。

    m1（主弯矩）：红色箭头，表示最大主应力方向
    m2（最小弯矩）：蓝色箭头，表示最小主应力方向

    Args:
        data: PyG Data 对象
        ax: matplotlib Axes
        arrow_spacing: 箭头间距（每隔多少个节点画一个箭头）
        arrow_scale: 箭头大小缩放
        arrow_width: 箭头线宽
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    x = _to_numpy(data, IDX_X)
    y = _to_numpy(data, IDX_Y)
    m1_vx = _to_numpy(data, IDX_M1_VX)
    m1_vy = _to_numpy(data, IDX_M1_VY)
    m2_vx = _to_numpy(data, IDX_M2_VX)
    m2_vy = _to_numpy(data, IDX_M2_VY)

    # 先画网格背景
    plot_mesh_topology(data, ax=ax, title='', edge_alpha=0.03, node_size=0)

    # 均匀间隔画箭头
    n = len(x)
    indices = np.arange(0, n, arrow_spacing)

    # m1 方向（红色）
    ax.quiver(x[indices], y[indices],
              m1_vx[indices], m1_vy[indices],
              color='red', alpha=0.6, scale=1/arrow_scale, width=arrow_width,
              headwidth=3, headlength=4, label='m1 (principal)')

    # m2 方向（蓝色）
    ax.quiver(x[indices], y[indices],
              m2_vx[indices], m2_vy[indices],
              color='blue', alpha=0.6, scale=1/arrow_scale, width=arrow_width,
              headwidth=3, headlength=4, label='m2 (secondary)')

    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.set_title(title)
    return ax


def plot_heatmap(data, field, ax=None, title='', cmap='hot', vmin=None, vmax=None,
                 point_size=1.5, alpha=0.8):
    """
    通用节点级热力图。

    Args:
        data: PyG Data 对象
        field: 字段名或特征索引（如 IDX_M1_ABS, 'y_node'）
        ax: matplotlib Axes
        title: 图表标题
        cmap: 配色方案
        vmin, vmax: 颜色范围
        point_size: 散点大小
        alpha: 透明度
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    x = _to_numpy(data, IDX_X)
    y = _to_numpy(data, IDX_Y)
    values = _to_numpy(data, field)

    idx = _downsample_nodes(x, y, max_points=10000)

    sc = ax.scatter(x[idx], y[idx], c=values[idx], cmap=cmap,
                    s=point_size, alpha=alpha, marker='.',
                    vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=title)
    ax.set_xlim(x.min() - 0.02, x.max() + 0.02)
    ax.set_ylim(y.min() - 0.02, y.max() + 0.02)
    ax.set_aspect('equal')
    ax.set_title(title)
    return ax


def plot_edge_heatmap(data, ax=None, title='Edge: close_to_psl', cmap='plasma',
                      line_width=0.5, alpha=0.5):
    """
    绘制边级热力图（按 close_to_psl 着色）。

    Args:
        data: PyG Data 对象
        ax: matplotlib Axes
        title: 图表标题
        cmap: 配色方案
        line_width: 线段宽度
        alpha: 透明度
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    x = _to_numpy(data, IDX_X)
    y = _to_numpy(data, IDX_Y)
    edge_labels = _to_numpy(data, 'y_edge')
    ei = _downsample_edges(data.edge_index)

    N_edges = ei.shape[1]
    from matplotlib.collections import LineCollection

    segments = []
    colors = []
    for i in range(N_edges):
        src, dst = ei[0, i].item(), ei[1, i].item()
        segments.append([(x[src], y[src]), (x[dst], y[dst])])
        colors.append(edge_labels[i])

    # 归一化颜色
    norm = Normalize(vmin=0, vmax=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=alpha,
                        linewidths=line_width)
    lc.set_array(np.array(colors))
    ax.add_collection(lc)

    plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04, label='close_to_psl')
    ax.set_xlim(x.min() - 0.02, x.max() + 0.02)
    ax.set_ylim(y.min() - 0.02, y.max() + 0.02)
    ax.set_aspect('equal')
    ax.set_title(title)
    return ax


def plot_feature_histograms(data, features=None, bins=50, figsize=(16, 10),
                            title='Node Feature Distributions'):
    """
    绘制节点特征分布直方图矩阵。

    Args:
        data: PyG Data 对象
        features: 要绘制的特征索引列表，默认全部 12 维
        bins: 直方图分箱数
        figsize: 图大小
        title: 图表标题
    """
    if features is None:
        features = list(range(12))

    n_feats = len(features)
    cols = min(4, n_feats)
    rows = math.ceil(n_feats / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    x_data = data.x.cpu().numpy()

    for i, feat_idx in enumerate(features):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        vals = x_data[:, feat_idx]
        ax.hist(vals, bins=bins, density=True, alpha=0.7, color='steelblue',
                edgecolor='white', linewidth=0.3)
        ax.set_title(FEATURE_NAMES[feat_idx], fontsize=9)
        ax.tick_params(labelsize=7)

    # 隐藏多余子图
    for i in range(n_feats, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_label_distributions(data, figsize=(10, 4),
                              title='Label Distributions'):
    """
    绘制标签分布图（y_node 奇异点 + y_edge PSL）。

    Args:
        data: PyG Data 对象
        figsize: 图大小
        title: 图表标题
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 节点标签：is_singularity
    y_node = _to_numpy(data, 'y_node')
    ax1.hist(y_node, bins=50, density=True, alpha=0.7, color='coral',
             edgecolor='white', linewidth=0.3)
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=0.8, label='threshold=0.5')
    ax1.set_xlabel('is_singularity')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Node: Singularity (mean={y_node.mean():.3f}, >0.5: {(y_node>0.5).mean()*100:.1f}%)')
    ax1.legend(fontsize=8)

    # 边标签：close_to_psl
    y_edge = _to_numpy(data, 'y_edge')
    ax2.hist(y_edge, bins=50, density=True, alpha=0.7, color='mediumseagreen',
             edgecolor='white', linewidth=0.3)
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=0.8, label='threshold=0.5')
    ax2.set_xlabel('close_to_psl')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Edge: close_to_psl (mean={y_edge.mean():.3f}, >0.5: {(y_edge>0.5).mean()*100:.1f}%)')
    ax2.legend(fontsize=8)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_magnitude_comparison(data, ax=None, title='m1_abs vs m2_abs',
                               point_size=1, alpha=0.5):
    """
    绘制 m1_abs 与 m2_abs 的散点对比图（每个节点一个点）。

    Args:
        data: PyG Data 对象
        ax: matplotlib Axes
        title: 图表标题
        point_size: 散点大小
        alpha: 透明度
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    m1 = _to_numpy(data, IDX_M1_ABS)
    m2 = _to_numpy(data, IDX_M2_ABS)

    ax.scatter(m1, m2, s=point_size, alpha=alpha, c='steelblue', marker='.')
    ax.plot([0, max(m1.max(), m2.max())], [0, max(m1.max(), m2.max())],
            'r--', linewidth=0.8, label='m1 = m2')
    ax.set_xlabel('m1_abs (principal moment)')
    ax.set_ylabel('m2_abs (secondary moment)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    return ax


# =============================================================================
# 综合仪表板
# =============================================================================

def visualize_dashboard(data, save_path=None, dpi=150, suptitle='Slab Stress Field Dashboard'):
    """
    生成完整的数据仪表板——一张大图包含所有关键可视化。

    ┌─────────────────────────────────────────────────┐
    │  ① 网格拓扑      │  ② 应力方向场（箭头）       │
    ├───────────────────┼──────────────────────────────┤
    │  ③ 奇异点热力图  │  ④ PSL 热力图（节点）        │
    ├───────────────────┼──────────────────────────────┤
    │  ⑤ m1_abs 热力图 │  ⑥ 边 close_to_psl 热力图    │
    ├───────────────────┴──────────────────────────────┤
    │  ⑦ 标签分布           │  ⑧ m1 vs m2 散点        │
    └─────────────────────────────────────────────────┘

    Args:
        data: PyG Data 对象
        save_path: 保存路径，为 None 时仅显示
        dpi: 分辨率
        suptitle: 总标题

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(24, 28))
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)

    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.30,
                  height_ratios=[1, 1, 1, 0.7])

    # ① 网格拓扑
    ax1 = fig.add_subplot(gs[0, 0])
    plot_mesh_topology(data, ax=ax1, title='Mesh Topology (edges + nodes)')

    # ② 应力方向场
    ax2 = fig.add_subplot(gs[0, 1])
    plot_stress_direction_field(data, ax=ax2, title='Principal Stress Directions',
                                 arrow_spacing=600)

    # ③ 奇异点热力图
    ax3 = fig.add_subplot(gs[1, 0])
    plot_heatmap(data, 'y_node', ax=ax3, title='Node: Singularity Probability',
                 cmap=COLORMAP_SINGULARITY, vmin=0, vmax=1)

    # ④ PSL 热力图（节点级）
    ax4 = fig.add_subplot(gs[1, 1])
    plot_heatmap(data, IDX_M1_ABS, ax=ax4, title='m1_abs Magnitude',
                 cmap=COLORMAP_MAGNITUDE)

    # ⑤ m2_abs 热力图
    ax5 = fig.add_subplot(gs[2, 0])
    plot_heatmap(data, IDX_M2_ABS, ax=ax5, title='m2_abs Magnitude',
                 cmap=COLORMAP_MAGNITUDE)

    # ⑥ 边 PSL 热力图
    ax6 = fig.add_subplot(gs[2, 1])
    plot_edge_heatmap(data, ax=ax6, title='Edge: close_to_psl',
                      cmap=COLORMAP_PSL, line_width=0.4, alpha=0.4)

    # ⑦ 标签分布
    ax7 = fig.add_subplot(gs[3, 0])
    y_node = _to_numpy(data, 'y_node')
    y_edge = _to_numpy(data, 'y_edge')
    ax7.hist(y_node, bins=50, alpha=0.5, color='coral', label='node: is_singularity',
             density=True, edgecolor='white', linewidth=0.2)
    ax7.hist(y_edge, bins=50, alpha=0.5, color='mediumseagreen', label='edge: close_to_psl',
             density=True, edgecolor='white', linewidth=0.2)
    ax7.axvline(0.5, color='red', linestyle='--', linewidth=0.8)
    ax7.set_xlabel('Probability')
    ax7.set_ylabel('Density')
    ax7.set_title('Label Distributions')
    ax7.legend(fontsize=8)

    # ⑧ m1 vs m2
    ax8 = fig.add_subplot(gs[3, 1])
    plot_magnitude_comparison(data, ax=ax8, title='m1_abs vs m2_abs', point_size=0.5)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'Dashboard saved to: {save_path}')

    return fig


# =============================================================================
# 增强前后对比
# =============================================================================

def visualize_augmentation_comparison(orig_data, aug_data_list, labels=None,
                                       save_path=None, dpi=150, figsize=None):
    """
    对比原始数据与多种增强变体的可视化。

    布局：第一行原始数据，后续每行一个增强变体。
    每行显示：网格拓扑 | 应力方向场 | 奇异点热力图 | m1_abs 热力图

    Args:
        orig_data: 原始 PyG Data 对象
        aug_data_list: 增强后 Data 对象列表
        labels: 各增强的标签列表（可选）
        save_path: 保存路径
        dpi: 分辨率
        figsize: 手动指定图大小

    Returns:
        matplotlib Figure
    """
    n_aug = len(aug_data_list)
    n_rows = 1 + n_aug  # 原始 + N 个增强
    n_cols = 4

    if figsize is None:
        figsize = (22, 5.5 * n_rows)

    if labels is None:
        labels = [f'Augmented #{i+1}' for i in range(n_aug)]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # ---- 原始数据行 ----
    _plot_comparison_row(orig_data, axes[0], row_label='Original', is_first_row=True)

    # ---- 增强数据行 ----
    for i, (aug_data, label) in enumerate(zip(aug_data_list, labels)):
        _plot_comparison_row(aug_data, axes[i + 1], row_label=label, is_first_row=False)

    fig.suptitle('Augmentation Comparison: Original vs Augmented Variants',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'Comparison saved to: {save_path}')

    return fig


def _plot_comparison_row(data, axes_row, row_label='', is_first_row=False):
    """绘制对比图中的一行"""
    titles = ['Mesh Topology', 'Stress Directions', 'Singularity Heatmap', 'm1_abs Heatmap']
    if is_first_row:
        titles = [f'{t}\n(Original)' for t in titles]
    else:
        titles = [f'{t}\n({row_label})' for t in titles]

    # 图1: 网格拓扑
    plot_mesh_topology(data, ax=axes_row[0], title=titles[0],
                       edge_alpha=0.05, node_size=1, node_color='#1f77b4')

    # 图2: 应力方向场
    plot_stress_direction_field(data, ax=axes_row[1], title=titles[1],
                                 arrow_spacing=800, arrow_scale=0.012, arrow_width=0.0015)

    # 图3: 奇异点热力图
    plot_heatmap(data, 'y_node', ax=axes_row[2], title=titles[2],
                 cmap=COLORMAP_SINGULARITY, vmin=0, vmax=1, point_size=1)

    # 图4: m1_abs 热力图
    plot_heatmap(data, IDX_M1_ABS, ax=axes_row[3], title=titles[3],
                 cmap=COLORMAP_MAGNITUDE, point_size=1)


# =============================================================================#                                                                                     # 批量数据集统计# =============================================================================
# 批量数据集统计
# =============================================================================

def visualize_dataset_statistics(data_list, save_path=None, dpi=150,
                                  figsize=(20, 18)):
    """
    对整个数据集的统计信息进行可视化。

    包含：
    - 各图节点数/边数分布
    - 各特征在全数据集上的分布
    - 各图标签均值的分布

    Args:
        data_list: PyG Data 对象列表
        save_path: 保存路径
        dpi: 分辨率
        figsize: 图大小

    Returns:
        matplotlib Figure
    """
    n_graphs = len(data_list)

    # 收集全局统计量
    num_nodes_list = []
    num_edges_list = []
    node_label_means = []
    edge_label_means = []
    m1_abs_means = []
    m2_abs_means = []

    for data in data_list:
        num_nodes_list.append(data.x.shape[0])
        num_edges_list.append(data.edge_index.shape[1] // 2)
        node_label_means.append(data.y_node.mean().item())
        edge_label_means.append(data.y_edge.mean().item())
        m1_abs_means.append(data.x[:, IDX_M1_ABS].mean().item())
        m2_abs_means.append(data.x[:, IDX_M2_ABS].mean().item())

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    # ① 节点数分布
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(n_graphs), sorted(num_nodes_list), color='steelblue', alpha=0.7)
    ax1.axhline(np.mean(num_nodes_list), color='red', linestyle='--', linewidth=1,
                label=f'mean={np.mean(num_nodes_list):.0f}')
    ax1.set_xlabel('Graph index (sorted)')
    ax1.set_ylabel('Num nodes')
    ax1.set_title('Nodes per Graph')
    ax1.legend(fontsize=8)

    # ② 边数分布
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(n_graphs), sorted(num_edges_list), color='seagreen', alpha=0.7)
    ax2.axhline(np.mean(num_edges_list), color='red', linestyle='--', linewidth=1,
                label=f'mean={np.mean(num_edges_list):.0f}')
    ax2.set_xlabel('Graph index (sorted)')
    ax2.set_ylabel('Num edges')
    ax2.set_title('Edges per Graph')
    ax2.legend(fontsize=8)

    # ③ 节点 vs 边数散点
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(num_nodes_list, num_edges_list, c='steelblue', alpha=0.6, s=20, marker='.')
    ax3.set_xlabel('Num nodes')
    ax3.set_ylabel('Num edges')
    ax3.set_title('Nodes vs Edges')

    # ④ 节点标签均值分布
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(node_label_means, bins=30, color='coral', alpha=0.7, edgecolor='white')
    ax4.set_xlabel('Mean is_singularity')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Mean Singularity per Graph (μ={np.mean(node_label_means):.4f})')

    # ⑤ 边标签均值分布
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(edge_label_means, bins=30, color='mediumseagreen', alpha=0.7, edgecolor='white')
    ax5.set_xlabel('Mean close_to_psl')
    ax5.set_ylabel('Count')
    ax5.set_title(f'Mean close_to_psl per Graph (μ={np.mean(edge_label_means):.4f})')

    # ⑥ 力值均值分布
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(m1_abs_means, bins=30, alpha=0.5, color='red', label='m1_abs', edgecolor='white')
    ax6.hist(m2_abs_means, bins=30, alpha=0.5, color='blue', label='m2_abs', edgecolor='white')
    ax6.set_xlabel('Mean magnitude')
    ax6.set_ylabel('Count')
    ax6.set_title('Mean Moment Magnitude per Graph')
    ax6.legend(fontsize=8)

    # ⑦-⑨ 各特征维度（取第一个图的分布）
    if n_graphs > 0:
        sample_data = data_list[0]
        x_all = sample_data.x.cpu().numpy()

        # ⑦ m1 方向分布
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter(x_all[:, IDX_M1_VX], x_all[:, IDX_M1_VY],
                    c=x_all[:, IDX_M1_ABS], cmap='Reds', alpha=0.3, s=1, marker='.')
        ax7.set_xlabel('m1_vx')
        ax7.set_ylabel('m1_vy')
        ax7.set_title(f'm1 Direction (sample graph, N={x_all.shape[0]})')
        ax7.set_aspect('equal')

        # ⑧ m2 方向分布
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.scatter(x_all[:, IDX_M2_VX], x_all[:, IDX_M2_VY],
                    c=x_all[:, IDX_M2_ABS], cmap='Blues', alpha=0.3, s=1, marker='.')
        ax8.set_xlabel('m2_vx')
        ax8.set_ylabel('m2_vy')
        ax8.set_title(f'm2 Direction (sample graph, N={x_all.shape[0]})')
        ax8.set_aspect('equal')

        # ⑨ 力值分布（样本图）
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.hist(x_all[:, IDX_M1_ABS], bins=50, alpha=0.5, color='red', label='m1_abs',
                 density=True, edgecolor='white', linewidth=0.2)
        ax9.hist(x_all[:, IDX_M2_ABS], bins=50, alpha=0.5, color='blue', label='m2_abs',
                 density=True, edgecolor='white', linewidth=0.2)
        ax9.set_xlabel('Magnitude')
        ax9.set_ylabel('Density')
        ax9.set_title('Magnitude Distribution (sample graph)')
        ax9.legend(fontsize=8)

    fig.suptitle(f'Dataset Statistics ({n_graphs} graphs)', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'Statistics saved to: {save_path}')

    return fig


# =============================================================================
# 快捷入口：一键可视化
# =============================================================================

def quick_visualize(data_or_path, output_dir='./visualization_output', prefix='slab'):
    """
    一键生成所有可视化图表。

    Args:
        data_or_path: PyG Data 对象、Data 列表，或 .pt 文件路径
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    if isinstance(data_or_path, str):
        data_list = torch.load(data_or_path, weights_only=False)
        print(f'Loaded {len(data_list)} graphs from {data_or_path}')
    elif isinstance(data_or_path, list):
        data_list = data_or_path
    else:
        # 单个 Data 对象
        data_list = [data_or_path]

    sample = data_list[0]

    # 1. 单图仪表板
    visualize_dashboard(sample, save_path=os.path.join(output_dir, f'{prefix}_dashboard.png'))

    # 2. 特征分布
    fig_feat = plot_feature_histograms(sample)
    fig_feat.savefig(os.path.join(output_dir, f'{prefix}_feature_hists.png'),
                     dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig_feat)

    # 3. 标签分布
    fig_labels = plot_label_distributions(sample)
    fig_labels.savefig(os.path.join(output_dir, f'{prefix}_label_distributions.png'),
                       dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig_labels)

    # 4. 数据集统计（多图时）
    if len(data_list) > 1:
        visualize_dataset_statistics(data_list,
                                      save_path=os.path.join(output_dir, f'{prefix}_dataset_stats.png'))

    # 5. 增强对比（如果有 augment 模块）
    try:
        from utils.augment import StressFieldAugmentor, AUGMENT_CONFIG
        augmentor = StressFieldAugmentor(AUGMENT_CONFIG)
        aug_variants = [augmentor(sample.clone()) for _ in range(2)]
        visualize_augmentation_comparison(
            sample, aug_variants,
            labels=['Augmented #1', 'Augmented #2'],
            save_path=os.path.join(output_dir, f'{prefix}_aug_comparison.png')
        )
    except ImportError:
        pass

    print(f'\nAll visualizations saved to: {output_dir}/')
    plt.close('all')


# =============================================================================
# 自检测试
# =============================================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print('Visualization module self-test')
    print('=' * 60)

    # 加载真实数据
    data_path = 'd:/composite_0602/graph_dataset/20260602_104_pro.pt'
    if not os.path.exists(data_path):
        # 尝试相对路径
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'graph_dataset', '20260602_104_pro.pt')

    if os.path.exists(data_path):
        data_list = torch.load(data_path, weights_only=False)
        print(f'Loaded {len(data_list)} graphs')
        sample = data_list[0]
        print(f'Sample: {sample.x.shape[0]} nodes, {sample.edge_index.shape[1]//2} edges')

        # 测试生成所有图表
        quick_visualize(data_list, output_dir='./visualization_output', prefix='test')
    else:
        print('Data file not found, generating with random data...')
        # 用随机数据测试
        N, E = 500, 1500
        dummy_x = torch.rand(N, 12)
        for ci, cj in [(2, 3), (7, 8)]:
            norms = dummy_x[:, [ci, cj]].norm(dim=1, keepdim=True)
            dummy_x[:, [ci, cj]] = dummy_x[:, [ci, cj]] / norms.clamp(min=1e-8)
        half_e = torch.randint(0, N, (2, E // 2))
        edge_index = torch.cat([half_e, half_e.flip(0)], dim=1)
        from torch_geometric.data import Data
        data = Data(x=dummy_x, edge_index=edge_index,
                    y_node=torch.rand(N, 1), y_edge=torch.rand(E, 1))
        quick_visualize(data, output_dir='./visualization_output', prefix='test_dummy')

    print('Done!')
