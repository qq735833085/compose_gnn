# build_augmented_dataset.py
# =============================================================================
# 静态数据增强数据集构建脚本
#
# 功能：加载已有 .pt 图数据集，对每个原始图生成 N 个增强变体，
#       合并保存为新的 .pt 文件，训练时直接加载，无需运行时增强。
#
# 用法：
#   python build_augmented_dataset.py                          # 默认配置
#   python build_augmented_dataset.py --n-augment 5            # 每图 5 个变体
#   python build_augmented_dataset.py --input path.pt --output path_aug.pt
#   python build_augmented_dataset.py --config conservative    # 保守增强模式
# =============================================================================

import os
import sys
import argparse
import torch
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.augment import (
    StressFieldAugmentor, AUGMENT_CONFIG, CONSERVATIVE_CONFIG, augment_dataset
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build augmented .pt dataset for slab stress GNN training')

    parser.add_argument('--input', type=str,
                        default='graph_dataset/20260602_104_pro.pt',
                        help='Input .pt file path')

    parser.add_argument('--output', type=str, default=None,
                        help='Output .pt file path (default: {input}_augmented_N.pt)')

    parser.add_argument('--n-augment', type=int, default=3,
                        help='Number of augmented variants per original graph (default: 3)')

    parser.add_argument('--config', type=str, default='default',
                        choices=['default', 'conservative'],
                        help='Augmentation config preset')

    parser.add_argument('--keep-originals', action='store_true', default=True,
                        help='Include original graphs in output (default: True)')
    parser.add_argument('--no-originals', action='store_false', dest='keep_originals',
                        help='Exclude original graphs, only keep augmented variants')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for tensor operations (default: cpu)')

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 验证输入 ----
    if not os.path.exists(args.input):
        print(f'Error: Input file not found: {args.input}')
        sys.exit(1)

    # ---- 选择配置 ----
    if args.config == 'conservative':
        config = CONSERVATIVE_CONFIG
        print('Using CONSERVATIVE_CONFIG')
    else:
        config = AUGMENT_CONFIG
        print(f'Using AUGMENT_CONFIG (rotation mode: {config["rotation"]["angle_mode"]})')

    # ---- 自动生成输出路径 ----
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f'{base}_augmented_x{args.n_augment}{ext}'

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # ---- 设置随机种子 ----
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 加载原始数据 ----
    print(f'\nLoading: {args.input}')
    original_list = torch.load(args.input, weights_only=False)
    n_orig = len(original_list)
    print(f'  {n_orig} original graphs loaded')

    # 统计原始数据规模
    total_nodes_orig = sum(d.x.shape[0] for d in original_list)
    total_edges_orig = sum(d.edge_index.shape[1] // 2 for d in original_list)
    print(f'  Avg nodes/graph: {total_nodes_orig/n_orig:.0f}, avg edges/graph: {total_edges_orig/n_orig:.0f}')

    # ---- 创建增强器 ----
    augmentor = StressFieldAugmentor(config)

    # ---- 批量增强 ----
    print(f'\nAugmenting: {n_orig} graphs x {args.n_augment} variants = {n_orig * args.n_augment} new graphs...')

    augmented_list = []
    if args.keep_originals:
        augmented_list.extend(original_list)  # 保留原始数据

    for orig_idx, data in enumerate(tqdm(original_list, desc='Augmenting')):
        for aug_idx in range(args.n_augment):
            aug_data = augmentor(data.clone())
            augmented_list.append(aug_data)

    n_total = len(augmented_list)
    n_aug_only = n_total - (n_orig if args.keep_originals else 0)

    # ---- 验证 ----
    print(f'\nVerification:')
    print(f'  Total graphs: {n_total} (originals: {n_orig if args.keep_originals else 0}, augmented: {n_aug_only})')

    # 检查所有图的完整性
    errors = []
    for i, data in enumerate(tqdm(augmented_list, desc='Verifying')):
        if data.x.shape[0] == 0:
            errors.append(f'Graph {i}: empty node features')
        if data.edge_index.shape[1] == 0:
            errors.append(f'Graph {i}: empty edge index')
        if data.y_node.shape[0] != data.x.shape[0]:
            errors.append(f'Graph {i}: y_node/node count mismatch')
        if data.y_edge.shape[0] != data.edge_index.shape[1]:
            errors.append(f'Graph {i}: y_edge/edge count mismatch')
        # 检查力值非负
        if (data.x[:, 4] < 0).any() or (data.x[:, 9] < 0).any():
            errors.append(f'Graph {i}: negative magnitude values')
        # 检查坐标范围
        x_coords = data.x[:, 0]
        y_coords = data.x[:, 1]
        if x_coords.min() < -0.01 or x_coords.max() > 1.01:
            errors.append(f'Graph {i}: x out of [0,1]: [{x_coords.min():.4f}, {x_coords.max():.4f}]')
        if y_coords.min() < -0.01 or y_coords.max() > 1.01:
            errors.append(f'Graph {i}: y out of [0,1]: [{y_coords.min():.4f}, {y_coords.max():.4f}]')

    if errors:
        print(f'  ERRORS FOUND ({len(errors)}):')
        for e in errors[:10]:
            print(f'    - {e}')
        if len(errors) > 10:
            print(f'    ... and {len(errors) - 10} more')
    else:
        print(f'  All {n_total} graphs verified OK')

    # ---- 保存 ----
    print(f'\nSaving to: {args.output}')
    torch.save(augmented_list, args.output)

    # ---- 文件大小 ----
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f'  File size: {file_size_mb:.1f} MB')

    # ---- 汇总 ----
    print(f'\n{"="*60}')
    print(f'Dataset augmentation complete!')
    print(f'  Input:  {args.input} ({n_orig} graphs)')
    print(f'  Output: {args.output} ({n_total} graphs)')
    print(f'  Expansion ratio: {n_total/n_orig:.1f}x')
    if args.keep_originals:
        print(f'  Composition: {n_orig} originals + {n_aug_only} augmented')
    print(f'{"="*60}')

    # ---- 使用提示 ----
    print(f'\nUsage in training script:')
    print(f'  dataset = torch.load("{args.output}", weights_only=False)')
    if args.keep_originals:
        print(f'  # NOTE: Originals are first {n_orig} graphs, augmented follow')
        print(f'  # Use random shuffle in DataLoader to mix them')
    print()


if __name__ == '__main__':
    main()
