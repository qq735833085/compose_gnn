# export_dataset_json.py — 导出数据集摘要 JSON，供 H5 可视化页面使用
import os, sys, json, torch, numpy as np, pandas as pd
from glob import glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_overview.json')


def downsample(arr, n):
    """均匀下采样到 n 个点"""
    if len(arr) <= n:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, n, dtype=int)
    return arr[idx].tolist()


def process_pt_file(pt_path, label, n_sample_nodes=2000, n_sample_edges=1500):
    """从 .pt 文件提取可视化数据"""
    data_list = torch.load(pt_path, weights_only=False)
    n_total = len(data_list)
    samples = []

    for i, data in enumerate(data_list):
        x = data.x.cpu().numpy()
        ei = data.edge_index.cpu().numpy()
        yn = data.y_node.cpu().numpy().ravel()
        ye = data.y_edge.cpu().numpy().ravel()

        n_nodes = x.shape[0]
        n_edges = ei.shape[1] // 2

        # 下采样节点（全量节点坐标 + 随机采样用于可视化）
        if n_nodes > n_sample_nodes:
            vis_idx = np.sort(np.random.choice(n_nodes, n_sample_nodes, replace=False))
        else:
            vis_idx = np.arange(n_nodes)

        # 下采样边
        if n_edges > n_sample_edges:
            edge_idx = np.sort(np.random.choice(n_edges, n_sample_edges, replace=False))
        else:
            edge_idx = np.arange(n_edges)

        # 提取边线段
        edge_segments = []
        for j in edge_idx:
            s, e = ei[0, j*2], ei[1, j*2]
            edge_segments.append([float(x[s, 0]), float(x[s, 1]),
                                  float(x[e, 0]), float(x[e, 1])])

        samples.append({
            'idx': i,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'coords_x': x[vis_idx, 0].tolist(),
            'coords_y': x[vis_idx, 1].tolist(),
            'm1_abs': x[vis_idx, 4].tolist(),    # idx 4 = m1_abs
            'm2_abs': x[vis_idx, 9].tolist(),    # idx 9 = m2_abs
            'm1_vx': x[vis_idx, 2].tolist(),
            'm1_vy': x[vis_idx, 3].tolist(),
            'm2_vx': x[vis_idx, 7].tolist(),
            'm2_vy': x[vis_idx, 8].tolist(),
            'y_node': yn[vis_idx].tolist(),
            'edge_segments': edge_segments,       # [[x1,y1,x2,y2], ...]
            'edge_psl': ye[edge_idx * 2].tolist() if len(ye) > 0 else [],
        })

    # 全局统计
    n_nodes_all = [d.x.shape[0] for d in data_list]
    n_edges_all = [d.edge_index.shape[1] // 2 for d in data_list]
    yn_mean = [d.y_node.mean().item() for d in data_list]
    ye_mean = [d.y_edge.mean().item() for d in data_list]
    m1_mean = [d.x[:, 4].mean().item() for d in data_list]
    m2_mean = [d.x[:, 9].mean().item() for d in data_list]

    return {
        'label': label,
        'n_total': n_total,
        'samples': samples,
        'stats': {
            'nodes': {'min': min(n_nodes_all), 'max': max(n_nodes_all), 'mean': np.mean(n_nodes_all)},
            'edges': {'min': min(n_edges_all), 'max': max(n_edges_all), 'mean': np.mean(n_edges_all)},
            'y_node_mean': {'min': min(yn_mean), 'max': max(yn_mean), 'mean': np.mean(yn_mean)},
            'y_edge_mean': {'min': min(ye_mean), 'max': max(ye_mean), 'mean': np.mean(ye_mean)},
            'm1_abs_mean': float(np.mean(m1_mean)),
            'm2_abs_mean': float(np.mean(m2_mean)),
        }
    }


def process_merged_csv(merged_dir):
    """从 merged_data CSV 提取统计"""
    cases = []
    for nf in sorted(glob(os.path.join(merged_dir, '*_merged_nodes.csv'))):
        case = os.path.basename(nf).replace('_merged_nodes.csv', '')
        dn = pd.read_csv(nf)
        ef_path = os.path.join(merged_dir, f'{case}_merged_edges.csv')
        de = pd.read_csv(ef_path) if os.path.exists(ef_path) else None

        n_nodes = len(dn)
        n_edges = len(de) if de is not None else 0
        n_sing = int(dn['is_singularity'].sum())
        n_psl1 = int(de['is_psl_1'].sum()) if de is not None else 0
        n_psl2 = int(de['is_psl_2'].sum()) if de is not None else 0
        yn_mean1 = float(dn['is_on_psl_raw_1'].mean())
        yn_mean2 = float(dn['is_on_psl_raw_2'].mean())

        cases.append({
            'name': case,
            'n_nodes': n_nodes, 'n_edges': n_edges,
            'n_singularity': n_sing,
            'n_psl1': n_psl1, 'n_psl2': n_psl2,
            'psl_prob_mean1': round(yn_mean1, 4),
            'psl_prob_mean2': round(yn_mean2, 4),
        })
    return cases


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    print('Exporting dataset overview JSON...')

    result = {}

    # 1. 合并数据统计
    merged_dir = os.path.join(base, 'merged_data')
    if os.path.exists(merged_dir):
        result['merged_cases'] = process_merged_csv(merged_dir)
        print(f'  Merged cases: {len(result["merged_cases"])}')

    # 2. 原始 .pt 数据（采样节点+边用于可视化）
    pt_orig = os.path.join(base, 'graph_dataset', 'merged_25cases.pt')
    if os.path.exists(pt_orig):
        result['dataset_original'] = process_pt_file(pt_orig, 'Original (25 cases)')
        print(f'  Original .pt: {result["dataset_original"]["n_total"]} graphs')

    # 3. 增强 .pt 数据
    pt_aug = os.path.join(base, 'graph_dataset', 'merged_25cases_augmented_x7.pt')
    if os.path.exists(pt_aug):
        # 只导出前 30 个样本（避免 JSON 过大）：10 orig + 20 aug
        data_list = torch.load(pt_aug, weights_only=False)
        subset = [data_list[i] for i in range(min(30, len(data_list)))]
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), 'tmp_subset.pt')
        torch.save(subset, tmp_path)
        result['dataset_augmented'] = process_pt_file(tmp_path, 'Augmented (200 cases, showing 30)')
        result['dataset_augmented']['n_total'] = 200
        os.remove(tmp_path)
        print(f'  Augmented .pt: 200 graphs (showing 30 samples)')

    # 4. 全局摘要
    if 'merged_cases' in result:
        cases = result['merged_cases']
        result['summary'] = {
            'n_cases': len(cases),
            'total_nodes': sum(c['n_nodes'] for c in cases),
            'total_edges': sum(c['n_edges'] for c in cases),
            'total_singularities': sum(c['n_singularity'] for c in cases),
            'total_psl1_edges': sum(c['n_psl1'] for c in cases),
            'total_psl2_edges': sum(c['n_psl2'] for c in cases),
            'case_names': [c['name'] for c in cases],
        }

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f'\nSaved: {OUTPUT} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
