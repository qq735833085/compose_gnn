# dataset_server.py — 本地 HTTP 服务器，支持二值 + 连续概率数据集浏览
import os, sys, json, csv, math
from http.server import HTTPServer, SimpleHTTPRequestHandler
from glob import glob
from urllib.parse import urlparse, parse_qs

import torch
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
AUG_DIR = os.path.join(BASE, 'datasets', '04_augmented')
CONTINUOUS_PATH = os.path.join(BASE, 'datasets', '03_graph', 'merged_25cases_continuous.pt')
PORT = 8765

# ---- 预加载连续概率数据集 ----
CONTINUOUS_DATA = None
if os.path.exists(CONTINUOUS_PATH):
    print(f'Loading continuous dataset: {CONTINUOUS_PATH}')
    CONTINUOUS_DATA = torch.load(CONTINUOUS_PATH, weights_only=False)
    print(f'  {len(CONTINUOUS_DATA)} graphs loaded')
else:
    print(f'[WARN] Continuous dataset not found: {CONTINUOUS_PATH}')


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        # ==================== 二值数据集 API ====================

        if parsed.path == '/api/cases':
            self.list_binary_cases()
            return

        if parsed.path.startswith('/api/preview/'):
            rel = parsed.path.replace('/api/preview/', '')
            filepath = os.path.join(BASE, rel)
            n = int(parse_qs(parsed.query).get('n', [2000])[0])
            if os.path.exists(filepath) and filepath.endswith('.csv'):
                self.send_csv_preview(filepath, n, parsed)
                return

        # ==================== 连续概率数据集 API ====================

        if parsed.path == '/api/continuous/info':
            self.continuous_info()
            return

        if parsed.path.startswith('/api/continuous/graph/'):
            idx_str = parsed.path.replace('/api/continuous/graph/', '')
            try:
                idx = int(idx_str)
            except ValueError:
                self.send_error(400, 'Invalid graph index')
                return
            self.send_continuous_graph(idx)
            return

        # 静态文件
        super().do_GET()

    # ==================== 二值数据集 ====================

    def list_binary_cases(self):
        dirs = sorted(glob(os.path.join(AUG_DIR, 'case_*')))
        cases = []
        for d in dirs:
            name = os.path.basename(d)
            nf = os.path.join(d, 'nodes.csv')
            ef = os.path.join(d, 'edges.csv')
            parts = name.split('_')
            orig_idx = int(parts[1]) if len(parts) > 1 else 0
            is_orig = 'orig' in name
            aug_idx = int(parts[-1]) if 'aug' in name else -1

            psl1_count = 0; psl2_count = 0; n_nodes = 0
            try:
                with open(ef, 'r') as f:
                    header = f.readline().strip().split(',')
                    col1 = header.index('is_psl_1') if 'is_psl_1' in header else -1
                    col2 = header.index('is_psl_2') if 'is_psl_2' in header else -1
                    for line in f:
                        parts_line = line.strip().split(',')
                        if col1 >= 0 and len(parts_line) > col1 and parts_line[col1] == '1':
                            psl1_count += 1
                        if col2 >= 0 and len(parts_line) > col2 and parts_line[col2] == '1':
                            psl2_count += 1
            except: pass
            try:
                with open(nf, 'r') as f:
                    n_nodes = sum(1 for _ in f) - 1
            except: pass

            cases.append({
                'name': name,
                'orig_idx': orig_idx,
                'is_original': is_orig,
                'aug_idx': aug_idx,
                'n_nodes': n_nodes,
                'psl1_count': psl1_count,
                'psl2_count': psl2_count,
                'nodes_path': f'/augmented_data/{name}/nodes.csv',
                'edges_path': f'/augmented_data/{name}/edges.csv',
            })
        self.send_json(cases)

    def send_csv_preview(self, filepath, n, parsed):
        req_cols = parse_qs(parsed.query).get('cols', [None])[0]
        if req_cols:
            req_cols = req_cols.split(',')

        rows = []
        stats = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            if req_cols:
                req_cols = [c for c in req_cols if c in cols]
                cols = req_cols

            for c in cols:
                stats[c] = {'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'count': 0}

            all_rows = list(reader)
            total = len(all_rows)

            if total <= n:
                sample = all_rows
            else:
                step = max(1, total // n)
                sample = [all_rows[i] for i in range(0, total, step)][:n]

            if req_cols:
                sample = [{c: row.get(c, '') for c in req_cols} for row in sample]

            for row in all_rows:
                for c in cols:
                    try:
                        v = float(row[c])
                        stats[c]['min'] = min(stats[c]['min'], v)
                        stats[c]['max'] = max(stats[c]['max'], v)
                        stats[c]['sum'] += v
                        stats[c]['count'] += 1
                    except (ValueError, TypeError):
                        pass

        simple_stats = {}
        for c, s in stats.items():
            if s['count'] > 0:
                simple_stats[c] = {
                    'min': round(s['min'], 6),
                    'max': round(s['max'], 6),
                    'mean': round(s['sum'] / s['count'], 6),
                }

        self.send_json({
            'columns': cols,
            'total_rows': total,
            'sample': sample,
            'stats': simple_stats,
        })

    # ==================== 连续概率数据集 ====================

    def continuous_info(self):
        """返回连续数据集概览"""
        if CONTINUOUS_DATA is None:
            self.send_json({'error': 'Continuous dataset not available'})
            return

        cases = []
        for i, g in enumerate(CONTINUOUS_DATA):
            n_nodes = g.x.shape[0]
            n_edges = g.edge_index.shape[1] // 2
            sing_prob = g.y_node.cpu().numpy().ravel()
            edge_prob = g.y_edge.cpu().numpy()  # [2E, 2]
            # 正向边 m1/m2 分别统计
            edge_prob_fwd = edge_prob[::2]  # [E, 2]
            m1_fwd = edge_prob_fwd[:, 0]
            m2_fwd = edge_prob_fwd[:, 1]

            cases.append({
                'idx': i,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'sing_max': float(sing_prob.max()),
                'sing_mean': float(sing_prob.mean()),
                'sing_nonzero_pct': float((sing_prob > 0.01).mean() * 100),
                'm1_max': float(m1_fwd.max()),
                'm1_mean': float(m1_fwd.mean()),
                'm1_nonzero_pct': float((m1_fwd > 0.01).mean() * 100),
                'm2_max': float(m2_fwd.max()),
                'm2_mean': float(m2_fwd.mean()),
                'm2_nonzero_pct': float((m2_fwd > 0.01).mean() * 100),
            })

        self.send_json({'n_graphs': len(CONTINUOUS_DATA), 'cases': cases})

    def send_continuous_graph(self, idx):
        """返回单个图的节点和边数据（用于可视化）"""
        if CONTINUOUS_DATA is None:
            self.send_json({'error': 'Continuous dataset not available'})
            return

        if idx < 0 or idx >= len(CONTINUOUS_DATA):
            self.send_error(404, f'Graph {idx} out of range (0-{len(CONTINUOUS_DATA)-1})')
            return

        g = CONTINUOUS_DATA[idx]
        x = g.x.cpu().numpy()
        edge_index = g.edge_index.cpu().numpy()
        y_node = g.y_node.cpu().numpy().ravel()
        y_edge = g.y_edge.cpu().numpy()  # [2E, 2] — 不 ravel

        N = x.shape[0]
        E = edge_index.shape[1] // 2

        # 节点数据采样（全量通常 ~11k，可全传）
        nodes = []
        for i in range(N):
            nodes.append({
                'node_id': i,
                'x': float(x[i, 0]),
                'y': float(x[i, 1]),
                'sing_prob': float(y_node[i]),
                'm1_vx': float(x[i, 2]),
                'm1_vy': float(x[i, 3]),
                'm1_abs': float(x[i, 4]),
                'm1_type': 't' if x[i, 5] > 0.5 else 'c',
                'm2_vx': float(x[i, 7]),
                'm2_vy': float(x[i, 8]),
                'm2_abs': float(x[i, 9]),
                'm2_type': 't' if x[i, 10] > 0.5 else 'c',
            })

        # 边数据（只传正向边，m1/m2双通道）
        edges = []
        for e in range(E):
            edges.append({
                'edge_id': e,
                'start_id': int(edge_index[0, e * 2]),
                'end_id': int(edge_index[1, e * 2]),
                'm1_prob': float(y_edge[e * 2, 0]) if y_edge.ndim > 1 else float(y_edge[e * 2]),
                'm2_prob': float(y_edge[e * 2, 1]) if y_edge.ndim > 1 else 0.0,
            })

        self.send_json({
            'idx': idx,
            'n_nodes': N,
            'n_edges': E,
            'nodes': nodes,
            'edges': edges,
        })

    # ==================== 通用 ====================

    def send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if '/api/' in str(args[0]):
            print(f'  API: {args[0]}')


def main():
    n_aug = len(glob(os.path.join(AUG_DIR, 'case_*')))
    n_cont = len(CONTINUOUS_DATA) if CONTINUOUS_DATA else 0
    print(f'''
  ╔══════════════════════════════════════════════╗
  ║     Slab Dataset Viewer Server               ║
  ║     http://localhost:{PORT}/dataset_viewer.html  ║
  ╠══════════════════════════════════════════════╣
  ║  Binary cases (04_augmented): {n_aug:<4}          ║
  ║  Continuous cases (03_graph):  {n_cont:<4}          ║
  ╚══════════════════════════════════════════════╝
  Press Ctrl+C to stop
''')
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nServer stopped.')


if __name__ == '__main__':
    main()
