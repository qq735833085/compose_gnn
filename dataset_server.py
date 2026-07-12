# dataset_server.py — 启动本地服务器，H5 页面可直接读取增强数据集 CSV
import os, sys, json, csv, math
from http.server import HTTPServer, SimpleHTTPRequestHandler
from glob import glob
from urllib.parse import urlparse, parse_qs

BASE = os.path.dirname(os.path.abspath(__file__))
AUG_DIR = os.path.join(BASE, 'datasets', '04_augmented')
PORT = 8765


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        # API: 列出所有案例
        if parsed.path == '/api/cases':
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

                # 快速读取 PSL 计数（只读边 CSV 的最后两列）
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
                        n_nodes = sum(1 for _ in f) - 1  # minus header
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
            return

        # API: 获取 CSV 摘要（前 N 行 + 统计）
        if parsed.path.startswith('/api/preview/'):
            rel = parsed.path.replace('/api/preview/', '')
            filepath = os.path.join(BASE, rel)
            n = int(parse_qs(parsed.query).get('n', [2000])[0])
            if os.path.exists(filepath) and filepath.endswith('.csv'):
                self.send_csv_preview(filepath, n, parsed)
                return

        # 静态文件
        super().do_GET()

    def send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def send_csv_preview(self, filepath, n, parsed):
        """返回 CSV 的采样数据 + 列统计"""
        # 支持 cols 参数过滤列
        req_cols = parse_qs(parsed.query).get('cols', [None])[0]
        if req_cols:
            req_cols = req_cols.split(',')

        rows = []
        stats = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            if req_cols:
                # 确保请求的列存在
                req_cols = [c for c in req_cols if c in cols]
                cols = req_cols

            for c in cols:
                stats[c] = {'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'count': 0}

            all_rows = list(reader)
            total = len(all_rows)

            # 均匀采样 n 行
            if total <= n:
                sample = all_rows
            else:
                step = max(1, total // n)
                sample = [all_rows[i] for i in range(0, total, step)][:n]

            # 如果指定了列，只保留指定列
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

        # 简化统计
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

    def log_message(self, format, *args):
        if '/api/' in str(args[0]):
            print(f'  API: {args[0]}')


def main():
    print(f'''
  Slab Dataset Viewer Server
  http://localhost:{PORT}/dataset_viewer.html

  增强数据目录: {AUG_DIR}
  案例数: {len(glob(os.path.join(AUG_DIR, "case_*")))}

  按 Ctrl+C 停止
''')
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nServer stopped.')


if __name__ == '__main__':
    main()
