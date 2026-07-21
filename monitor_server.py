#!/usr/bin/env python
# monitor_server.py — Web 监控面板服务器
"""
启动 Web 监控面板:
    python monitor_server.py                    # 默认端口 8080
    python monitor_server.py --port 8888        # 自定义端口
    python monitor_server.py --host 0.0.0.0     # 允许局域网访问

然后浏览器打开 http://localhost:8080
"""

import os, sys, json, time, csv
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

BASE_DIR = Path(__file__).parent
COMPARE_DIR = BASE_DIR / "trained_model" / "v5_compare"
PHYSICS_DIR = BASE_DIR / "trained_model" / "v5_physics"

# ===================== API 逻辑 =====================

def find_experiments():
    """扫描所有实验目录，返回实验信息列表"""
    experiments = []
    for watch_dir in [COMPARE_DIR, PHYSICS_DIR]:
        if not watch_dir.exists():
            continue
        for ts_dir in sorted(watch_dir.iterdir(), reverse=True):
            if not ts_dir.is_dir():
                continue
            for exp_dir in sorted(ts_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                csv_path = exp_dir / "metrics.csv"
                if not csv_path.exists():
                    continue
                rows = _read_csv(csv_path)
                if not rows:
                    continue

                last = _parse_row(rows[-1])
                if last is None:
                    continue

                # 确定组别和架构
                rel = str(exp_dir.relative_to(BASE_DIR / "trained_model"))
                parts = rel.replace("\\", "/").split("/")
                # e.g. v5_compare/2026_07_20_2202/v5_gat_hinge
                group = parts[0] if len(parts) > 0 else "unknown"
                exp_name = exp_dir.name

                # 判断是否完成
                total_epochs = _get_total_epochs(exp_name)
                is_active = last["epoch"] < total_epochs

                # 最佳指标
                best_sing_ep, best_sing = _best_metric(rows, "sing_dice")
                best_edge_ep, best_edge = _best_metric(rows, "edge_dice")

                experiments.append({
                    "id": f"{group}/{exp_name}",
                    "group": group,
                    "name": exp_name,
                    "arch": _extract_arch(exp_name),
                    "is_active": is_active,
                    "current_epoch": last["epoch"],
                    "total_epochs": total_epochs,
                    "progress_pct": round(last["epoch"] / total_epochs * 100, 1),
                    "last_metrics": last,
                    "best_sing": {"epoch": best_sing_ep, "value": round(best_sing, 4)},
                    "best_edge": {"epoch": best_edge_ep, "value": round(best_edge, 4)},
                    "csv_path": str(csv_path),
                })

    # 按组和时间排序
    return sorted(experiments, key=lambda x: (x["group"], x["name"]))


def get_metrics_history(exp_id):
    """获取某个实验的全部 metrics 历史（供图表使用）"""
    for exp in find_experiments():
        if exp["id"] == exp_id:
            csv_path = Path(exp["csv_path"])
            if not csv_path.exists():
                return None
            rows = _read_csv(csv_path)
            history = []
            for row in rows:
                r = _parse_row(row)
                if r:
                    history.append(r)
            return history
    return None


def get_summary():
    """获取全局摘要"""
    exps = find_experiments()
    active = [e for e in exps if e["is_active"]]
    completed = [e for e in exps if not e["is_active"]]

    # 各架构对比
    arch_compare = {}
    for e in exps:
        arch = e["arch"]
        if arch not in arch_compare:
            arch_compare[arch] = e
        else:
            # 保留最新的
            pass

    return {
        "total_experiments": len(exps),
        "active_count": len(active),
        "completed_count": len(completed),
        "active": [e["name"] for e in active],
        "comparison": [
            {"arch": e["arch"], "sing_dice": e["best_sing"]["value"],
             "edge_dice": e["best_edge"]["value"],
             "sing_auc": e["last_metrics"]["sing_auc"]}
            for e in sorted(exps, key=lambda x: x["best_sing"]["value"], reverse=True)
        ],
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _read_csv(csv_path):
    rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        pass
    return rows


def _parse_row(row):
    try:
        return {
            "epoch": int(row.get("epoch", 0)),
            "d_loss": round(float(row.get("d_loss", 0)), 6),
            "g_loss": round(float(row.get("g_loss", 0)), 6),
            "recon": round(float(row.get("g_recon", 0)), 6),
            "adv": round(float(row.get("g_adv", 0)), 6),
            "r1": round(float(row.get("gp_r1", 0)), 8),
            "sing_auc": round(float(row.get("sing_auc", 0)), 4),
            "sing_dice": round(float(row.get("sing_dice", 0)), 4),
            "m1_auc": round(float(row.get("m1_auc", 0)), 4),
            "m1_dice": round(float(row.get("m1_dice", 0)), 4),
            "m2_auc": round(float(row.get("m2_auc", 0)), 4),
            "m2_dice": round(float(row.get("m2_dice", 0)), 4),
            "edge_auc": round(float(row.get("edge_auc", 0)), 4),
            "edge_dice": round(float(row.get("edge_dice", 0)), 4),
        }
    except (ValueError, KeyError):
        return None


def _best_metric(rows, key):
    best_val = -1
    best_epoch = 0
    for row in rows:
        r = _parse_row(row)
        if r and r[key] > best_val:
            best_val = r[key]
            best_epoch = r["epoch"]
    return best_epoch, best_val


def _get_total_epochs(name):
    return 300


def _extract_arch(name):
    """从实验名提取架构"""
    name_lower = name.lower()
    if "physics" in name_lower:
        if "gat" in name_lower:
            return "GAT+Phys"
        return "Physics"
    if "gat" in name_lower:
        return "GAT"
    if "gcn" in name_lower:
        return "GCN"
    if "sage" in name_lower:
        return "SAGE"
    return name


# ===================== HTTP Server =====================

HTML_PAGE = None  # cached
STATIC_FILES = {}  # cache for static files


def _load_static(path, content_type):
    if path not in STATIC_FILES:
        file_path = BASE_DIR / path
        if file_path.exists():
            with open(file_path, 'rb') as f:
                STATIC_FILES[path] = (f.read(), content_type)
    return STATIC_FILES.get(path, (None, None))


def get_html():
    global HTML_PAGE
    if HTML_PAGE is None:
        html_path = BASE_DIR / "monitor_dashboard.html"
        with open(html_path, 'r', encoding='utf-8') as f:
            HTML_PAGE = f.read()
    return HTML_PAGE


class MonitorHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # 静默日志
        pass

    def _send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html, code=200):
        body = html.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, data, content_type, code=200):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', len(data))
        self.send_header('Cache-Control', 'public, max-age=3600')
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, msg, code=404):
        self._send_json({"error": msg}, code)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        try:
            if path == '/chart.umd.min.js':
                data, ct = _load_static('chart.umd.min.js', 'application/javascript')
                if data:
                    self._send_static(data, ct)
                else:
                    self._send_error("chart.js not found", 404)
                return

            if path == '/' or path == '/index.html':
                self._send_html(get_html())

            elif path == '/api/summary':
                self._send_json(get_summary())

            elif path == '/api/experiments':
                exps = find_experiments()
                # 精简返回（不包含完整 last_metrics）
                result = []
                for e in exps:
                    result.append({
                        "id": e["id"],
                        "group": e["group"],
                        "name": e["name"],
                        "arch": e["arch"],
                        "is_active": e["is_active"],
                        "current_epoch": e["current_epoch"],
                        "total_epochs": e["total_epochs"],
                        "progress_pct": e["progress_pct"],
                        "best_sing": e["best_sing"],
                        "best_edge": e["best_edge"],
                        "last_metrics": e["last_metrics"],
                    })
                self._send_json(result)

            elif path == '/api/metrics':
                exp_id = params.get('id', [None])[0]
                if not exp_id:
                    self._send_error("Missing 'id' parameter", 400)
                    return
                history = get_metrics_history(exp_id)
                if history is None:
                    self._send_error(f"Experiment not found: {exp_id}", 404)
                    return
                self._send_json(history)

            elif path == '/api/health':
                self._send_json({"status": "ok", "time": datetime.now().isoformat()})

            else:
                self._send_error("Not Found", 404)

        except Exception as e:
            self._send_error(str(e), 500)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compose-GNN Web Monitor Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), MonitorHandler)
    print(f"\n{'='*60}")
    print(f"  Compose-GNN Training Monitor")
    print(f"  Server: http://{args.host}:{args.port}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
