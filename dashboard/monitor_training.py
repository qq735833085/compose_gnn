#!/usr/bin/env python
# monitor_training.py — 实时监控 GNN 对比实验 + 物理约束实验训练进度
"""
Usage:
    python monitor_training.py              # 监控 v5_compare 目录
    python monitor_training.py --all        # 监控所有实验目录
    python monitor_training.py --interval 10  # 每 10 秒刷新
    python monitor_training.py --once       # 只输出一次

监控内容:
  - 当前运行的实验 (GAT / GCN / SAGE / Physics)
  - 训练进度 (epoch / total)
  - 实时 Loss (D_loss, G_loss, recon, adv)
  - 验证指标 (Sing AUC/Dice, Edge AUC/Dice)
  - 历史最佳指标
"""

import os, sys, time
import csv
from datetime import datetime
from pathlib import Path

# Fix Windows GBK encoding issues (Python 3.7+)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ===================== 配置 =====================
BASE_DIR = Path(__file__).parent.parent  # 项目根目录
COMPARE_DIR = BASE_DIR / "trained_model" / "v5_compare"
PHYSICS_DIR = BASE_DIR / "trained_model" / "v5_physics"


def find_experiment_dirs(base_dir):
    """扫描目录下所有实验子目录"""
    if not base_dir.exists():
        return []
    dirs = []
    for d in base_dir.iterdir():
        if d.is_dir():
            for sub in d.iterdir():
                if sub.is_dir() and (sub / "metrics.csv").exists():
                    dirs.append(sub)
    return sorted(dirs, key=lambda x: x.name)


def read_metrics(csv_path):
    """读取 CSV 返回所有行"""
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_row(row):
    """解析一行 CSV -> 提取关键值"""
    try:
        return {
            "epoch": int(row.get("epoch", 0)),
            "d_loss": float(row.get("d_loss", 0)),
            "g_loss": float(row.get("g_loss", 0)),
            "recon": float(row.get("g_recon", 0)),
            "adv": float(row.get("g_adv", 0)),
            "r1": float(row.get("gp_r1", 0)),
            "sing_auc": float(row.get("sing_auc", 0)),
            "sing_dice": float(row.get("sing_dice", 0)),
            "m1_auc": float(row.get("m1_auc", 0)),
            "m1_dice": float(row.get("m1_dice", 0)),
            "m2_auc": float(row.get("m2_auc", 0)),
            "m2_dice": float(row.get("m2_dice", 0)),
            "edge_auc": float(row.get("edge_auc", 0)),
            "edge_dice": float(row.get("edge_dice", 0)),
        }
    except (ValueError, KeyError):
        return None


def get_best_metrics(rows, metric_key):
    """从已有数据中找最佳 epoch"""
    best_val = -1
    best_epoch = 0
    for row in rows:
        r = parse_row(row)
        if r is None:
            continue
        if r[metric_key] > best_val:
            best_val = r[metric_key]
            best_epoch = r["epoch"]
    return best_epoch, best_val


def get_trend(rows, metric_key, window=5):
    """最近 N 个 epoch 的趋势 (ASCII safe)"""
    vals = []
    for row in rows[-window:]:
        r = parse_row(row)
        if r:
            vals.append(r[metric_key])
    if len(vals) < 2:
        return " -"
    if vals[-1] > vals[0] * 1.01:
        return "UP"
    elif vals[-1] < vals[0] * 0.99:
        return "DN"
    else:
        return "=="


def format_progress_bar(pct, width=20):
    """ASCII 进度条"""
    filled = int(pct / 100 * width)
    return "[" + "=" * filled + "." * (width - filled) + "]"


def get_total_epochs(exp_name):
    """根据实验名推断总 epochs 数"""
    return 300


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def color_tag(text, trend):
    """用 ASCII 标记趋势"""
    if trend == "UP":
        return f"+{text}+"  # improving
    elif trend == "DN":
        return f"-{text}-"  # declining
    else:
        return f" {text} "


def print_separator(char="=", width=62):
    print(char * width)


def print_header():
    print("=" * 62)
    print("    Compose-GNN Training Monitor v1.0")
    print("=" * 62)


def print_experiment_card(exp_dir, is_active=False):
    """打印单个实验的监控卡片 (ASCII-only)"""
    csv_path = exp_dir / "metrics.csv"
    exp_name = exp_dir.name

    rows = read_metrics(csv_path)
    if not rows:
        return None

    total_epochs = get_total_epochs(exp_name)

    last = parse_row(rows[-1])
    if last is None:
        return None

    current_epoch = last["epoch"]
    progress_pct = current_epoch / total_epochs * 100

    # 历史最佳
    best_sing_ep, best_sing = get_best_metrics(rows, "sing_dice")
    best_edge_ep, best_edge = get_best_metrics(rows, "edge_dice")

    # 趋势
    sing_trend = get_trend(rows, "sing_dice")
    edge_trend = get_trend(rows, "edge_dice")

    # 状态
    status = "[RUNNING]" if is_active else "[ DONE  ]"
    bar = format_progress_bar(progress_pct)

    print()
    print(f"  *** {exp_name}  {status}")
    print(f"  |   Progress:  {bar}  Epoch {current_epoch}/{total_epochs} ({progress_pct:.0f}%)")

    # Losses
    print(f"  |   D Loss:    {last['d_loss']:8.3f}   G Loss: {last['g_loss']:8.3f}")
    print(f"  |   Recon:     {last['recon']:8.3f}   Adv:   {last['adv']:+8.3f}   R1: {last['r1']:.6f}")

    # Sing metrics
    sing_str = color_tag(f"{last['sing_dice']:.4f} {sing_trend}", sing_trend)
    print(f"  |   Sing AUC:  {last['sing_auc']:.4f}      Sing Dice: {sing_str}")
    print(f"  |   [Best Sing] E{best_sing_ep:03d} Dice={best_sing:.4f}")

    # Edge metrics
    edge_str = color_tag(f"{last['edge_dice']:.4f} {edge_trend}", edge_trend)
    print(f"  |   Edge AUC:  {last['edge_auc']:.4f}      Edge Dice: {edge_str}")
    print(f"  |   [Best Edge] E{best_edge_ep:03d} Dice={best_edge:.4f}")

    # M1/M2 detail
    print(f"  |   M1 AUC: {last['m1_auc']:.4f} Dice={last['m1_dice']:.4f}  |  M2 AUC: {last['m2_auc']:.4f} Dice={last['m2_dice']:.4f}")

    return last


def print_comparison_table(all_exps):
    """打印所有已完成实验的对比表 (ASCII)"""
    if len(all_exps) < 2:
        return

    print()
    print("  ======= Architecture Comparison =======")
    print(f"  {'Rank':<6s} {'Arch':>8s}  {'Sing Dice':>10s}  {'Edge Dice':>10s}  {'Sing AUC':>10s}")
    print(f"  {'-'*6} {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

    results = []
    for exp in all_exps:
        rows = read_metrics(exp / "metrics.csv")
        if rows:
            last = parse_row(rows[-1])
            _, best_sing = get_best_metrics(rows, "sing_dice")
            _, best_edge = get_best_metrics(rows, "edge_dice")
            if last:
                arch = exp.name.replace("v5_", "").replace("_hinge", "").upper()
                results.append((arch, best_sing, best_edge, last["sing_auc"]))

    results.sort(key=lambda x: x[1], reverse=True)
    for i, (arch, sing, edge, sauc) in enumerate(results):
        rank = f"#{i+1}"
        if i == 0:
            rank = f"#{i+1} *"
        print(f"  {rank:<6s} {arch:>8s}  {sing:10.4f}  {edge:10.4f}  {sauc:10.4f}")

    if results:
        print(f"\n  * Best: {results[0][0]} (Sing Dice={results[0][1]:.4f})")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compose-GNN Training Monitor")
    parser.add_argument("--interval", type=int, default=10,
                        help="Refresh interval in seconds (default: 10)")
    parser.add_argument("--all", action="store_true",
                        help="Monitor all experiment dirs (including physics)")
    parser.add_argument("--once", action="store_true",
                        help="Print once and exit")
    args = parser.parse_args()

    watch_dirs = [COMPARE_DIR]
    if args.all:
        watch_dirs.append(PHYSICS_DIR)

    try:
        while True:
            clear_screen()
            print_header()
            print(f"  Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Refresh: every {args.interval}s | Ctrl+C to stop")
            print(f"  Watching: {', '.join(str(d) for d in watch_dirs)}")
            print_separator("-")

            all_completed = []
            active_count = 0

            for watch_dir in watch_dirs:
                exps = find_experiment_dirs(watch_dir)
                if not exps:
                    print(f"\n  [WAIT] No experiments found in {watch_dir}")
                    continue

                for exp in exps:
                    rows = read_metrics(exp / "metrics.csv")
                    if not rows:
                        continue

                    last = parse_row(rows[-1])
                    if last is None:
                        continue

                    total = get_total_epochs(exp.name)
                    is_active = last["epoch"] < total

                    print_experiment_card(exp, is_active=is_active)

                    if is_active:
                        active_count += 1
                    else:
                        all_completed.append(exp)

            # 对比表
            if all_completed:
                print_comparison_table(all_completed)

            # 全部完成
            if active_count == 0 and all_completed:
                print()
                print("=" * 62)
                print("  *** ALL EXPERIMENTS COMPLETED! ***")
                print("=" * 62)

            if args.once:
                break

            print(f"\n  Next update in {args.interval}s...")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n\n  Monitor stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
