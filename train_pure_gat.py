# train_pure_gat.py — 纯 GAT 基线实验（无 GAN）
# =============================================================================
# 与 v5 Hinge GAN 使用完全相同的 GeneratorGNN 架构和 L1 损失，
# 但不使用判别器和对抗训练。用于量化 GAN 组件对性能的实际贡献。
#
# 对比对象：train_gan_v5_compare.py (GAT + Hinge GAN)
#   - 相同：GeneratorGNN (87k params), L1 loss, Adam lr=2e-4, bs=2, 300 epochs
#   - 差异：无 Discriminator, 无对抗损失, 无 R1 penalty, 无 label smoothing
# =============================================================================

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np, os, sys, random
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gan_models import GeneratorGNN, init_weights


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_metrics(node_pred, edge_pred, node_target, edge_target):
    """与 v5_compare 完全相同的指标计算"""
    from sklearn.metrics import roc_auc_score

    np_, nt = node_pred.cpu().numpy().ravel(), node_target.cpu().numpy().ravel()
    ep, et = edge_pred.cpu().numpy(), edge_target.cpu().numpy()

    def sauc(yt, yp):
        if yt.sum() == 0 or (yt == 1).all():
            return 0.5
        return roc_auc_score(yt, yp)

    # 奇异点：阈值取 99 百分位
    st = np.percentile(nt, 99) if nt.max() > 0 else 0.5
    sb = (nt > st).astype(int)
    r = {
        'sing_auc': sauc(sb, np_),
        'sing_dice': 2 * ((np_ > 0.5) * sb).sum() / ((np_ > 0.5).sum() + sb.sum() + 1e-8),
    }

    # M1 / M2 边
    for lb, p, t in [('m1', ep[:, 0], et[:, 0]), ('m2', ep[:, 1], et[:, 1])]:
        th = np.percentile(t, 84) if t.max() > 0 else 0.5
        tb = (t > th).astype(int)
        pb = (p > 0.5).astype(int)
        r[f'{lb}_auc'] = sauc(tb, p)
        r[f'{lb}_dice'] = 2 * (pb * tb).sum() / (pb.sum() + tb.sum() + 1e-8)

    # 边聚合
    ea_t = np.concatenate([
        (et[:, 0] > np.percentile(et[:, 0], 84)).astype(int),
        (et[:, 1] > np.percentile(et[:, 1], 84)).astype(int),
    ])
    ea_p = np.concatenate([ep[:, 0], ep[:, 1]])
    r['edge_auc'] = sauc(ea_t, ea_p)
    r['edge_dice'] = 2 * ((ea_p > 0.5) * ea_t).sum() / ((ea_p > 0.5).sum() + ea_t.sum() + 1e-8)

    return r


def train_pure_gat(
    data_path,
    gnn_type='gat',
    epochs=300,
    lr=2e-4,
    batch_size=8,
    hidden_dim=128,
    hidden_dim2=64,
    dropout=0.1,
    weight_decay=0,
    grad_clip_norm=5.0,
    device='cuda',
    save_root='./trained_model/pure_gat',
    seed=42,
):
    """
    纯 GAT 训练（无 GAN）。

    Args:
        data_path: 数据集 .pt 文件路径
        gnn_type: GNN 类型 ('gat', 'gcn', 'sage')
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批次大小
        hidden_dim: GNN 第一层隐藏维度
        hidden_dim2: GNN 第二层隐藏维度
        dropout: Dropout 概率
        weight_decay: Adam 权重衰减
        grad_clip_norm: 梯度裁剪阈值
        device: 'cuda' 或 'cpu'
        save_root: 模型保存根目录
        seed: 随机种子
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ---- 目录 ----
    label = f'pure_{gnn_type}_l1'
    cur_time = datetime.now().strftime("%Y_%m_%d_%H%M")
    save_dir = os.path.join(save_root, cur_time, label)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Experiment: {label}")
    print(f"Save dir:  {save_dir}")
    print(f"Device:    {device}")

    # ---- 数据 ----
    dataset = torch.load(data_path, weights_only=False)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n_train = int(len(dataset) * 0.9)

    train_loader = DataLoader(
        [dataset[i] for i in indices[:n_train]],
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        [dataset[i] for i in indices[n_train:]],
        batch_size=batch_size, shuffle=False,
    )
    print(f"Dataset:   {len(dataset)} graphs → {n_train} train / {len(dataset) - n_train} val")

    # ---- 模型 ----
    model = GeneratorGNN(
        input_dim=12, hidden_dim=hidden_dim, hidden_dim2=hidden_dim2,
        gnn_type=gnn_type, dropout=dropout,
    ).to(device)
    model.apply(init_weights)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model ({gnn_type}): {n_params:,} parameters")

    # ---- 优化器 & 调度器 ----
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999),
                           weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )

    # ---- CSV 日志 ----
    log_path = os.path.join(save_dir, 'metrics.csv')
    with open(log_path, 'w') as f:
        f.write("epoch,loss,"
                "sing_auc,sing_dice,m1_auc,m1_dice,m2_auc,m2_dice,edge_auc,edge_dice,"
                "lr\n")

    # ---- 训练 ----
    best_edge, best_sing = 0.0, 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{gnn_type}] E{epoch+1}/{epochs}")

        for batch in pbar:
            batch = batch.to(device)
            rn, re = batch.y_node, batch.y_edge

            optimizer.zero_grad()
            node_pred, edge_pred = model(batch)
            loss = loss_fn(node_pred, rn) + loss_fn(edge_pred, re)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)

        # ---- 验证 ----
        model.eval()
        all_m = []
        for batch in val_loader:
            batch = batch.to(device)
            with torch.no_grad():
                npred, epred = model(batch)
            all_m.append(compute_metrics(npred, epred, batch.y_node, batch.y_edge))
        avg = {k: np.mean([m[k] for m in all_m]) for k in all_m[0]}

        # ---- 保存 ----
        if avg['edge_dice'] > best_edge:
            best_edge = avg['edge_dice']
            torch.save({'G': model.state_dict()},
                       os.path.join(save_dir, 'best_edge.pth'))
        if avg['sing_dice'] > best_sing:
            best_sing = avg['sing_dice']
            torch.save({'G': model.state_dict()},
                       os.path.join(save_dir, 'best_sing.pth'))
        if (epoch + 1) % 100 == 0:
            torch.save({'G': model.state_dict()},
                       os.path.join(save_dir, f'ckpt_{epoch+1}.pth'))

        # ---- 日志 ----
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{avg_loss:.6f},"
                    f"{avg['sing_auc']:.4f},{avg['sing_dice']:.4f},"
                    f"{avg['m1_auc']:.4f},{avg['m1_dice']:.4f},"
                    f"{avg['m2_auc']:.4f},{avg['m2_dice']:.4f},"
                    f"{avg['edge_auc']:.4f},{avg['edge_dice']:.4f},"
                    f"{optimizer.param_groups[0]['lr']:.2e}\n")

        print(f"[{gnn_type}] E{epoch+1}: loss={avg_loss:.4f} | "
              f"Sing AUC={avg['sing_auc']:.3f} D={avg['sing_dice']:.3f} | "
              f"Edge AUC={avg['edge_auc']:.3f} D={avg['edge_dice']:.3f}")

        scheduler.step()

    print(f"\n[{gnn_type}] Done! Best Sing Dice={best_sing:.4f}  Edge Dice={best_edge:.4f}")
    return model, save_dir, best_sing, best_edge


# =============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Exp7: Pure GAT Baseline (No GAN)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gnn_type', type=str, default='gat')
    args = parser.parse_args()

    DATA = "datasets/03_graph/merged_25cases_continuous_augmented_x7.pt"

    if not os.path.exists(DATA):
        print(f"Missing dataset: {DATA}")
        sys.exit(1)

    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print(f"Pure GAT Baseline — L1 Reconstruction (No GAN) | BS={args.batch_size}")
    print("=" * 60)

    model, save_dir, best_s, best_e = train_pure_gat(
        DATA, gnn_type=args.gnn_type, epochs=args.epochs,
        batch_size=args.batch_size, device=DEV, seed=args.seed,
    )

    print(f"\nFinal Result: Sing Dice={best_s:.4f}  Edge Dice={best_e:.4f}")
    print(f"Models saved to: {save_dir}")
