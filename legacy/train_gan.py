# train_gan.py
# =============================================================================
# 条件 GAN 训练脚本：图生成器 + 图判别器
#
# Generator G: 应力场 → 概率分布 (node_prob, edge_prob)
# Discriminator D: (应力场, 标签) → real/fake
#
# 损失:
#   L_D = BCE(D(real), 1) + BCE(D(fake), 0) + λ_gp * GP
#   L_G = λ_recon * L1(pred, target) + λ_adv * BCE(D(fake), 1)
#
# 训练策略:
#   1. 小数据集 (25 cases) + 在线增强 (旋转/反射/缩放)
#   2. 梯度惩罚 (WGAN-GP) 稳定 D 训练
#   3. 每轮 D 更新 2 次，G 更新 1 次
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
import os
import random
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

from models.gan_models import GeneratorGNN, DiscriminatorGNN, init_weights
from utils.augment import StressFieldAugmentor, AUGMENT_CONFIG


# ============================= 工具函数 =============================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gradient_penalty(D, data, real_node, real_edge, fake_node, fake_edge, lambda_gp=10.0):
    """
    WGAN-GP: 在真假之间插值，计算梯度惩罚。
    适用于图级别判别器输出。
    """
    device = data.x.device
    batch_size = real_node.size(0)

    # 随机插值系数
    eps = torch.rand(batch_size, 1, device=device)
    eps_edge = torch.rand(fake_edge.size(0), 1, device=device)

    # 对节点标签插值
    node_interp = eps * real_node + (1 - eps) * fake_node
    node_interp = node_interp.detach().requires_grad_(True)

    # 对边标签插值
    edge_interp = eps_edge * real_edge + (1 - eps_edge) * fake_edge
    edge_interp = edge_interp.detach().requires_grad_(True)

    # 判别器对插值样本的输出
    score_interp = D(data, node_interp, edge_interp)

    # 计算梯度
    grads = torch.autograd.grad(
        outputs=score_interp,
        inputs=[node_interp, edge_interp],
        grad_outputs=torch.ones_like(score_interp),
        create_graph=True,
        retain_graph=True,
    )

    # 梯度范数
    grad_norm = 0.0
    for g in grads:
        if g is not None:
            grad_norm += (g.norm(2, dim=1) ** 2).mean()

    gp = lambda_gp * ((grad_norm.sqrt() - 1.0) ** 2).mean()
    return gp


# ============================= 指标计算 =============================

@torch.no_grad()
def compute_metrics(node_pred, edge_pred, node_target, edge_target,
                    sing_threshold=0.5, edge_threshold=0.5):
    """
    计算连续预测相对于二值化标签的 AUC / Dice。
    注意：node_target 是连续值，这里用阈值做近似评估。
    """
    from sklearn.metrics import roc_auc_score

    node_pred_np = node_pred.cpu().numpy()
    node_target_np = node_target.cpu().numpy()
    edge_pred_np = edge_pred.cpu().numpy()
    edge_target_np = edge_target.cpu().numpy()

    # ---- 奇异点：node_pred [N,1], node_target [N,1] ----
    sing_pred = node_pred_np.ravel()
    sing_target = node_target_np.ravel()
    sing_th = np.percentile(sing_target, 99) if sing_target.max() > 0 else 0.5
    sing_target_bin = (sing_target > sing_th).astype(int)

    def safe_auc(y_true, y_pred):
        if y_true.sum() == 0 or (y_true == 1).all():
            return 0.5
        return roc_auc_score(y_true, y_pred)

    sing_auc = safe_auc(sing_target_bin, sing_pred)
    sing_dice = 2 * ((sing_pred > sing_threshold) * sing_target_bin).sum() / \
                ((sing_pred > sing_threshold).sum() + sing_target_bin.sum() + 1e-8)

    # ---- PSL 边 (m1 + m2) ----
    edge_pred_m1 = edge_pred_np[:, 0]   # m1-PSL
    edge_pred_m2 = edge_pred_np[:, 1]   # m2-PSL
    edge_target_m1 = edge_target_np[:, 0]
    edge_target_m2 = edge_target_np[:, 1]

    # m1
    th1 = np.percentile(edge_target_m1, 84) if edge_target_m1.max() > 0 else 0.5
    et1_bin = (edge_target_m1 > th1).astype(int)
    m1_auc = safe_auc(et1_bin, edge_pred_m1)
    m1_dice = 2 * ((edge_pred_m1 > 0.5) * et1_bin).sum() / \
              ((edge_pred_m1 > 0.5).sum() + et1_bin.sum() + 1e-8)

    # m2
    th2 = np.percentile(edge_target_m2, 84) if edge_target_m2.max() > 0 else 0.5
    et2_bin = (edge_target_m2 > th2).astype(int)
    m2_auc = safe_auc(et2_bin, edge_pred_m2)
    m2_dice = 2 * ((edge_pred_m2 > 0.5) * et2_bin).sum() / \
              ((edge_pred_m2 > 0.5).sum() + et2_bin.sum() + 1e-8)

    # 综合
    edge_all_target = np.concatenate([et1_bin, et2_bin])
    edge_all_pred = np.concatenate([edge_pred_m1, edge_pred_m2])

    return {
        'sing_auc': sing_auc,
        'sing_dice': sing_dice,
        'm1_auc': m1_auc, 'm1_dice': m1_dice,
        'm2_auc': m2_auc, 'm2_dice': m2_dice,
        'edge_auc': safe_auc(edge_all_target, edge_all_pred),
        'edge_dice': 2 * ((edge_all_pred > 0.5) * edge_all_target).sum() / \
                      ((edge_all_pred > 0.5).sum() + edge_all_target.sum() + 1e-8),
    }


# ============================= 训练函数 =============================

def train_gan(data_path,
              epochs=300,
              lr_g=1e-4,
              lr_d=1e-4,
              batch_size=1,   # 单图 batch
              hidden_dim=128,
              hidden_dim2=64,
              gnn_type='gat',
              lambda_recon=10.0,
              lambda_adv=1.0,
              lambda_gp=0.0,   # BCE-GAN 不需要 GP
              d_updates=2,
              device='cuda',
              save_root='./trained_model',
              seed=42):
    """
    GAN 训练主循环。

    Args:
        lambda_recon: 重建损失权重 (L1)
        lambda_adv:    对抗损失权重
        lambda_gp:     梯度惩罚权重
        d_updates:     每轮 G 更新前 D 的更新次数
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ---- 创建保存路径 ----
    model_label = f'GAN_{gnn_type}_h{hidden_dim}_lr{lr_g}_rec{lambda_recon}'
    cur_time = datetime.now().strftime("%Y_%m_%d_%H%M")
    save_dir = os.path.join(save_root, cur_time, model_label)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Experiment: {model_label}")
    print(f"Save dir: {save_dir}")

    # ---- 加载数据 ----
    print(f"\nLoading: {data_path}")
    dataset = torch.load(data_path, weights_only=False)
    n_total = len(dataset)
    print(f"  {n_total} graphs loaded")

    # 划分训练/验证 (180/20)
    indices = list(range(n_total))
    random.shuffle(indices)
    n_train = 180
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    print(f"  Train: {n_train}, Val: {len(val_indices)}")

    # ---- 初始化模型 ----
    generator = GeneratorGNN(
        input_dim=12, hidden_dim=hidden_dim, hidden_dim2=hidden_dim2,
        gnn_type=gnn_type, dropout=0.1
    ).to(device)
    generator.apply(init_weights)

    discriminator = DiscriminatorGNN(
        input_dim=12, node_label_dim=1, hidden_dim=hidden_dim,
        gnn_type=gnn_type, dropout=0.2
    ).to(device)
    discriminator.apply(init_weights)

    print(f"\nGenerator: {sum(p.numel() for p in generator.parameters()):,} params")
    print(f"Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} params")

    # ---- 优化器 ----
    opt_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # 学习率调度
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=lr_g * 0.1)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs, eta_min=lr_d * 0.1)

    # ---- 损失 ----
    # Hinge Loss: D 期望 D(real)≥1, D(fake)≤-1; G 期望 D(fake) 尽可能大
    l1_loss = nn.L1Loss()

    # ---- 训练日志 ----
    log_path = os.path.join(save_dir, 'metrics.csv')
    with open(log_path, 'w') as f:
        f.write("epoch,d_loss,g_loss,g_recon,g_adv,sing_auc,sing_dice,m1_auc,m1_dice,m2_auc,m2_dice\n")

    best_edge_dice = 0.0
    best_sing_dice = 0.0

    # ---- DataLoader（静态增强数据集，无在线增强） ----
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- 训练循环 ----
    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_g_recon = 0.0
        epoch_g_adv = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            batch = batch.to(device)

            real_node = batch.y_node   # [N, 2] 连续标签
            real_edge = batch.y_edge   # [2E, 1] 连续标签

            # ============ (1) 训练判别器 D ============
            for _ in range(d_updates):
                opt_d.zero_grad()

                # 生成假样本
                with torch.no_grad():
                    fake_node, fake_edge = generator(batch)

                # D Hinge Loss: max(0, 1-D(real)) + max(0, 1+D(fake))
                real_score = discriminator(batch, real_node, real_edge)
                fake_score = discriminator(batch, fake_node, fake_edge)
                d_real_loss = F.relu(1.0 - real_score).mean()
                d_fake_loss = F.relu(1.0 + fake_score).mean()

                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                opt_d.step()

            # ============ (2) 训练生成器 G ============
            opt_g.zero_grad()

            fake_node, fake_edge = generator(batch)

            # 重建损失（L1）
            g_recon = l1_loss(fake_node, real_node) + l1_loss(fake_edge, real_edge)

            # 对抗损失 (Hinge: G 希望 D(fake) 尽可能大)
            fake_score = discriminator(batch, fake_node, fake_edge)
            g_adv = -fake_score.mean()

            g_loss = lambda_recon * g_recon + lambda_adv * g_adv
            g_loss.backward()
            opt_g.step()

            # 统计
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_g_recon += g_recon.item()
            epoch_g_adv += g_adv.item()

            pbar.set_postfix(
                D=f'{d_loss.item():.3f}',
                G=f'{g_loss.item():.3f}',
                recon=f'{g_recon.item():.3f}',
                adv=f'{g_adv.item():.3f}',
            )

        # ---- 验证 ----
        generator.eval()
        all_metrics = []

        for batch in val_loader:
            batch = batch.to(device)
            with torch.no_grad():
                node_pred, edge_pred = generator(batch)

            m = compute_metrics(node_pred, edge_pred, batch.y_node, batch.y_edge)
            all_metrics.append(m)

        # 平均验证指标
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

        # ---- 日志 ----
        n_b = len(train_loader)
        avg_d = epoch_d_loss / n_b
        avg_g = epoch_g_loss / n_b
        avg_recon = epoch_g_recon / n_b
        avg_adv = epoch_g_adv / n_b

        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{avg_d:.6f},{avg_g:.6f},{avg_recon:.6f},{avg_adv:.6f},"
                    f"{avg_metrics['sing_auc']:.4f},{avg_metrics['sing_dice']:.4f},"
                    f"{avg_metrics['m1_auc']:.4f},{avg_metrics['m1_dice']:.4f},"
                    f"{avg_metrics['m2_auc']:.4f},{avg_metrics['m2_dice']:.4f}\n")

        # ---- 保存最佳 ----
        if avg_metrics['edge_dice'] > best_edge_dice:
            best_edge_dice = avg_metrics['edge_dice']
            torch.save(generator.state_dict(), os.path.join(save_dir, 'best_generator_edge.pth'))

        if avg_metrics['sing_dice'] > best_sing_dice:
            best_sing_dice = avg_metrics['sing_dice']
            torch.save(generator.state_dict(), os.path.join(save_dir, 'best_generator_sing.pth'))

        # ---- 打印 ----
        print(f"Epoch {epoch+1}: D={avg_d:.3f} G={avg_g:.3f} recon={avg_recon:.3f} adv={avg_adv:.3f} | "
              f"Sing AUC={avg_metrics['sing_auc']:.3f} Dice={avg_metrics['sing_dice']:.3f} | "
              f"M1 AUC={avg_metrics['m1_auc']:.3f} Dice={avg_metrics['m1_dice']:.3f} | "
              f"M2 AUC={avg_metrics['m2_auc']:.3f} Dice={avg_metrics['m2_dice']:.3f}")

        # ---- 学习率衰减 ----
        scheduler_g.step()
        scheduler_d.step()

    print(f"\nTraining done.")
    print(f"Best Edge Dice: {best_edge_dice:.4f}")
    print(f"Best Sing Dice: {best_sing_dice:.4f}")
    print(f"Model saved to: {save_dir}")

    return generator, discriminator


# ============================= 入口 =============================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    DATA_PATH = "datasets/03_graph/merged_25cases_continuous_augmented_x7.pt"

    # 先检查数据集是否存在
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Continuous dataset not found: {DATA_PATH}")
        print(f"Run: python build_graph_dataset_continuous.py")
        sys.exit(1)

    train_gan(
        data_path=DATA_PATH,
        epochs=300,
        lr_g=1e-4,
        lr_d=1e-4,
        batch_size=1,   # 单图 batch
        hidden_dim=128,
        hidden_dim2=64,
        gnn_type='gat',
        lambda_recon=10.0,
        lambda_adv=1.0,
        lambda_gp=10.0,
        d_updates=2,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_root='./trained_model',
        seed=42,
    )
