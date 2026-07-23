# train_gan_v3.py — Stabilized GAN Training (v2 问题修复版)
# =============================================================================
# 基于 v2 实验的分析，做出的关键改进:
#   1. R1 梯度惩罚 (‖∇D(real)‖²) — 替代 WGAN-GP，零中心，仅惩罚真实数据
#   2. Label Smoothing (real=0.9, fake=0.1) — 防止 D 过拟合
#   3. 不平衡学习率 (lr_g=2e-4, lr_d=5e-5) — D 慢 4 倍，给 G 更多机会
#   4. D 容量降低 (hidden_dim=64) — 191k→? 参数，平衡 G/D 能力
#   5. D 输入噪声 (σ=0.05) — 防止 D 靠微小差异完美区分真假
#   6. 降低对抗损失权重 (λ_adv=0.5) — 减少 D 对 G 的压制
#   7. G 梯度裁剪 — 防止 G 梯度爆炸
#   8. 仅跑 Hinge + LSGAN — 跳过不稳定的 WGAN-GP
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import os
import sys
import random
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gan_models import GeneratorGNN, DiscriminatorGNN, init_weights


# ============================= 随机种子 =============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================= R1 梯度惩罚 =============================
def r1_penalty(D, data, real_node, real_edge, lambda_r1=1.0):
    """
    R1 正则化: λ/2 * ‖∇D(real)‖²
    - 仅对真实数据计算梯度惩罚
    - 零中心 (不强制梯度范数=1，而是惩罚任何非零梯度)
    - 比 WGAN-GP 更稳定，适合各类 GAN 损失（Hinge, LSGAN, BCE）
    """
    real_node_grad = real_node.detach().requires_grad_(True)
    real_edge_grad = real_edge.detach().requires_grad_(True)

    score_real = D(data, real_node_grad, real_edge_grad)

    grads = torch.autograd.grad(
        outputs=score_real,
        inputs=[real_node_grad, real_edge_grad],
        grad_outputs=torch.ones_like(score_real),
        create_graph=True,
        retain_graph=True,
    )

    grad_norm = 0.0
    for g in grads:
        if g is not None:
            grad_norm += g.pow(2).sum()

    return (lambda_r1 / 2) * grad_norm / real_node_grad.size(0)


# ============================= D 输入噪声 =============================
def add_discriminator_noise(node_labels, edge_labels, noise_std=0.05, training=True):
    """向 D 输入注入小量高斯噪声，防止 D 靠微小差异完美区分真假"""
    if not training or noise_std <= 0:
        return node_labels, edge_labels
    node_noisy = node_labels + torch.randn_like(node_labels) * noise_std
    edge_noisy = edge_labels + torch.randn_like(edge_labels) * noise_std
    return node_noisy, edge_noisy


# ============================= 验证指标 =============================
@torch.no_grad()
def compute_metrics(node_pred, edge_pred, node_target, edge_target,
                    sing_threshold=0.5, edge_threshold=0.5):
    from sklearn.metrics import roc_auc_score

    node_pred_np = node_pred.cpu().numpy().ravel()
    node_target_np = node_target.cpu().numpy().ravel()
    edge_pred_np = edge_pred.cpu().numpy()
    edge_target_np = edge_target.cpu().numpy()

    def safe_auc(y_true, y_pred):
        if y_true.sum() == 0 or (y_true == 1).all():
            return 0.5
        return roc_auc_score(y_true, y_pred)

    sing_th = np.percentile(node_target_np, 99) if node_target_np.max() > 0 else 0.5
    sing_bin = (node_target_np > sing_th).astype(int)

    results = {
        'sing_auc': safe_auc(sing_bin, node_pred_np),
        'sing_dice': 2 * ((node_pred_np > sing_threshold) * sing_bin).sum() /
                      ((node_pred_np > sing_threshold).sum() + sing_bin.sum() + 1e-8),
    }

    for label, pred, true in [('m1', edge_pred_np[:, 0], edge_target_np[:, 0]),
                               ('m2', edge_pred_np[:, 1], edge_target_np[:, 1])]:
        th = np.percentile(true, 84) if true.max() > 0 else 0.5
        true_bin = (true > th).astype(int)
        pred_bin = (pred > edge_threshold).astype(int)
        results[f'{label}_auc'] = safe_auc(true_bin, pred)
        results[f'{label}_dice'] = (2 * (pred_bin * true_bin).sum() /
                                     (pred_bin.sum() + true_bin.sum() + 1e-8))

    edge_all_true = np.concatenate([
        (edge_target_np[:, 0] > np.percentile(edge_target_np[:, 0], 84)).astype(int),
        (edge_target_np[:, 1] > np.percentile(edge_target_np[:, 1], 84)).astype(int),
    ])
    edge_all_pred = np.concatenate([edge_pred_np[:, 0], edge_pred_np[:, 1]])
    results['edge_auc'] = safe_auc(edge_all_true, edge_all_pred)
    results['edge_dice'] = (2 * ((edge_all_pred > edge_threshold) * edge_all_true).sum() /
                             ((edge_all_pred > edge_threshold).sum() + edge_all_true.sum() + 1e-8))
    return results


# ============================= 训练主函数 =============================
def train_gan_v3(data_path,
                 epochs=300,
                 lr_g=2e-4,              # ★ G 学习率更高
                 lr_d=5e-5,              # ★ D 学习率更低 (G/D = 4:1)
                 batch_size=2,
                 hidden_dim_g=128,        # G hidden dim
                 hidden_dim_d=64,         # ★ D hidden dim 减半
                 gnn_type='gat',
                 gan_mode='hinge',
                 lambda_recon=10.0,
                 lambda_adv=0.5,          # ★ 对抗损失权重降低
                 lambda_r1=1.0,           # ★ R1 梯度惩罚权重
                 label_smooth=0.9,        # ★ 真实标签平滑值
                 d_noise=0.05,            # ★ D 输入噪声标准差
                 d_updates=1,             # ★ 每轮 D 更新 1 次 (而非 2 次)
                 g_clip_norm=5.0,         # ★ G 梯度裁剪
                 warmup_epochs=5,
                 device='cuda',
                 save_root='./trained_model',
                 seed=42):
    """
    GAN v3 — 稳定化版本。

    相比 v2 核心变化:
        lr_g=2e-4, lr_d=5e-5    (G比D快4倍)
        hidden_dim_d=64          (D参数量减半，平衡G/D能力)
        lambda_adv=0.5           (对抗信号减半)
        lambda_r1=1.0            (R1正则化，防止D过拟合)
        label_smooth=0.9         (标签平滑，防止D过度自信)
        d_noise=0.05             (D输入噪声)
        d_updates=1              (每轮D仅更新1次)
        g_clip_norm=5.0          (G梯度裁剪)
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ---- 保存路径 ----
    model_label = f'GANv3_{gnn_type}_{gan_mode}_bs{batch_size}_Gh{hidden_dim_g}_Dh{hidden_dim_d}'
    cur_time = datetime.now().strftime("%Y_%m_%d_%H%M")
    save_dir = os.path.join(save_root, cur_time, model_label)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Experiment: {model_label}")
    print(f"Save dir:  {save_dir}")

    # ---- 加载数据 ----
    print(f"\nLoading: {data_path}")
    dataset = torch.load(data_path, weights_only=False)
    n_total = len(dataset)
    indices = list(range(n_total))
    random.shuffle(indices)
    n_train = int(n_total * 0.9)
    train_dataset = [dataset[i] for i in indices[:n_train]]
    val_dataset = [dataset[i] for i in indices[n_train:]]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  Train: {n_train} → {len(train_loader)} batches (bs={batch_size})")
    print(f"  Val:   {len(val_dataset)} → {len(val_loader)} batches")

    # ---- 初始化模型 ----
    generator = GeneratorGNN(
        input_dim=12, hidden_dim=hidden_dim_g, hidden_dim2=64,
        gnn_type=gnn_type, dropout=0.1
    ).to(device)
    generator.apply(init_weights)

    discriminator = DiscriminatorGNN(
        input_dim=12, node_label_dim=1, hidden_dim=hidden_dim_d,
        gnn_type=gnn_type, dropout=0.3   # ★ D dropout 提高
    ).to(device)
    discriminator.apply(init_weights)

    n_params_g = sum(p.numel() for p in generator.parameters())
    n_params_d = sum(p.numel() for p in discriminator.parameters())
    print(f"\nGenerator:     {n_params_g:,} params")
    print(f"Discriminator: {n_params_d:,} params (G/D ratio = {n_params_g/n_params_d:.2f})")

    # ---- 优化器 (不平衡 LR) ----
    opt_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs - warmup_epochs,
                                                        eta_min=lr_g * 0.01)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs - warmup_epochs,
                                                        eta_min=lr_d * 0.01)

    # ---- 损失 ----
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    # ---- 日志 ----
    log_path = os.path.join(save_dir, 'metrics.csv')
    header = ("epoch,d_loss,g_loss,g_recon,g_adv,gp_r1,"
              "sing_auc,sing_dice,m1_auc,m1_dice,m2_auc,m2_dice,edge_auc,edge_dice,"
              "lr_g,lr_d\n")
    with open(log_path, 'w') as f:
        f.write(header)

    hp_path = os.path.join(save_dir, 'hyperparams.txt')
    with open(hp_path, 'w') as f:
        for k, v in dict(
            gan_mode=gan_mode, batch_size=batch_size, epochs=epochs,
            lr_g=lr_g, lr_d=lr_d, hidden_dim_g=hidden_dim_g, hidden_dim_d=hidden_dim_d,
            gnn_type=gnn_type, lambda_recon=lambda_recon, lambda_adv=lambda_adv,
            lambda_r1=lambda_r1, label_smooth=label_smooth, d_noise=d_noise,
            d_updates=d_updates, g_clip_norm=g_clip_norm, warmup_epochs=warmup_epochs,
            seed=seed, generator_params=n_params_g, discriminator_params=n_params_d,
            g_d_ratio=n_params_g / n_params_d if n_params_d > 0 else float('inf'),
        ).items():
            f.write(f"{k}: {v}\n")

    best_edge_dice = 0.0
    best_sing_dice = 0.0

    # ---- 训练循环 ----
    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        epoch_losses = {'d': 0.0, 'g': 0.0, 'g_recon': 0.0, 'g_adv': 0.0, 'gp_r1': 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)

            real_node = batch.y_node
            real_edge = batch.y_edge

            # ============ (1) 训练 Discriminator ============
            for _ in range(d_updates):
                opt_d.zero_grad()

                with torch.no_grad():
                    fake_node, fake_edge = generator(batch)

                # D 输入加噪声（防止 D 过拟合到微小差异）
                real_node_noisy, real_edge_noisy = add_discriminator_noise(
                    real_node, real_edge, noise_std=d_noise, training=True)
                fake_node_noisy, fake_edge_noisy = add_discriminator_noise(
                    fake_node, fake_edge, noise_std=d_noise, training=True)

                real_score = discriminator(batch, real_node_noisy, real_edge_noisy)
                fake_score = discriminator(batch, fake_node_noisy, fake_edge_noisy)

                # Label smoothing: real→0.9, fake→0.1
                real_target = torch.full_like(real_score, label_smooth)
                fake_target = torch.full_like(fake_score, 1.0 - label_smooth)

                if gan_mode == 'hinge':
                    d_real_loss = F.relu(real_target - real_score).mean()
                    d_fake_loss = F.relu(fake_score + real_target).mean()
                    d_loss = d_real_loss + d_fake_loss

                elif gan_mode == 'lsgan':
                    d_real_loss = mse_loss(real_score, real_target)
                    d_fake_loss = mse_loss(fake_score, fake_target)
                    d_loss = 0.5 * (d_real_loss + d_fake_loss)

                else:
                    raise ValueError(f"Unknown gan_mode: {gan_mode}")

                # R1 梯度惩罚
                if lambda_r1 > 0:
                    r1 = r1_penalty(discriminator, batch, real_node_noisy, real_edge_noisy,
                                    lambda_r1=lambda_r1)
                    d_loss = d_loss + r1
                    epoch_losses['gp_r1'] += r1.item()

                d_loss.backward()
                opt_d.step()

            # ============ (2) 训练 Generator ============
            opt_g.zero_grad()

            fake_node, fake_edge = generator(batch)
            fake_node_noisy, fake_edge_noisy = add_discriminator_noise(
                fake_node, fake_edge, noise_std=d_noise, training=True)
            fake_score = discriminator(batch, fake_node_noisy, fake_edge_noisy)

            # 重建损失 (L1)
            g_recon = l1_loss(fake_node, real_node) + l1_loss(fake_edge, real_edge)

            # 对抗损失
            if gan_mode == 'hinge':
                g_adv = -fake_score.mean()
            elif gan_mode == 'lsgan':
                g_adv = 0.5 * mse_loss(fake_score, torch.full_like(fake_score, label_smooth))

            g_loss = lambda_recon * g_recon + lambda_adv * g_adv
            g_loss.backward()

            # ★ G 梯度裁剪
            torch.nn.utils.clip_grad_norm_(generator.parameters(), g_clip_norm)

            opt_g.step()

            # 统计
            epoch_losses['d'] += d_loss.item()
            epoch_losses['g'] += g_loss.item()
            epoch_losses['g_recon'] += g_recon.item()
            epoch_losses['g_adv'] += g_adv.item()

            pbar.set_postfix(
                D=f'{d_loss.item():.3f}',
                G=f'{g_loss.item():.3f}',
                recon=f'{g_recon.item():.3f}',
                adv=f'{g_adv.item():.3f}',
                R1=f'{r1.item() if lambda_r1 > 0 else 0:.3f}',
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

        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

        # ---- 日志 ----
        n_b = len(train_loader)
        current_lr_g = opt_g.param_groups[0]['lr']
        current_lr_d = opt_d.param_groups[0]['lr']

        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{epoch_losses['d']/n_b:.6f},{epoch_losses['g']/n_b:.6f},"
                    f"{epoch_losses['g_recon']/n_b:.6f},{epoch_losses['g_adv']/n_b:.6f},"
                    f"{epoch_losses['gp_r1']/n_b:.6f},"
                    f"{avg_metrics['sing_auc']:.4f},{avg_metrics['sing_dice']:.4f},"
                    f"{avg_metrics['m1_auc']:.4f},{avg_metrics['m1_dice']:.4f},"
                    f"{avg_metrics['m2_auc']:.4f},{avg_metrics['m2_dice']:.4f},"
                    f"{avg_metrics['edge_auc']:.4f},{avg_metrics['edge_dice']:.4f},"
                    f"{current_lr_g:.2e},{current_lr_d:.2e}\n")

        # ---- 保存最佳 ----
        if avg_metrics['edge_dice'] > best_edge_dice:
            best_edge_dice = avg_metrics['edge_dice']
            torch.save({
                'epoch': epoch + 1, 'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'best_edge_dice': best_edge_dice,
            }, os.path.join(save_dir, 'best_model_edge.pth'))

        if avg_metrics['sing_dice'] > best_sing_dice:
            best_sing_dice = avg_metrics['sing_dice']
            torch.save({
                'epoch': epoch + 1, 'generator': generator.state_dict(),
                'best_sing_dice': best_sing_dice,
            }, os.path.join(save_dir, 'best_model_sing.pth'))

        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch + 1, 'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # ---- 打印 ----
        print(f"Epoch {epoch+1:3d}: "
              f"D={epoch_losses['d']/n_b:.3f} G={epoch_losses['g']/n_b:.3f} "
              f"recon={epoch_losses['g_recon']/n_b:.3f} adv={epoch_losses['g_adv']/n_b:.3f} "
              f"R1={epoch_losses['gp_r1']/n_b:.3f} | "
              f"Sing AUC={avg_metrics['sing_auc']:.3f} Dice={avg_metrics['sing_dice']:.3f} | "
              f"M1 AUC={avg_metrics['m1_auc']:.3f} Dice={avg_metrics['m1_dice']:.3f} | "
              f"M2 AUC={avg_metrics['m2_auc']:.3f} Dice={avg_metrics['m2_dice']:.3f} | "
              f"Edge AUC={avg_metrics['edge_auc']:.3f} Dice={avg_metrics['edge_dice']:.3f}")

        # ---- LR ----
        if epoch >= warmup_epochs:
            scheduler_g.step()
            scheduler_d.step()

    # ---- 结束 ----
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"  Best Edge Dice:  {best_edge_dice:.4f}")
    print(f"  Best Sing Dice:  {best_sing_dice:.4f}")
    print(f"  Saved to:        {save_dir}")

    torch.save({
        'epoch': epochs, 'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'best_edge_dice': best_edge_dice, 'best_sing_dice': best_sing_dice,
    }, os.path.join(save_dir, 'final_model.pth'))

    return generator, discriminator, save_dir


# ============================= 入口 =============================
if __name__ == '__main__':
    DATA_PATH = "datasets/03_graph/merged_25cases_continuous_augmented_x7.pt"

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found: {DATA_PATH}")
        sys.exit(1)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- 实验配置 ----
    configs = [
        # 实验 3: Hinge GAN 稳定版
        {
            'gan_mode': 'hinge',
            'lambda_recon': 10.0,
            'lambda_adv': 0.5,
            'lambda_r1': 1.0,
            'd_noise': 0.05,
        },
        # 实验 4: LSGAN 稳定版
        {
            'gan_mode': 'lsgan',
            'lambda_recon': 10.0,
            'lambda_adv': 0.5,
            'lambda_r1': 1.0,
            'd_noise': 0.05,
        },
    ]

    for i, cfg in enumerate(configs):
        print("\n" + "=" * 70)
        print(f"Experiment {i+1}/{len(configs)}: {cfg['gan_mode'].upper()} (stabilized)")
        print("=" * 70)

        train_gan_v3(
            data_path=DATA_PATH,
            epochs=300,
            lr_g=2e-4,
            lr_d=5e-5,
            batch_size=2,
            hidden_dim_g=128,
            hidden_dim_d=64,
            gnn_type='gat',
            gan_mode=cfg['gan_mode'],
            lambda_recon=cfg['lambda_recon'],
            lambda_adv=cfg['lambda_adv'],
            lambda_r1=cfg['lambda_r1'],
            label_smooth=0.9,
            d_noise=cfg['d_noise'],
            d_updates=1,
            g_clip_norm=5.0,
            warmup_epochs=5,
            device=DEVICE,
            save_root='./trained_model',
            seed=42,
        )
