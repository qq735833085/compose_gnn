# train_gan_v2.py — Enhanced GAN Training with batch_size=4
# =============================================================================
# 基于现有 GAN 架构的新实验，特点:
#   1. batch_size=4 — 多图批量训练，提升训练效率和梯度稳定性
#   2. 多种 GAN 损失可选: hinge / wgan-gp / lsgan
#   3. 梯度惩罚 (WGAN-GP) 稳定判别器
#   4. Generator 使用 GATv2 骨干网络 (更强大的注意力机制)
#   5. 完善的验证指标 (AUC, Dice, Precision, Recall)
#   6. 学习率 warmup + cosine 衰减
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
from loss import MultiTaskStressLoss


# ============================= 随机种子 =============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================= 梯度惩罚 (WGAN-GP) =============================
def gradient_penalty(D, data, real_node, real_edge, fake_node, fake_edge, lambda_gp=10.0):
    """
    WGAN-GP: 在真假样本间插值，计算判别器梯度范数惩罚。
    适配 batch 训练 — 每个图独立插值。
    """
    device = data.x.device
    B = data.num_graphs

    # 每个图独立采样 eps（node 维度需要用 batch 对齐）
    eps_node = torch.rand(real_node.size(0), 1, device=device)
    eps_edge = torch.rand(fake_edge.size(0), 1, device=device)

    # 插值
    node_interp = (eps_node * real_node + (1 - eps_node) * fake_node).detach().requires_grad_(True)
    edge_interp = (eps_edge * real_edge + (1 - eps_edge) * fake_edge).detach().requires_grad_(True)

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

    # 梯度范数惩罚
    grad_norm = 0.0
    for g in grads:
        if g is not None:
            grad_norm += (g.norm(2, dim=1) ** 2).mean()

    gp = lambda_gp * ((grad_norm.sqrt() - 1.0) ** 2).mean()
    return gp


# ============================= 验证指标 =============================
@torch.no_grad()
def compute_metrics(node_pred, edge_pred, node_target, edge_target,
                    sing_threshold=0.5, edge_threshold=0.5):
    """计算 AUC, Dice, Precision, Recall"""
    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    node_pred_np = node_pred.cpu().numpy().ravel()
    node_target_np = node_target.cpu().numpy().ravel()
    edge_pred_np = edge_pred.cpu().numpy()
    edge_target_np = edge_target.cpu().numpy()

    def safe_auc(y_true, y_pred):
        if y_true.sum() == 0 or (y_true == 1).all():
            return 0.5
        return roc_auc_score(y_true, y_pred)

    # ---- 奇异点 ----
    sing_th = np.percentile(node_target_np, 99) if node_target_np.max() > 0 else 0.5
    sing_bin = (node_target_np > sing_th).astype(int)

    sing_auc = safe_auc(sing_bin, node_pred_np)
    sing_pred_bin = (node_pred_np > sing_threshold).astype(int)
    sing_dice = 2 * (sing_pred_bin * sing_bin).sum() / (sing_pred_bin.sum() + sing_bin.sum() + 1e-8)

    # ---- PSL edges ----
    m1_pred = edge_pred_np[:, 0]; m2_pred = edge_pred_np[:, 1]
    m1_true = edge_target_np[:, 0]; m2_true = edge_target_np[:, 1]

    results = {'sing_auc': sing_auc, 'sing_dice': sing_dice}

    for label, pred, true in [('m1', m1_pred, m1_true), ('m2', m2_pred, m2_true)]:
        th = np.percentile(true, 84) if true.max() > 0 else 0.5
        true_bin = (true > th).astype(int)
        pred_bin = (pred > edge_threshold).astype(int)

        results[f'{label}_auc'] = safe_auc(true_bin, pred)
        results[f'{label}_dice'] = (2 * (pred_bin * true_bin).sum() /
                                     (pred_bin.sum() + true_bin.sum() + 1e-8))

    # 综合 edge 指标
    edge_all_true = np.concatenate([
        (m1_true > np.percentile(m1_true, 84)).astype(int),
        (m2_true > np.percentile(m2_true, 84)).astype(int),
    ])
    edge_all_pred = np.concatenate([m1_pred, m2_pred])
    results['edge_auc'] = safe_auc(edge_all_true, edge_all_pred)
    results['edge_dice'] = (2 * ((edge_all_pred > edge_threshold) * edge_all_true).sum() /
                             ((edge_all_pred > edge_threshold).sum() + edge_all_true.sum() + 1e-8))

    return results


# ============================= 训练主函数 =============================
def train_gan_v2(data_path,
                 epochs=300,
                 lr_g=1e-4,
                 lr_d=1e-4,
                 batch_size=2,          # ★ 关键变化: batch_size=2
                 hidden_dim=128,
                 hidden_dim2=64,
                 gnn_type='gat',
                 gan_mode='hinge',       # 'hinge' | 'wgan-gp' | 'lsgan'
                 lambda_recon=10.0,
                 lambda_adv=1.0,
                 lambda_gp=10.0,
                 d_updates=2,            # 每轮 G 更新前 D 的更新次数
                 warmup_epochs=5,
                 device='cuda',
                 save_root='./trained_model',
                 seed=42):
    """
    GAN v2 训练主循环 — batch_size=4 版本。

    Args:
        gan_mode: GAN 损失类型
            - 'hinge':   Hinge loss (稳定，推荐)
            - 'wgan-gp': WGAN with gradient penalty
            - 'lsgan':   Least Squares GAN
        batch_size:    批大小 (默认 4)
        d_updates:     每轮 Generator 更新前 Discriminator 更新次数
        warmup_epochs: 学习率 warmup 轮数
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- 创建保存路径 ----
    model_label = f'GANv2_{gnn_type}_{gan_mode}_bs{batch_size}_h{hidden_dim}_rec{lambda_recon}'
    cur_time = datetime.now().strftime("%Y_%m_%d_%H%M")
    save_dir = os.path.join(save_root, cur_time, model_label)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Experiment: {model_label}")
    print(f"Save dir:  {save_dir}")

    # ---- 加载数据 ----
    print(f"\nLoading: {data_path}")
    dataset = torch.load(data_path, weights_only=False)
    n_total = len(dataset)
    print(f"  {n_total} graphs loaded")

    # 划分训练/验证 (90%/10%)
    indices = list(range(n_total))
    random.shuffle(indices)
    n_train = int(n_total * 0.9)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  Train: {n_train} graphs → {len(train_loader)} batches (bs={batch_size})")
    print(f"  Val:   {len(val_indices)} graphs → {len(val_loader)} batches")

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

    n_params_g = sum(p.numel() for p in generator.parameters())
    n_params_d = sum(p.numel() for p in discriminator.parameters())
    print(f"\nGenerator:     {n_params_g:,} params")
    print(f"Discriminator: {n_params_d:,} params")
    print(f"Total:         {n_params_g + n_params_d:,} params")

    # ---- 优化器 ----
    opt_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # 学习率调度 (cosine annealing with warmup)
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs - warmup_epochs,
                                                        eta_min=lr_g * 0.01)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs - warmup_epochs,
                                                        eta_min=lr_d * 0.01)

    # ---- 损失函数 ----
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    # ---- 训练日志 ----
    log_path = os.path.join(save_dir, 'metrics.csv')
    header = ("epoch,d_loss,g_loss,g_recon,g_adv,gp,"
              "sing_auc,sing_dice,m1_auc,m1_dice,m2_auc,m2_dice,edge_auc,edge_dice,"
              "lr_g,lr_d\n")
    with open(log_path, 'w') as f:
        f.write(header)

    # ---- 超参数日志 ----
    hp_path = os.path.join(save_dir, 'hyperparams.txt')
    with open(hp_path, 'w') as f:
        for k, v in dict(
            gan_mode=gan_mode, batch_size=batch_size, epochs=epochs,
            lr_g=lr_g, lr_d=lr_d, hidden_dim=hidden_dim, hidden_dim2=hidden_dim2,
            gnn_type=gnn_type, lambda_recon=lambda_recon, lambda_adv=lambda_adv,
            lambda_gp=lambda_gp, d_updates=d_updates, warmup_epochs=warmup_epochs,
            seed=seed, data_path=data_path, n_train=n_train, n_val=len(val_indices),
            generator_params=n_params_g, discriminator_params=n_params_d,
        ).items():
            f.write(f"{k}: {v}\n")

    best_edge_dice = 0.0
    best_sing_dice = 0.0

    # ---- 训练循环 ----
    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        epoch_losses = {'d': 0.0, 'g': 0.0, 'g_recon': 0.0, 'g_adv': 0.0, 'gp': 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)

            real_node = batch.y_node   # [total_N, 1] 连续标签
            real_edge = batch.y_edge   # [total_2E, 2] 连续标签

            # ============ (1) 训练 Discriminator ============
            for _ in range(d_updates):
                opt_d.zero_grad()

                # 生成假样本
                with torch.no_grad():
                    fake_node, fake_edge = generator(batch)

                # D 损失 (根据 gan_mode)
                real_score = discriminator(batch, real_node, real_edge)
                fake_score = discriminator(batch, fake_node, fake_edge)

                if gan_mode == 'hinge':
                    d_real_loss = F.relu(1.0 - real_score).mean()
                    d_fake_loss = F.relu(1.0 + fake_score).mean()
                    d_loss = d_real_loss + d_fake_loss

                elif gan_mode == 'wgan-gp':
                    d_loss = fake_score.mean() - real_score.mean()
                    gp = gradient_penalty(discriminator, batch,
                                          real_node, real_edge,
                                          fake_node, fake_edge,
                                          lambda_gp=lambda_gp)
                    d_loss = d_loss + gp
                    epoch_losses['gp'] += gp.item()

                elif gan_mode == 'lsgan':
                    d_real_loss = mse_loss(real_score, torch.ones_like(real_score))
                    d_fake_loss = mse_loss(fake_score, torch.zeros_like(fake_score))
                    d_loss = 0.5 * (d_real_loss + d_fake_loss)

                else:
                    raise ValueError(f"Unknown gan_mode: {gan_mode}")

                d_loss.backward()
                opt_d.step()

            # ============ (2) 训练 Generator ============
            opt_g.zero_grad()

            fake_node, fake_edge = generator(batch)

            # 重建损失 (L1)
            g_recon = l1_loss(fake_node, real_node) + l1_loss(fake_edge, real_edge)

            # 对抗损失
            fake_score = discriminator(batch, fake_node, fake_edge)

            if gan_mode == 'hinge':
                g_adv = -fake_score.mean()
            elif gan_mode == 'wgan-gp':
                g_adv = -fake_score.mean()
            elif gan_mode == 'lsgan':
                g_adv = 0.5 * mse_loss(fake_score, torch.ones_like(fake_score))

            g_loss = lambda_recon * g_recon + lambda_adv * g_adv
            g_loss.backward()
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

        # ---- 日志记录 ----
        n_b = len(train_loader)
        avg_d = epoch_losses['d'] / n_b
        avg_g = epoch_losses['g'] / n_b
        avg_recon = epoch_losses['g_recon'] / n_b
        avg_adv = epoch_losses['g_adv'] / n_b
        avg_gp = epoch_losses['gp'] / n_b

        current_lr_g = opt_g.param_groups[0]['lr']
        current_lr_d = opt_d.param_groups[0]['lr']

        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{avg_d:.6f},{avg_g:.6f},{avg_recon:.6f},{avg_adv:.6f},{avg_gp:.6f},"
                    f"{avg_metrics['sing_auc']:.4f},{avg_metrics['sing_dice']:.4f},"
                    f"{avg_metrics['m1_auc']:.4f},{avg_metrics['m1_dice']:.4f},"
                    f"{avg_metrics['m2_auc']:.4f},{avg_metrics['m2_dice']:.4f},"
                    f"{avg_metrics['edge_auc']:.4f},{avg_metrics['edge_dice']:.4f},"
                    f"{current_lr_g:.2e},{current_lr_d:.2e}\n")

        # ---- 保存最佳模型 ----
        if avg_metrics['edge_dice'] > best_edge_dice:
            best_edge_dice = avg_metrics['edge_dice']
            torch.save({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict(),
                'best_edge_dice': best_edge_dice,
            }, os.path.join(save_dir, 'best_model_edge.pth'))

        if avg_metrics['sing_dice'] > best_sing_dice:
            best_sing_dice = avg_metrics['sing_dice']
            torch.save({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'best_sing_dice': best_sing_dice,
            }, os.path.join(save_dir, 'best_model_sing.pth'))

        # ---- 定期保存检查点 ----
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # ---- 打印 ----
        print(f"Epoch {epoch+1:3d}: "
              f"D={avg_d:.3f} G={avg_g:.3f} recon={avg_recon:.3f} adv={avg_adv:.3f} | "
              f"Sing AUC={avg_metrics['sing_auc']:.3f} Dice={avg_metrics['sing_dice']:.3f} | "
              f"M1 AUC={avg_metrics['m1_auc']:.3f} Dice={avg_metrics['m1_dice']:.3f} | "
              f"M2 AUC={avg_metrics['m2_auc']:.3f} Dice={avg_metrics['m2_dice']:.3f} | "
              f"Edge AUC={avg_metrics['edge_auc']:.3f} Dice={avg_metrics['edge_dice']:.3f}")

        # ---- 学习率调度 (warmup 后启用) ----
        if epoch >= warmup_epochs:
            scheduler_g.step()
            scheduler_d.step()

    # ---- 训练结束 ----
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"  Best Edge Dice:  {best_edge_dice:.4f}")
    print(f"  Best Sing Dice:  {best_sing_dice:.4f}")
    print(f"  Model saved to:  {save_dir}")

    # 保存最终模型
    torch.save({
        'epoch': epochs,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'best_edge_dice': best_edge_dice,
        'best_sing_dice': best_sing_dice,
    }, os.path.join(save_dir, 'final_model.pth'))

    return generator, discriminator


# ============================= 入口 =============================
if __name__ == '__main__':
    # ---- 配置 ----
    DATA_PATH = "datasets/03_graph/merged_25cases_continuous_augmented_x7.pt"

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found: {DATA_PATH}")
        print(f"Available datasets:")
        for f in os.listdir("datasets/03_graph"):
            print(f"  - {f}")
        sys.exit(1)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- 实验配置 ----
    configs = [
        # 实验 1: Hinge GAN (稳定基线)
        {
            'gan_mode': 'hinge',
            'lambda_recon': 10.0,
            'lambda_adv': 1.0,
            'lambda_gp': 0.0,
            'd_updates': 2,
        },
        # 实验 2: WGAN-GP (梯度惩罚)
        {
            'gan_mode': 'wgan-gp',
            'lambda_recon': 10.0,
            'lambda_adv': 1.0,
            'lambda_gp': 10.0,
            'd_updates': 3,
        },
        # 实验 3: LSGAN (最小二乘)
        {
            'gan_mode': 'lsgan',
            'lambda_recon': 10.0,
            'lambda_adv': 1.0,
            'lambda_gp': 0.0,
            'd_updates': 2,
        },
    ]

    for i, cfg in enumerate(configs):
        print("\n" + "=" * 70)
        print(f"Experiment {i+1}/{len(configs)}: {cfg['gan_mode'].upper()}")
        print("=" * 70)

        train_gan_v2(
            data_path=DATA_PATH,
            epochs=300,
            lr_g=1e-4,
            lr_d=1e-4,
            batch_size=2,                 # ★ batch_size=2
            hidden_dim=128,
            hidden_dim2=64,
            gnn_type='gat',
            gan_mode=cfg['gan_mode'],
            lambda_recon=cfg['lambda_recon'],
            lambda_adv=cfg['lambda_adv'],
            lambda_gp=cfg['lambda_gp'],
            d_updates=cfg['d_updates'],
            warmup_epochs=5,
            device=DEVICE,
            save_root='./trained_model',
            seed=42,
        )
