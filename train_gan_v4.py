# train_gan_v4.py — 综合改进版 (Loss + Architecture + Training Strategy)
# =============================================================================
# 集成策略:
#   L1. Focal + Dice Loss 替换 L1（直接优化评估指标）
#   L2. 自适应任务权重 (Uncertainty Weighting)
#   A1. GeneratorGNNv4: 3 层 GNN + 独立 M1/M2 边解码器
#   A2. DropEdge 正则化 (p=0.1)，防止过拟合
#   T1. EMA of Generator weights (decay=0.999)
#   T2. 两阶段训练: Recon warmup (50 epochs) → 渐进对抗 (λ_adv: 0→0.5)
#   P1. 温度缩放后处理（验证时自动搜索最优温度）
#
# v3 保留: R1 惩罚, Label Smoothing, 不平衡 LR, D 输入噪声
# =============================================================================

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np, os, sys, random
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gan_models import GeneratorGNNv4, DiscriminatorGNN, init_weights
from loss import MultiTaskStressLoss


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ============================= EMA =============================
class EMAModel:
    """指数移动平均模型权重，推理时更稳定"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone()
                       for name, param in model.named_parameters()}
    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    def apply(self):
        """将 shadow weights 应用到模型（用于验证）"""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


# ============================= R1 =============================
def r1_penalty(D, data, real_node, real_edge, lambda_r1=1.0):
    rn = real_node.detach().requires_grad_(True)
    re = real_edge.detach().requires_grad_(True)
    score = D(data, rn, re)
    grads = torch.autograd.grad(outputs=score, inputs=[rn, re],
                                 grad_outputs=torch.ones_like(score),
                                 create_graph=True, retain_graph=True)
    gnorm = sum(g.pow(2).sum() for g in grads if g is not None)
    return (lambda_r1 / 2) * gnorm / rn.size(0)


# ============================= D 噪声 =============================
def add_disc_noise(nl, el, noise_std=0.05, training=True):
    if not training or noise_std <= 0: return nl, el
    return (nl + torch.randn_like(nl) * noise_std,
            el + torch.randn_like(el) * noise_std)


# ============================= 验证指标 =============================
@torch.no_grad()
def compute_metrics(node_pred, edge_pred, node_target, edge_target,
                    sing_temp=1.0, edge_temp=1.0):
    """支持温度缩放的指标计算"""
    from sklearn.metrics import roc_auc_score
    node_p = (node_pred / sing_temp).cpu().numpy().ravel()
    node_t = node_target.cpu().numpy().ravel()
    edge_p = edge_pred.cpu().numpy()
    edge_t = edge_target.cpu().numpy()

    def safe_auc(y_t, y_p):
        if y_t.sum() == 0 or (y_t == 1).all(): return 0.5
        return roc_auc_score(y_t, y_p)

    sing_th = np.percentile(node_t, 99) if node_t.max() > 0 else 0.5
    sing_bin = (node_t > sing_th).astype(int)
    results = {
        'sing_auc': safe_auc(sing_bin, node_p),
        'sing_dice': 2 * ((node_p > 0.5) * sing_bin).sum() /
                      ((node_p > 0.5).sum() + sing_bin.sum() + 1e-8),
    }

    for label, p, t in [('m1', edge_p[:, 0], edge_t[:, 0]),
                         ('m2', edge_p[:, 1], edge_t[:, 1])]:
        th = np.percentile(t, 84) if t.max() > 0 else 0.5
        tb = (t > th).astype(int)
        pb = ((p / edge_temp) > 0.5).astype(int)
        results[f'{label}_auc'] = safe_auc(tb, p)
        results[f'{label}_dice'] = 2*(pb*tb).sum()/(pb.sum()+tb.sum()+1e-8)

    e_all_t = np.concatenate([(edge_t[:,0] > np.percentile(edge_t[:,0],84)).astype(int),
                               (edge_t[:,1] > np.percentile(edge_t[:,1],84)).astype(int)])
    e_all_p = np.concatenate([edge_p[:,0], edge_p[:,1]])
    results['edge_auc'] = safe_auc(e_all_t, e_all_p)
    results['edge_dice'] = 2*((e_all_p>0.5)*e_all_t).sum()/((e_all_p>0.5).sum()+e_all_t.sum()+1e-8)
    return results


# ============================= 自适应任务权重 =============================
class UncertaintyWeighting(nn.Module):
    """
    Uncertainty-based multi-task weighting (Kendall et al. 2018)
    L_total = Σ L_i / (2σ_i²) + log(σ_i)
    学习 log σ 而非 σ（数值稳定）
    """
    def __init__(self, n_tasks=2):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        """losses: list of [sing_loss, edge_loss]"""
        weighted = 0.0
        for i, loss in enumerate(losses):
            sigma2 = self.log_sigma[i].exp()
            weighted = weighted + loss / (2 * sigma2) + 0.5 * self.log_sigma[i]
        return weighted, self.log_sigma.detach().cpu().tolist()


# ============================= 温度缩放搜索 =============================
@torch.no_grad()
def find_best_temperature(G, val_loader, device, n_trials=20):
    """在验证集上搜索最优温度（最大化 Sing Dice）"""
    # 收集所有预测
    all_node_p, all_node_t = [], []
    for batch in val_loader:
        batch = batch.to(device)
        npred, _, _ = G(batch)
        all_node_p.append(npred.cpu())
        all_node_t.append(batch.y_node.cpu())
    node_p = torch.cat(all_node_p).numpy().ravel()
    node_t = torch.cat(all_node_t).numpy().ravel()
    sing_th = np.percentile(node_t, 99) if node_t.max() > 0 else 0.5
    sing_bin = (node_t > sing_th).astype(int)

    best_temp, best_dice = 1.0, 0.0
    for temp in np.logspace(-1, 1, n_trials):
        p = node_p / temp
        dice = 2 * ((p > 0.5) * sing_bin).sum() / ((p > 0.5).sum() + sing_bin.sum() + 1e-8)
        if dice > best_dice:
            best_dice, best_temp = dice, temp
    return best_temp, best_dice


# ============================= 主训练 =============================
def train_gan_v4(data_path,
                 epochs=300, lr_g=2e-4, lr_d=5e-5, batch_size=2,
                 hidden_dim_g=128, hidden_dim_d=64, gnn_type='gat',
                 gan_mode='hinge',
                 lambda_recon=5.0,
                 lambda_adv_max=0.5,        # ★ 最终对抗权重
                 lambda_r1=1.0, label_smooth=0.9, d_noise=0.05,
                 sing_dice_w=0.7, psl_dice_w=0.5,
                 w_sing=2.0, w_psl1=1.0, w_psl2=1.0,
                 dropedge=0.1,              # ★ DropEdge 概率
                 ema_decay=0.999,           # ★ EMA 衰减
                 warmup_recon=50,           # ★ 纯重建 warmup 轮数
                 adv_ramp_epochs=50,        # ★ 对抗权重从 0 渐变到 max 的轮数
                 g_clip_norm=5.0,
                 use_uncertainty=False,     # ★ 是否使用自适应任务权重
                 device='cuda', save_root='./trained_model', seed=42):
    """
    GAN v4 — 综合改进版。

    新增策略:
        warmup_recon:  前 N 轮仅用重建损失（不用对抗），给 G 一个良好的起点
        adv_ramp:      然后渐进引入对抗损失 (λ_adv: 0 → max)
        ema_decay:     EMA 模型权重（推理更稳定）
        dropedge:      DropEdge 正则化
        use_uncertainty: 自适应任务权重
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    label = (f'GANv4_{gnn_type}_{gan_mode}_FocalDice_EMA'
             f'{"_Unc" if use_uncertainty else ""}_DE{dropedge}_wup{warmup_recon}')
    cur_time = datetime.now().strftime("%Y_%m_%d_%H%M")
    save_dir = os.path.join(save_root, cur_time, label)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Experiment: {label}\nSave dir:  {save_dir}")

    # ---- 数据 ----
    print(f"\nLoading: {data_path}")
    dataset = torch.load(data_path, weights_only=False)
    indices = list(range(len(dataset))); random.shuffle(indices)
    n_train = int(len(dataset) * 0.9)
    train_loader = DataLoader([dataset[i] for i in indices[:n_train]],
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([dataset[i] for i in indices[n_train:]],
                             batch_size=batch_size, shuffle=False)
    print(f"  Train: {n_train} → {len(train_loader)} batches (bs={batch_size})")

    # ---- 模型 ----
    G = GeneratorGNNv4(input_dim=12, hidden_dim=hidden_dim_g, hidden_dim2=64,
                        gnn_type=gnn_type, dropout=0.1, dropedge=dropedge).to(device)
    G.apply(init_weights)
    D = DiscriminatorGNN(input_dim=12, node_label_dim=1, hidden_dim=hidden_dim_d,
                          gnn_type=gnn_type, dropout=0.3).to(device)
    D.apply(init_weights)
    ema = EMAModel(G, decay=ema_decay)
    print(f"G (v4): {sum(p.numel() for p in G.parameters()):,} | "
          f"D: {sum(p.numel() for p in D.parameters()):,} | "
          f"DropEdge={dropedge} | EMA decay={ema_decay}")

    # ---- 优化器 ----
    opt_g = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    sch_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs-warmup_recon, eta_min=lr_g*0.01)
    sch_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs-warmup_recon, eta_min=lr_d*0.01)

    # ---- 损失 ----
    recon_loss = MultiTaskStressLoss(
        sing_gamma=4.0, sing_alpha=0.95, sing_dice_weight=sing_dice_w,
        psl_gamma=1.0, psl_alpha=0.75, psl_dice_weight=psl_dice_w,
        w_sing=w_sing, w_psl1=w_psl1, w_psl2=w_psl2)
    mse = nn.MSELoss()
    uncert = UncertaintyWeighting(n_tasks=2).to(device) if use_uncertainty else None
    if uncert:
        print(f"  Uncertainty weighting enabled (learned σ per task)")
        opt_g.add_param_group({'params': uncert.parameters(), 'lr': lr_g * 0.1})

    # ---- 日志 ----
    log_path = os.path.join(save_dir, 'metrics.csv')
    with open(log_path, 'w') as f:
        f.write("epoch,d_loss,g_loss,g_recon,g_adv,gp_r1,lambda_adv,"
                "sing_auc,sing_dice,sing_dice_ema,m1_auc,m1_dice,m2_auc,m2_dice,"
                "edge_auc,edge_dice,edge_dice_ema,lr_g,lr_d\n")
    with open(os.path.join(save_dir, 'hyperparams.txt'), 'w') as f:
        for k, v in dict(gan_mode=gan_mode, batch_size=batch_size, epochs=epochs,
                         hidden_dim_g=hidden_dim_g, hidden_dim_d=hidden_dim_d,
                         lambda_recon=lambda_recon, lambda_adv_max=lambda_adv_max,
                         warmup_recon=warmup_recon, adv_ramp_epochs=adv_ramp_epochs,
                         ema_decay=ema_decay, dropedge=dropedge,
                         use_uncertainty=use_uncertainty,
                         sing_dice_w=sing_dice_w, psl_dice_w=psl_dice_w,
                         g_clip_norm=g_clip_norm,
        ).items(): f.write(f"{k}: {v}\n")

    best_edge_dice, best_sing_dice = 0.0, 0.0
    best_sing_dice_ema, best_edge_dice_ema = 0.0, 0.0

    for epoch in range(epochs):
        G.train(); D.train()
        losses = {'d': 0.0, 'g': 0.0, 'g_recon': 0.0, 'g_adv': 0.0, 'gp_r1': 0.0}

        # ★ 两阶段对抗权重调度
        if epoch < warmup_recon:
            lambda_adv = 0.0  # 纯重建
        elif epoch < warmup_recon + adv_ramp_epochs:
            progress = (epoch - warmup_recon) / adv_ramp_epochs
            lambda_adv = lambda_adv_max * progress  # 线性渐变
        else:
            lambda_adv = lambda_adv_max

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [λ_adv={lambda_adv:.2f}]")
        for batch in pbar:
            batch = batch.to(device)
            real_node, real_edge = batch.y_node, batch.y_edge

            # ---- (1) 训练 D (仅在 λ_adv > 0 时) ----
            # ★ D 需要完整的边集合，禁用 DropEdge 生成
            if lambda_adv > 0:
                for _ in range(1):  # d_updates=1
                    opt_d.zero_grad()
                    # 临时禁用 DropEdge 以生成完整边预测
                    orig_de = G.dropedge; G.dropedge = 0.0
                    with torch.no_grad():
                        fn, fe, _ = G(batch)
                    G.dropedge = orig_de  # 恢复
                    rn, re = add_disc_noise(real_node, real_edge, d_noise)
                    fn2, fe2 = add_disc_noise(fn, fe, d_noise)
                    rs = D(batch, rn, re); fs = D(batch, fn2, fe2)
                    rt = torch.full_like(rs, label_smooth)
                    if gan_mode == 'hinge':
                        d_loss = F.relu(rt-rs).mean() + F.relu(fs+rt).mean()
                    else:
                        d_loss = 0.5*(mse(rs, rt) + mse(fs, torch.full_like(fs, 1-label_smooth)))
                    if lambda_r1 > 0:
                        r1 = r1_penalty(D, batch, rn, re, lambda_r1)
                        d_loss = d_loss + r1; losses['gp_r1'] += r1.item()
                    d_loss.backward(); opt_d.step()
                losses['d'] += d_loss.item()
            else:
                d_loss = torch.tensor(0.0)

            # ---- (2) 训练 G ----
            opt_g.zero_grad()
            # ★ Recon forward: 使用 DropEdge (正则化)
            fake_node, fake_edge, edge_mask = G(batch)

            # ★ DropEdge 对齐: edge_mask 标记保留的边
            if edge_mask is not None:
                real_edge_aligned = real_edge[edge_mask]
            else:
                real_edge_aligned = real_edge
            # 分别计算 sing 和 edge 的 tensor 损失（用于可选的 uncertainty weighting）
            sing_focal = recon_loss.focal_sing(fake_node, real_node)
            sing_dice = recon_loss.dice_sing(fake_node, real_node)
            loss_sing_tensor = sing_focal + recon_loss.sing_dice_w * sing_dice

            psl1_focal = recon_loss.focal_psl1(fake_edge[:, 0:1], real_edge_aligned[:, 0:1])
            psl1_dice = recon_loss.dice_psl1(fake_edge[:, 0:1], real_edge_aligned[:, 0:1])
            loss_psl1_tensor = psl1_focal + recon_loss.psl1_dice_w * psl1_dice

            psl2_focal = recon_loss.focal_psl2(fake_edge[:, 1:2], real_edge_aligned[:, 1:2])
            psl2_dice = recon_loss.dice_psl2(fake_edge[:, 1:2], real_edge_aligned[:, 1:2])
            loss_psl2_tensor = psl2_focal + recon_loss.psl2_dice_w * psl2_dice

            loss_sing = recon_loss.w_sing * loss_sing_tensor
            loss_edge = recon_loss.w_psl1 * loss_psl1_tensor + recon_loss.w_psl2 * loss_psl2_tensor

            if use_uncertainty and uncert is not None:
                g_recon, _ = uncert([loss_sing, loss_edge])
            else:
                g_recon = loss_sing + loss_edge

            # 对抗损失 (需要完整边集合)
            if lambda_adv > 0:
                # 用无 DropEdge 的 forward 生成完整边预测 (用于 D 输入)
                orig_de = G.dropedge; G.dropedge = 0.0
                fake_node_full, fake_edge_full, _ = G(batch)
                G.dropedge = orig_de
                fn2, fe2 = add_disc_noise(fake_node_full, fake_edge_full, d_noise)
                fs = D(batch, fn2, fe2)
                g_adv = -fs.mean() if gan_mode == 'hinge' else 0.5*mse(fs, torch.full_like(fs, label_smooth))
                losses['g_adv'] += g_adv.item()
            else:
                g_adv = torch.tensor(0.0, device=device)

            g_loss = lambda_recon * g_recon + lambda_adv * g_adv
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), g_clip_norm)
            opt_g.step()

            # ★ EMA 更新
            ema.update()

            losses['g'] += g_loss.item()
            losses['g_recon'] += g_recon.item()

            pbar.set_postfix(D=f'{d_loss.item():.3f}', G=f'{g_loss.item():.3f}',
                           recon=f'{g_recon.item():.3f}', adv=f'{g_adv.item():.3f}',
                           singF=f'{sing_focal.item():.3f}', singD=f'{sing_dice.item():.3f}')

        # ---- 验证 ----
        G.eval()
        all_metrics, all_metrics_ema = [], []

        # 标准模型指标
        for batch in val_loader:
            batch = batch.to(device)
            with torch.no_grad(): npred, epred, _ = G(batch)
            all_metrics.append(compute_metrics(npred, epred, batch.y_node, batch.y_edge))
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

        # EMA 模型指标
        ema.apply()
        for batch in val_loader:
            batch = batch.to(device)
            with torch.no_grad(): npred, epred, _ = G(batch)
            all_metrics_ema.append(compute_metrics(npred, epred, batch.y_node, batch.y_edge))
        avg_ema = {k: np.mean([m[k] for m in all_metrics_ema]) for k in all_metrics_ema[0]}
        # 恢复原权重
        for name, param in G.named_parameters():
            if name in ema.shadow:
                param.data.copy_(ema.shadow[name])
        G.train()

        # ---- 日志 ----
        n_b = len(train_loader)
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{losses['d']/n_b:.6f},{losses['g']/n_b:.6f},"
                    f"{losses['g_recon']/n_b:.6f},{losses['g_adv']/n_b:.6f},{losses['gp_r1']/n_b:.6f},"
                    f"{lambda_adv:.4f},"
                    f"{avg['sing_auc']:.4f},{avg['sing_dice']:.4f},{avg_ema['sing_dice']:.4f},"
                    f"{avg['m1_auc']:.4f},{avg['m1_dice']:.4f},"
                    f"{avg['m2_auc']:.4f},{avg['m2_dice']:.4f},"
                    f"{avg['edge_auc']:.4f},{avg['edge_dice']:.4f},{avg_ema['edge_dice']:.4f},"
                    f"{opt_g.param_groups[0]['lr']:.2e},{opt_d.param_groups[0]['lr']:.2e}\n")

        # ---- 最佳保存 (基于 EMA 指标) ----
        if avg_ema['edge_dice'] > best_edge_dice_ema:
            best_edge_dice_ema = avg_ema['edge_dice']
            ema.apply()
            torch.save({'epoch': epoch+1, 'generator': G.state_dict()},
                       os.path.join(save_dir, 'best_model_edge_ema.pth'))
            # 恢复
            for name, param in G.named_parameters():
                if name in ema.shadow: param.data.copy_(ema.shadow[name])

        if avg_ema['sing_dice'] > best_sing_dice_ema:
            best_sing_dice_ema = avg_ema['sing_dice']
            ema.apply()
            torch.save({'epoch': epoch+1, 'generator': G.state_dict()},
                       os.path.join(save_dir, 'best_model_sing_ema.pth'))
            for name, param in G.named_parameters():
                if name in ema.shadow: param.data.copy_(ema.shadow[name])

        # 非 EMA 最佳
        if avg['edge_dice'] > best_edge_dice:
            best_edge_dice = avg['edge_dice']
            torch.save({'epoch': epoch+1, 'generator': G.state_dict(), 'discriminator': D.state_dict()},
                       os.path.join(save_dir, 'best_model_edge.pth'))
        if avg['sing_dice'] > best_sing_dice:
            best_sing_dice = avg['sing_dice']
            torch.save({'epoch': epoch+1, 'generator': G.state_dict()},
                       os.path.join(save_dir, 'best_model_sing.pth'))
        if (epoch+1) % 100 == 0:
            torch.save({'epoch': epoch+1, 'generator': G.state_dict(), 'discriminator': D.state_dict()},
                       os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        print(f"Epoch {epoch+1:3d}: λ_adv={lambda_adv:.2f} "
              f"D={losses['d']/n_b:.3f} G={losses['g']/n_b:.3f} "
              f"recon={losses['g_recon']/n_b:.3f} adv={losses['g_adv']/n_b:.3f} | "
              f"Sing AUC={avg['sing_auc']:.3f} D_raw={avg['sing_dice']:.3f} D_ema={avg_ema['sing_dice']:.3f} | "
              f"Edge D_raw={avg['edge_dice']:.3f} D_ema={avg_ema['edge_dice']:.3f}")

        if epoch >= warmup_recon: sch_g.step(); sch_d.step()

    # ---- 最终温度缩放 ----
    print(f"\n{'='*60}")
    ema.apply()
    sing_temp, sing_dice_cal = find_best_temperature(G, val_loader, device)
    print(f"Optimal Sing temperature: {sing_temp:.3f} → Sing Dice (calibrated) = {sing_dice_cal:.4f}")

    # 计算最终 EMA 指标
    all_final = []
    for batch in val_loader:
        batch = batch.to(device)
        with torch.no_grad(): npred, epred, _ = G(batch)
        all_final.append(compute_metrics(npred, epred, batch.y_node, batch.y_edge,
                                          sing_temp=sing_temp))
    final_avg = {k: np.mean([m[k] for m in all_final]) for k in all_final[0]}
    print(f"Final (EMA + Temp={sing_temp:.3f}): Sing AUC={final_avg['sing_auc']:.3f} "
          f"Dice={final_avg['sing_dice']:.3f} | Edge Dice={final_avg['edge_dice']:.3f}")
    print(f"Best EMA:  Sing Dice={best_sing_dice_ema:.4f}  Edge Dice={best_edge_dice_ema:.4f}")
    print(f"Best Raw:  Sing Dice={best_sing_dice:.4f}  Edge Dice={best_edge_dice:.4f}")

    # 恢复原权重后保存最终模型
    for name, param in G.named_parameters():
        if name in ema.shadow: param.data.copy_(ema.shadow[name])
    torch.save({'epoch': epochs, 'generator': G.state_dict(), 'discriminator': D.state_dict(),
                'best_sing_dice_ema': best_sing_dice_ema, 'best_edge_dice_ema': best_edge_dice_ema,
                'sing_temp': sing_temp, 'sing_dice_cal': sing_dice_cal,
    }, os.path.join(save_dir, 'final_model.pth'))
    print(f"Saved to: {save_dir}")
    return G, D, save_dir


# ========== 入口 ==========
if __name__ == '__main__':
    DATA_PATH = "datasets/03_graph/merged_25cases_continuous_augmented_x7.pt"
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found: {DATA_PATH}"); sys.exit(1)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    configs = [
        # 实验 A: 核心改进 (Focal+Dice + EMA + 两阶段)
        {
            'label': 'A: Focal+Dice + EMA + 2-Stage',
            'use_uncertainty': False,
            'dropedge': 0.1,
            'warmup_recon': 50,
            'lambda_recon': 5.0,
        },
        # 实验 B: + 自适应任务权重
        {
            'label': 'B: + Uncertainty Weighting',
            'use_uncertainty': True,
            'dropedge': 0.1,
            'warmup_recon': 50,
            'lambda_recon': 5.0,
        },
        # 实验 C: 激进配置 (更长 warmup, 更高 DropEdge)
        {
            'label': 'C: Aggressive (warmup=100, DE=0.2)',
            'use_uncertainty': False,
            'dropedge': 0.2,
            'warmup_recon': 100,
            'lambda_recon': 3.0,
        },
    ]

    for i, cfg in enumerate(configs):
        print("\n" + "=" * 70)
        print(f"Experiment {i+1}/{len(configs)}: {cfg['label']}")
        print("=" * 70)
        train_gan_v4(
            data_path=DATA_PATH, epochs=300, batch_size=2,
            hidden_dim_g=128, hidden_dim_d=64, gnn_type='gat',
            gan_mode='hinge',
            lambda_recon=cfg['lambda_recon'],
            use_uncertainty=cfg['use_uncertainty'],
            dropedge=cfg['dropedge'],
            warmup_recon=cfg['warmup_recon'],
            device=DEVICE,
        )
