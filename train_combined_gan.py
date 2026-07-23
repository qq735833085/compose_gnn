# train_combined_gan.py — 实验 C: Edge-Aware D + Focal+Dice Loss
# =============================================================================
# 组合实验 A 和 B 的改进：
#   1. EdgeAwareDiscriminatorGNN — 边预测获得直接梯度路径
#   2. MultiTaskStressLoss (Focal + Dice) — 处理极端类别不平衡
#
# 这是两个正交改进的组合：D 架构改善边梯度 + 更好的重建损失。
#
# 对比基线：train_gan_v5_compare.py (v5 Hinge GAN + GAT)
# =============================================================================

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np, os, sys, random
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gan_models import GeneratorGNN, EdgeAwareDiscriminatorGNN, init_weights
from loss import MultiTaskStressLoss


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def r1_penalty(D, data, rn, re, l=1.0):
    rn = rn.detach().requires_grad_(True)
    re = re.detach().requires_grad_(True)
    s = D(data, rn, re)
    gs = torch.autograd.grad(outputs=s, inputs=[rn, re],
                              grad_outputs=torch.ones_like(s),
                              create_graph=True, retain_graph=True,
                              allow_unused=True)
    return (l / 2) * sum(g.pow(2).sum() for g in gs if g is not None) / rn.size(0)


def add_disc_noise(nl, el, s=0.05, t=True):
    if not t or s <= 0: return nl, el
    return (nl + torch.randn_like(nl) * s, el + torch.randn_like(el) * s)


@torch.no_grad()
def compute_metrics(node_pred, edge_pred, node_target, edge_target):
    from sklearn.metrics import roc_auc_score
    np_, nt = node_pred.cpu().numpy().ravel(), node_target.cpu().numpy().ravel()
    ep, et = edge_pred.cpu().numpy(), edge_target.cpu().numpy()

    def sauc(yt, yp):
        if yt.sum() == 0 or (yt == 1).all(): return 0.5
        return roc_auc_score(yt, yp)

    st = np.percentile(nt, 99) if nt.max() > 0 else 0.5
    sb = (nt > st).astype(int)
    r = {'sing_auc': sauc(sb, np_),
         'sing_dice': 2 * ((np_ > 0.5) * sb).sum() / ((np_ > 0.5).sum() + sb.sum() + 1e-8)}

    for lb, p, t in [('m1', ep[:, 0], et[:, 0]), ('m2', ep[:, 1], et[:, 1])]:
        th = np.percentile(t, 84) if t.max() > 0 else 0.5
        tb = (t > th).astype(int); pb = (p > 0.5).astype(int)
        r[f'{lb}_auc'] = sauc(tb, p)
        r[f'{lb}_dice'] = 2 * (pb * tb).sum() / (pb.sum() + tb.sum() + 1e-8)

    ea_t = np.concatenate([(et[:, 0] > np.percentile(et[:, 0], 84)).astype(int),
                            (et[:, 1] > np.percentile(et[:, 1], 84)).astype(int)])
    ea_p = np.concatenate([ep[:, 0], ep[:, 1]])
    r['edge_auc'] = sauc(ea_t, ea_p)
    r['edge_dice'] = 2 * ((ea_p > 0.5) * ea_t).sum() / ((ea_p > 0.5).sum() + ea_t.sum() + 1e-8)
    return r


def train_combined(data_path, epochs=300, lr_g=2e-4, lr_d=5e-5, batch_size=8,
                   hidden_dim_g=128, hidden_dim_d=64,
                   lambda_recon=1.0, lambda_adv=0.5, lambda_r1=1.0,
                   label_smooth=0.9, d_noise=0.05, d_updates=1,
                   g_clip_norm=5.0, warmup_epochs=5,
                   device='cuda', save_root='./trained_model/combined', seed=42):
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    l1 = nn.L1Loss()

    label = 'combined_edge_aware_focal'
    cur_time = datetime.now().strftime("%Y_%m_%d_%H%M")
    save_dir = os.path.join(save_root, cur_time, label)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Experiment C: Edge-Aware D + Focal+Dice Loss")
    print(f"Save dir: {save_dir}")

    dataset = torch.load(data_path, weights_only=False)
    indices = list(range(len(dataset))); random.shuffle(indices)
    n_train = int(len(dataset) * 0.9)
    train_loader = DataLoader([dataset[i] for i in indices[:n_train]],
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([dataset[i] for i in indices[n_train:]],
                             batch_size=batch_size, shuffle=False)

    G = GeneratorGNN(input_dim=12, hidden_dim=hidden_dim_g, hidden_dim2=64,
                      gnn_type='gat', dropout=0.1).to(device); G.apply(init_weights)

    # ★ Edge-Aware D (实验 A)
    D = EdgeAwareDiscriminatorGNN(input_dim=12, node_label_dim=1, hidden_dim=hidden_dim_d,
                                   gnn_type='gat', dropout=0.3).to(device); D.apply(init_weights)
    print(f"G (GAT): {sum(p.numel() for p in G.parameters()):,} | "
          f"D (EdgeAware): {sum(p.numel() for p in D.parameters()):,}")

    # ★ Focal + Dice 多任务损失 (实验 B)
    criterion = MultiTaskStressLoss(
        sing_gamma=4.0, sing_alpha=0.95, sing_dice_weight=0.3,
        psl_gamma=1.0, psl_alpha=0.75, psl_dice_weight=0.5,
        w_sing=2.0, w_psl1=1.0, w_psl2=1.0,
    ).to(device)

    opt_g = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    sg = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs - warmup_epochs, eta_min=lr_g * 0.01)
    sd = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs - warmup_epochs, eta_min=lr_d * 0.01)

    log_path = os.path.join(save_dir, 'metrics.csv')
    with open(log_path, 'w') as f:
        f.write("epoch,d_loss,g_loss,g_recon,g_adv,gp_r1,"
                "sing_auc,sing_dice,m1_auc,m1_dice,m2_auc,m2_dice,edge_auc,edge_dice,"
                "l_sing_focal,l_sing_dice,l_psl1_focal,l_psl1_dice,l_psl2_focal,l_psl2_dice,"
                "lr_g,lr_d\n")

    with open(os.path.join(save_dir, 'hyperparams.txt'), 'w') as f:
        for k, v in dict(experiment='combined_edge_aware_focal', epochs=epochs,
                         lr_g=lr_g, lr_d=lr_d, batch_size=batch_size,
                         hidden_dim_g=hidden_dim_g, hidden_dim_d=hidden_dim_d,
                         lambda_recon=lambda_recon, lambda_adv=lambda_adv, lambda_r1=lambda_r1,
                         label_smooth=label_smooth, d_noise=d_noise, seed=seed,
                         loss='MultiTaskStressLoss(Focal+Dice)',
                         discriminator='EdgeAwareDiscriminatorGNN',
                         w_sing=2.0, w_psl1=1.0, w_psl2=1.0,
                         sing_gamma=4.0, sing_alpha=0.95,
                         psl_gamma=1.0, psl_alpha=0.75).items():
            f.write(f"{k}: {v}\n")

    best_edge, best_sing = 0.0, 0.0
    for epoch in range(epochs):
        G.train(); D.train()
        ls = {'d': 0, 'g': 0, 'recon': 0, 'adv': 0, 'r1': 0}
        sub_losses = {'sf': 0, 'sd': 0, 'p1f': 0, 'p1d': 0, 'p2f': 0, 'p2d': 0}
        pbar = tqdm(train_loader, desc=f"[Combined] E{epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            rn, re = batch.y_node, batch.y_edge

            # (1) D
            for _ in range(d_updates):
                opt_d.zero_grad()
                with torch.no_grad(): fn, fe = G(batch)
                rn2, re2 = add_disc_noise(rn, re, d_noise)
                fn2, fe2 = add_disc_noise(fn, fe, d_noise)
                rs, fs = D(batch, rn2, re2), D(batch, fn2, fe2)
                rt = torch.full_like(rs, label_smooth)
                d_loss = F.relu(rt - rs).mean() + F.relu(fs + rt).mean()
                if lambda_r1 > 0:
                    r1 = r1_penalty(D, batch, rn2, re2, lambda_r1)
                    d_loss = d_loss + r1; ls['r1'] += r1.item()
                d_loss.backward(); opt_d.step()
            ls['d'] += d_loss.item()

            # (2) G ★ Focal+Dice 重建 + 对抗
            opt_g.zero_grad()
            fn, fe = G(batch)
            g_recon, log_dict = criterion(fn, rn, fe, re)  # ★ Focal + Dice
            fn2, fe2 = add_disc_noise(fn, fe, d_noise)
            fs = D(batch, fn2, fe2)  # ★ Edge-Aware D
            g_adv = -fs.mean()
            g_loss = lambda_recon * g_recon + lambda_adv * g_adv
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), g_clip_norm)
            opt_g.step()
            ls['g'] += g_loss.item(); ls['recon'] += g_recon.item(); ls['adv'] += g_adv.item()
            sub_losses['sf'] += log_dict['sing/focal']
            sub_losses['sd'] += log_dict['sing/dice']
            sub_losses['p1f'] += log_dict['psl1/focal']
            sub_losses['p1d'] += log_dict['psl1/dice']
            sub_losses['p2f'] += log_dict['psl2/focal']
            sub_losses['p2d'] += log_dict['psl2/dice']
            pbar.set_postfix(D=f'{d_loss.item():.3f}', G=f'{g_loss.item():.3f}',
                           recon=f'{g_recon.item():.3f}', adv=f'{g_adv.item():.3f}')

        # Val
        G.eval()
        all_m = []
        for batch in val_loader:
            batch = batch.to(device)
            with torch.no_grad(): npred, epred = G(batch)
            all_m.append(compute_metrics(npred, epred, batch.y_node, batch.y_edge))
        avg = {k: np.mean([m[k] for m in all_m]) for k in all_m[0]}

        nb = len(train_loader)
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{ls['d']/nb:.6f},{ls['g']/nb:.6f},{ls['recon']/nb:.6f},"
                    f"{ls['adv']/nb:.6f},{ls['r1']/nb:.6f},"
                    f"{avg['sing_auc']:.4f},{avg['sing_dice']:.4f},"
                    f"{avg['m1_auc']:.4f},{avg['m1_dice']:.4f},"
                    f"{avg['m2_auc']:.4f},{avg['m2_dice']:.4f},"
                    f"{avg['edge_auc']:.4f},{avg['edge_dice']:.4f},"
                    f"{sub_losses['sf']/nb:.6f},{sub_losses['sd']/nb:.6f},"
                    f"{sub_losses['p1f']/nb:.6f},{sub_losses['p1d']/nb:.6f},"
                    f"{sub_losses['p2f']/nb:.6f},{sub_losses['p2d']/nb:.6f},"
                    f"{opt_g.param_groups[0]['lr']:.2e},{opt_d.param_groups[0]['lr']:.2e}\n")

        if avg['edge_dice'] > best_edge:
            best_edge = avg['edge_dice']
            torch.save({'G': G.state_dict(), 'D': D.state_dict()},
                       os.path.join(save_dir, 'best_edge.pth'))
        if avg['sing_dice'] > best_sing:
            best_sing = avg['sing_dice']
            torch.save({'G': G.state_dict()},
                       os.path.join(save_dir, 'best_sing.pth'))
        if (epoch + 1) % 100 == 0:
            torch.save({'G': G.state_dict(), 'D': D.state_dict()},
                       os.path.join(save_dir, f'ckpt_{epoch+1}.pth'))

        print(f"[Combined] E{epoch+1}: D={ls['d']/nb:.3f} G={ls['g']/nb:.3f} "
              f"recon={ls['recon']/nb:.3f} adv={ls['adv']/nb:.3f} | "
              f"Sing AUC={avg['sing_auc']:.3f} D={avg['sing_dice']:.3f} | "
              f"Edge AUC={avg['edge_auc']:.3f} D={avg['edge_dice']:.3f}")
        if epoch >= warmup_epochs: sg.step(); sd.step()

    print(f"\n[Combined] Done! Best Sing Dice={best_sing:.4f} Edge Dice={best_edge:.4f}")
    return G, D, save_dir, best_sing, best_edge


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Exp10: Combined (Edge-Aware D + Focal+Dice)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    DATA = "datasets/03_graph/merged_25cases_continuous_augmented_x7.pt"
    if not os.path.exists(DATA): print(f"Missing: {DATA}"); sys.exit(1)
    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print(f"Experiment C: Combined (Edge-Aware D + Focal+Dice) | BS={args.batch_size}")
    print("=" * 60)
    train_combined(DATA, epochs=args.epochs, batch_size=args.batch_size,
                   device=DEV, seed=args.seed)
