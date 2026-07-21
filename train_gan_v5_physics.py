# train_gan_v5_physics.py — 物理约束 GAN
# =============================================================================
# 在 v3 稳定框架上添加 3 项物理约束:
#
#   P1. 稀疏性约束 (Sparsity)
#      L_sparse = mean(|sing_prob|)
#      物理依据: 奇异点仅占节点总数的 0.02%，强制稀疏预测
#
#   P2. 图拉普拉斯平滑 (Laplacian Smoothness)
#      L_smooth = mean((h_src - h_dst)^2) across all edges
#      物理依据: 应力场连续，相邻节点概率不应突变
#      等价于 h^T·L·h / |E|，仅对非奇异处施加
#
#   P3. 奇异点-应力方向一致性 (Singularity-PSL Consistency)
#      L_consist = mean(sing * |edge_psl_std|) per node
#      物理依据: 奇异点处应力各向同性，PSL 方向应退化
#      非奇异点处邻域 PSL 应一致
#
# 总损失:
#   G_loss = λ_recon * L_recon + λ_adv * L_adv
#           + λ_sparse * L_sparse + λ_smooth * L_smooth + λ_consist * L_consist
# =============================================================================

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np, os, sys, random
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gan_models import GeneratorGNN, DiscriminatorGNN, init_weights

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def r1_penalty(D, data, rn, re, l=1.0):
    rn = rn.detach().requires_grad_(True); re = re.detach().requires_grad_(True)
    s = D(data, rn, re)
    gs = torch.autograd.grad(outputs=s, inputs=[rn, re],
                              grad_outputs=torch.ones_like(s),
                              create_graph=True, retain_graph=True,
                              allow_unused=True)  # GCN/SAGE don't use edge features
    return (l/2)*sum(g.pow(2).sum() for g in gs if g is not None)/rn.size(0)

def add_disc_noise(nl, el, s=0.05, t=True):
    if not t or s<=0: return nl, el
    return (nl+torch.randn_like(nl)*s, el+torch.randn_like(el)*s)

# ===================== 物理约束 =====================

def physics_losses(sing_prob, edge_prob, edge_index, batch):
    """
    计算三项物理约束损失（带 NaN 保护）。
    """
    eps = 1e-8
    src, dst = edge_index[0], edge_index[1]
    device = sing_prob.device

    # ---- P1: 稀疏性 ----
    L_sparse = sing_prob.mean()

    # ---- P2: 图拉普拉斯平滑 ----
    smooth_weight = 1.0 - sing_prob.squeeze(1).clamp(0, 1)
    src_w = smooth_weight[src]
    dst_w = smooth_weight[dst]
    avg_w = (src_w + dst_w).clamp(min=0) / 2

    L_smooth = (avg_w * (sing_prob[src].squeeze(1) - sing_prob[dst].squeeze(1)).pow(2)).mean()

    # ---- P3: 奇异点-应力方向一致性 ----
    num_nodes = sing_prob.size(0)
    from torch_geometric.utils import scatter
    m1 = edge_prob[:, 0].clamp(min=eps, max=1.0 - eps)
    m2 = edge_prob[:, 1].clamp(min=eps, max=1.0 - eps)

    m1_mean = scatter(m1, dst, dim=0, dim_size=num_nodes, reduce='mean')
    m2_mean = scatter(m2, dst, dim=0, dim_size=num_nodes, reduce='mean')

    deg = scatter(torch.ones_like(m1), dst, dim=0, dim_size=num_nodes, reduce='sum')
    mask = deg > 1

    if mask.sum() > 0:
        m1_var = scatter((m1 - m1_mean[dst]).pow(2), dst, dim=0, dim_size=num_nodes, reduce='mean')
        m2_var = scatter((m2 - m2_mean[dst]).pow(2), dst, dim=0, dim_size=num_nodes, reduce='mean')
        psl_std = (m1_var + m2_var + eps).sqrt()
        L_consist = ((1.0 - sing_prob.squeeze(1)[mask].clamp(0, 1)) * psl_std[mask]).mean()
    else:
        L_consist = torch.tensor(0.0, device=device)

    # NaN 保护: 将所有 NaN 替换为 0
    result = {
        'phys_sparse': L_sparse,
        'phys_smooth': L_smooth,
        'phys_consist': L_consist,
    }
    for k in result:
        if torch.isnan(result[k]) or torch.isinf(result[k]):
            result[k] = torch.tensor(0.0, device=device)
    return result

# ===================== 验证指标（同 v3） =====================
@torch.no_grad()
def compute_metrics(node_pred, edge_pred, node_target, edge_target):
    from sklearn.metrics import roc_auc_score
    np_, nt = node_pred.cpu().numpy().ravel(), node_target.cpu().numpy().ravel()
    ep, et = edge_pred.cpu().numpy(), edge_target.cpu().numpy()
    def sauc(yt, yp):
        if yt.sum()==0 or (yt==1).all(): return 0.5
        return roc_auc_score(yt, yp)
    st = np.percentile(nt,99) if nt.max()>0 else 0.5
    sb = (nt>st).astype(int)
    r = {'sing_auc': sauc(sb,np_),
         'sing_dice': 2*((np_>0.5)*sb).sum()/((np_>0.5).sum()+sb.sum()+1e-8)}
    for lb, p, t in [('m1',ep[:,0],et[:,0]),('m2',ep[:,1],et[:,1])]:
        th = np.percentile(t,84) if t.max()>0 else 0.5
        tb = (t>th).astype(int); pb = (p>0.5).astype(int)
        r[f'{lb}_auc']=sauc(tb,p)
        r[f'{lb}_dice']=2*(pb*tb).sum()/(pb.sum()+tb.sum()+1e-8)
    ea_t = np.concatenate([(et[:,0]>np.percentile(et[:,0],84)).astype(int),
                            (et[:,1]>np.percentile(et[:,1],84)).astype(int)])
    ea_p = np.concatenate([ep[:,0], ep[:,1]])
    r['edge_auc']=sauc(ea_t,ea_p)
    r['edge_dice']=2*((ea_p>0.5)*ea_t).sum()/((ea_p>0.5).sum()+ea_t.sum()+1e-8)
    return r

# ===================== 主训练 =====================
def train_physics(data_path, gnn_type='gat', epochs=300,
                  lr_g=2e-4, lr_d=5e-5, batch_size=2,
                  hidden_dim_g=128, hidden_dim_d=64,
                  lambda_recon=10.0, lambda_adv=0.5, lambda_r1=1.0,
                  # ★ 物理约束权重
                  lambda_sparse=0.1, lambda_smooth=1.0, lambda_consist=0.5,
                  label_smooth=0.9, d_noise=0.05, d_updates=1,
                  g_clip_norm=5.0, warmup_epochs=5,
                  device='cuda', save_root='./trained_model/v5_physics', seed=42):
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    mse = nn.MSELoss(); l1 = nn.L1Loss()

    label = f'v5_{gnn_type}_physics_S{lambda_sparse}_M{lambda_smooth}_C{lambda_consist}'
    cur_time = datetime.now().strftime("%Y_%m_%d_%H%M")
    save_dir = os.path.join(save_root, cur_time, label)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Experiment: {label}\nSave dir:  {save_dir}")

    dataset = torch.load(data_path, weights_only=False)
    indices = list(range(len(dataset))); random.shuffle(indices)
    n_train = int(len(dataset)*0.9)
    train_loader = DataLoader([dataset[i] for i in indices[:n_train]],
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([dataset[i] for i in indices[n_train:]],
                             batch_size=batch_size, shuffle=False)

    G = GeneratorGNN(input_dim=12, hidden_dim=hidden_dim_g, hidden_dim2=64,
                      gnn_type=gnn_type, dropout=0.1).to(device); G.apply(init_weights)
    d_gnn = gnn_type if gnn_type in ('gat','gcn','sage') else 'gcn'
    D = DiscriminatorGNN(input_dim=12, node_label_dim=1, hidden_dim=hidden_dim_d,
                          gnn_type=d_gnn, dropout=0.3).to(device); D.apply(init_weights)
    print(f"G: {sum(p.numel() for p in G.parameters()):,} | "
          f"D: {sum(p.numel() for p in D.parameters()):,} | "
          f"λ_sparse={lambda_sparse} λ_smooth={lambda_smooth} λ_consist={lambda_consist}")

    opt_g = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5,0.999))
    opt_d = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5,0.999))
    sg = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs-warmup_epochs, eta_min=lr_g*0.01)
    sd = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs-warmup_epochs, eta_min=lr_d*0.01)

    log_path = os.path.join(save_dir, 'metrics.csv')
    with open(log_path,'w') as f:
        f.write("epoch,d_loss,g_loss,g_recon,g_adv,gp_r1,"
                "phys_sparse,phys_smooth,phys_consist,"
                "sing_auc,sing_dice,m1_auc,m1_dice,m2_auc,m2_dice,edge_auc,edge_dice,lr_g,lr_d\n")
    with open(os.path.join(save_dir,'hyperparams.txt'),'w') as f:
        for k,v in dict(gnn_type=gnn_type, lambda_sparse=lambda_sparse,
                         lambda_smooth=lambda_smooth, lambda_consist=lambda_consist,
                         lambda_recon=lambda_recon, lambda_adv=lambda_adv).items():
            f.write(f"{k}: {v}\n")

    best_edge, best_sing = 0.0, 0.0
    for epoch in range(epochs):
        G.train(); D.train()
        ls = {'d':0,'g':0,'recon':0,'adv':0,'r1':0,'sp':0,'sm':0,'cn':0}
        pbar = tqdm(train_loader, desc=f"[{gnn_type}+Phys] E{epoch+1}/{epochs}")
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
                d_loss = F.relu(rt-rs).mean() + F.relu(fs+rt).mean()
                if lambda_r1>0:
                    r1 = r1_penalty(D, batch, rn2, re2, lambda_r1)
                    d_loss = d_loss + r1; ls['r1'] += r1.item()
                d_loss.backward(); opt_d.step()
            ls['d'] += d_loss.item()

            # (2) G
            opt_g.zero_grad()
            fn, fe = G(batch)

            # 重建损失
            g_recon = l1(fn, rn) + l1(fe, re)

            # 对抗损失
            fn2, fe2 = add_disc_noise(fn, fe, d_noise)
            fs = D(batch, fn2, fe2)
            g_adv = -fs.mean()

            # ★ 物理约束 (仅当有权重时计算，避免不必要的 NaN 风险)
            use_physics = (lambda_sparse > 0 or lambda_smooth > 0 or lambda_consist > 0)
            if use_physics:
                phys = physics_losses(fn, fe, batch.edge_index, batch.batch)
                g_phys = (lambda_sparse * phys['phys_sparse'] +
                          lambda_smooth * phys['phys_smooth'] +
                          lambda_consist * phys['phys_consist'])
                ls['sp'] += phys['phys_sparse'].item()
                ls['sm'] += phys['phys_smooth'].item()
                ls['cn'] += phys['phys_consist'].item()
            else:
                g_phys = torch.tensor(0.0, device=device)

            g_loss = lambda_recon*g_recon + lambda_adv*g_adv + g_phys

            # NaN 检测: 跳过坏梯度
            if torch.isnan(g_loss) or torch.isinf(g_loss):
                opt_g.zero_grad()
                continue

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), g_clip_norm)
            opt_g.step()

            ls['g'] += g_loss.item(); ls['recon'] += g_recon.item(); ls['adv'] += g_adv.item()

            sp_val = phys['phys_sparse'].item() if use_physics else 0.0
            sm_val = phys['phys_smooth'].item() if use_physics else 0.0
            pbar.set_postfix(D=f'{d_loss.item():.3f}', G=f'{g_loss.item():.3f}',
                           recon=f'{g_recon.item():.3f}', adv=f'{g_adv.item():.3f}',
                           sp=f'{sp_val:.3f}', sm=f'{sm_val:.3f}')

        # Val
        G.eval()
        all_m = []
        for batch in val_loader:
            batch = batch.to(device)
            with torch.no_grad(): npred, epred = G(batch)
            all_m.append(compute_metrics(npred, epred, batch.y_node, batch.y_edge))
        avg = {k: np.mean([m[k] for m in all_m]) for k in all_m[0]}

        nb = len(train_loader)
        with open(log_path,'a') as f:
            f.write(f"{epoch+1},{ls['d']/nb:.6f},{ls['g']/nb:.6f},{ls['recon']/nb:.6f},"
                    f"{ls['adv']/nb:.6f},{ls['r1']/nb:.6f},"
                    f"{ls['sp']/nb:.6f},{ls['sm']/nb:.6f},{ls['cn']/nb:.6f},"
                    f"{avg['sing_auc']:.4f},{avg['sing_dice']:.4f},"
                    f"{avg['m1_auc']:.4f},{avg['m1_dice']:.4f},"
                    f"{avg['m2_auc']:.4f},{avg['m2_dice']:.4f},"
                    f"{avg['edge_auc']:.4f},{avg['edge_dice']:.4f},"
                    f"{opt_g.param_groups[0]['lr']:.2e},{opt_d.param_groups[0]['lr']:.2e}\n")

        if avg['edge_dice']>best_edge:
            best_edge = avg['edge_dice']
            torch.save({'G':G.state_dict(),'D':D.state_dict()},
                       os.path.join(save_dir,'best_edge.pth'))
        if avg['sing_dice']>best_sing:
            best_sing = avg['sing_dice']
            torch.save({'G':G.state_dict()}, os.path.join(save_dir,'best_sing.pth'))
        if (epoch+1)%100==0:
            torch.save({'G':G.state_dict(),'D':D.state_dict()},
                       os.path.join(save_dir,f'ckpt_{epoch+1}.pth'))

        print(f"[{gnn_type}+Phys] E{epoch+1}: D={ls['d']/nb:.3f} G={ls['g']/nb:.3f} "
              f"recon={ls['recon']/nb:.3f} adv={ls['adv']/nb:.3f} | "
              f"phys: sp={ls['sp']/nb:.4f} sm={ls['sm']/nb:.4f} cn={ls['cn']/nb:.4f} | "
              f"Sing AUC={avg['sing_auc']:.3f} D={avg['sing_dice']:.3f} | "
              f"Edge D={avg['edge_dice']:.3f}")
        if epoch>=warmup_epochs: sg.step(); sd.step()

    print(f"\n[{gnn_type}+Phys] Done! Best Sing Dice={best_sing:.4f} Edge Dice={best_edge:.4f}")
    return G, D, save_dir, best_sing, best_edge

if __name__ == '__main__':
    DATA = "datasets/03_graph/merged_25cases_continuous_augmented_x7.pt"
    if not os.path.exists(DATA): print(f"Missing: {DATA}"); sys.exit(1)
    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 使用最佳 GNN 架构 (GAT, 来自 v3 结果)
    # 3 组物理约束权重消融实验
    configs = [
        {'label': 'A: Baseline (no physics)',  'l_sp':0.0, 'l_sm':0.0, 'l_cn':0.0},
        {'label': 'B: +Sparsity+Smoothness',   'l_sp':0.1, 'l_sm':1.0, 'l_cn':0.0},
        {'label': 'C: +Sparsity+Smooth+Consist','l_sp':0.1, 'l_sm':1.0, 'l_cn':0.5},
    ]

    results = {}
    for cfg in configs:
        print(f"\n{'='*60}\nPhysics: {cfg['label']}\n{'='*60}")
        _, _, _, best_s, best_e = train_physics(
            DATA, gnn_type='gat', epochs=300,
            lambda_sparse=cfg['l_sp'], lambda_smooth=cfg['l_sm'],
            lambda_consist=cfg['l_cn'], device=DEV)
        results[cfg['label']] = {'sing_dice': best_s, 'edge_dice': best_e}

    print(f"\n{'='*60}\nPHYSICS ABLATION RESULTS\n{'='*60}")
    for label, r in results.items():
        print(f"  {label:>35s}: Sing Dice={r['sing_dice']:.4f}  Edge Dice={r['edge_dice']:.4f}")
