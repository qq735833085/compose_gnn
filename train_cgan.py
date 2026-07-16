# train_cgan.py — Conditional GAN 训练脚本
# =============================================================================
# 训练流程:
#   1. Generator 生成 sing_prob + psl_prob
#   2. Discriminator 判断 (slab, sing_prob) 真伪
#   3. 交替更新 G 和 D
# =============================================================================

import os, sys, torch, random, numpy as np
import torch.optim as optim
from torch_geometric.loader import DataLoader
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.cgan import FieldGenerator, FieldDiscriminator, GANLoss
from loss import MultiTaskStressLoss


def compute_metrics(node_pred, node_true, edge_pred, edge_true):
    """计算 AUC 和 Dice"""
    def safe_auc(y_true, y_pred):
        if y_true.sum() == 0 or (y_true == 1).all(): return 0.5
        return roc_auc_score(y_true, y_pred)

    # 阈值化（对连续概率场取 >0.5 进行二值评估）
    sing_auc = safe_auc((node_true > 0.5).float().cpu().numpy().ravel(),
                         node_pred.cpu().numpy().ravel())
    psl_auc = safe_auc((edge_true > 0.5).float().cpu().numpy().ravel(),
                        edge_pred.cpu().numpy().ravel())

    def dice(y_t, y_p, th=0.5):
        pb = (y_p > th).astype(int); tb = (y_t > th).astype(int)
        inter = (pb * tb).sum()
        return 2 * inter / (pb.sum() + tb.sum() + 1e-8)

    sing_dice = dice(node_true.cpu().numpy().ravel(), node_pred.cpu().numpy().ravel(), 0.3)
    psl_dice = dice(edge_true.cpu().numpy().ravel(), edge_pred.cpu().numpy().ravel(), 0.3)
    return sing_auc, psl_auc, sing_dice, psl_dice


def train_cgan(data_path, epochs=200, lr=2e-4, batch_size=1, device='cpu',
               save_root='./trained_model', seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    dataset = torch.load(data_path, weights_only=False)
    n_train = int(len(dataset) * 0.9)
    train_loader = DataLoader(dataset[:n_train], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[n_train:], batch_size=1, shuffle=False)

    # 模型
    G = FieldGenerator(input_dim=12).to(device)
    D = FieldDiscriminator(input_dim=13).to(device)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    gan_loss = GANLoss(adv_weight=0.1, focal_gamma=1.0)
    val_criterion = MultiTaskStressLoss()

    # 保存路径
    cur_time = datetime.now().strftime("%Y_%m_%d_%H%M")
    save_dir = os.path.join(save_root, cur_time, 'cGAN')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save: {save_dir}")

    best_psl_auc = 0.0

    for epoch in range(epochs):
        G.train(); D.train()
        g_loss_total = 0.0; d_loss_total = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            batch = batch.to(device)
            B = batch.num_graphs

            # ---- 1. 训练 Discriminator ----
            # Real
            real_sing = batch.y_node
            d_real = D(batch.x, real_sing, batch.edge_index, batch.batch)

            # Fake
            with torch.no_grad():
                fake_sing, fake_psl = G(batch.x, batch.edge_index)
            d_fake = D(batch.x, fake_sing.detach(), batch.edge_index, batch.batch)

            d_loss, d_logs = gan_loss.discriminator_loss(d_real, d_fake)
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # ---- 2. 训练 Generator ----
            fake_sing, fake_psl = G(batch.x, batch.edge_index)
            d_fake = D(batch.x, fake_sing, batch.edge_index, batch.batch)

            g_loss, g_logs = gan_loss.generator_loss(
                fake_sing, batch.y_node, fake_psl, batch.y_edge, d_fake)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
            loop.set_postfix(G=f'{g_loss.item():.3f}', D=f'{d_loss.item():.3f}',
                            g_adv=f'{g_logs["g_adv"]:.3f}')

        # ---- 验证 ----
        G.eval()
        val_sing_aucs, val_psl_aucs, val_sing_dices, val_psl_dices = [], [], [], []
        val_loss_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                sing_pred, psl_pred = G(batch.x, batch.edge_index)
                loss, _ = val_criterion(sing_pred, batch.y_node,
                                        psl_pred, batch.y_edge[:, 0:1])
                val_loss_total += loss.item()

                sa, pa, sd, pd = compute_metrics(
                    sing_pred, batch.y_node, psl_pred, batch.y_edge[:, 0:1])
                val_sing_aucs.append(sa); val_psl_aucs.append(pa)
                val_sing_dices.append(sd); val_psl_dices.append(pd)

        avg_sing_auc = np.mean(val_sing_aucs)
        avg_psl_auc = np.mean(val_psl_aucs)
        avg_sing_dice = np.mean(val_sing_dices)
        avg_psl_dice = np.mean(val_psl_dices)

        # 日志
        log_path = os.path.join(save_dir, 'metrics.csv')
        if epoch == 0:
            with open(log_path, 'w') as f:
                f.write("epoch,g_loss,d_loss,val_loss,sing_auc,psl_auc,sing_dice,psl_dice\n")
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{g_loss_total/len(train_loader):.4f},"
                    f"{d_loss_total/len(train_loader):.4f},{val_loss_total/len(val_loader):.4f},"
                    f"{avg_sing_auc:.4f},{avg_psl_auc:.4f},{avg_sing_dice:.4f},{avg_psl_dice:.4f}\n")

        if avg_psl_auc > best_psl_auc:
            best_psl_auc = avg_psl_auc
            torch.save({'G': G.state_dict(), 'D': D.state_dict()},
                       os.path.join(save_dir, 'best_model.pth'))
            print(f"  New best PSL AUC={best_psl_auc:.4f}")

        print(f"Epoch {epoch+1}: G={g_loss_total/len(train_loader):.3f} "
              f"D={d_loss_total/len(train_loader):.3f} "
              f"Val={val_loss_total/len(val_loader):.3f} | "
              f"Sing AUC={avg_sing_auc:.3f} D={avg_sing_dice:.3f} | "
              f"PSL AUC={avg_psl_auc:.3f} D={avg_psl_dice:.3f}")

    print(f"Done. Best PSL AUC={best_psl_auc:.4f}")
    return G, D


if __name__ == '__main__':
    DATA_PATH = r"D:\composite_0602\datasets\03_graph\merged_25cases_prob.pt"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_PATH}")
    print()

    train_cgan(DATA_PATH, epochs=200, lr=2e-4, batch_size=1, device=DEVICE)
