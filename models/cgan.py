# models/cgan.py — Conditional GAN for singularity + PSL probability field generation
# =============================================================================
# Generator: GNN → sing_prob [N,1] + psl_prob [E,1]  (连续概率场)
# Discriminator: GNN + global pool → real/fake
# =============================================================================

import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool


# ======================== Generator ========================
class FieldGenerator(nn.Module):
    """生成完整的奇异点 + PSL 连续概率场"""
    def __init__(self, input_dim=12, hidden_dim=128, out_dim=64):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.res_proj = nn.Linear(hidden_dim, out_dim)

        # 节点解码器 → [N, 1] 奇异点概率场
        self.node_head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2), nn.LeakyReLU(0.2),
            nn.Linear(out_dim // 2, 1), nn.Sigmoid())

        # 边解码器 → [E, 1] PSL 概率场
        self.edge_head = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim // 2), nn.LeakyReLU(0.2),
            nn.Linear(out_dim // 2, 1), nn.Sigmoid())

    def forward(self, x, edge_index):
        h0 = self.embed(x)
        h1 = F.leaky_relu(self.conv1(h0, edge_index), 0.2)
        h2 = F.leaky_relu(self.conv2(h1, edge_index), 0.2)
        h = h2 + self.res_proj(h0)

        sing_prob = self.node_head(h)             # [N, 1]
        row, col = edge_index[0], edge_index[1]
        edge_feat = torch.cat([h[row], h[col]], dim=1)
        psl_prob = self.edge_head(edge_feat)      # [E, 1]
        return sing_prob, psl_prob


# ======================== Discriminator ========================
class FieldDiscriminator(nn.Module):
    """
    图级判别器：判断 (slab, probability_field) 对是否真实。
    输入节点特征附加生成的概率场 → GNN → global pool → real/fake
    """
    def __init__(self, input_dim=13, hidden_dim=128):  # 12 + 1(sing_prob)
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1))  # no sigmoid (handled by BCEWithLogits)

    def forward(self, x, sing_prob, edge_index, batch=None):
        """batch: 批次索引 (多图 batch 训练时使用)"""
        # 拼接特征
        h = torch.cat([x, sing_prob], dim=1)  # [N, 13]
        h = F.leaky_relu(self.conv1(h, edge_index), 0.2)
        h = F.leaky_relu(self.conv2(h, edge_index), 0.2)
        h = F.leaky_relu(self.conv3(h, edge_index), 0.2)

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        g = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        return self.classifier(g)  # [B, 1] logits


# ======================== Loss helpers ========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = pred * target + (1 - pred) * (1 - target)
        w = (self.alpha * target + (1 - self.alpha) * (1 - target)) * (1 - pt) ** self.gamma
        return w.mean() * bce.mean()  # scaled


class GANLoss(nn.Module):
    """对抗训练损失包装器"""
    def __init__(self, adv_weight=0.1, focal_gamma=1.0):
        super().__init__()
        self.adv_weight = adv_weight
        self.recon_sing = FocalLoss(gamma=focal_gamma, alpha=0.8)
        self.recon_psl = FocalLoss(gamma=focal_gamma, alpha=0.6)
        self.bce = nn.BCEWithLogitsLoss()

    def generator_loss(self, sing_pred, sing_true, psl_pred, psl_true, disc_fake):
        """G_loss = recon + adv * BCE(D(fake), 1)"""
        l_sing = self.recon_sing(sing_pred, sing_true)
        l_psl = self.recon_psl(psl_pred, psl_true)
        l_adv = self.bce(disc_fake, torch.ones_like(disc_fake))
        return l_sing + l_psl + self.adv_weight * l_adv, {
            'g_sing': l_sing.item(), 'g_psl': l_psl.item(), 'g_adv': l_adv.item()}

    def discriminator_loss(self, disc_real, disc_fake):
        """D_loss = BCE(real,1) + BCE(fake,0)"""
        l_real = self.bce(disc_real, torch.ones_like(disc_real))
        l_fake = self.bce(disc_fake, torch.zeros_like(disc_fake))
        return l_real + l_fake, {'d_real': l_real.item(), 'd_fake': l_fake.item()}
