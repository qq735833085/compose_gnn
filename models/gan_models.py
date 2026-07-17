# models/gan_models.py
# =============================================================================
# GAN 架构：Generator（预测概率分布）+ Discriminator（判别真假分布）
#
# Generator:  GNN 编码器 → 节点概率 + 边概率
#   - 输入：应力场特征 x [N,12], edge_index [2,2E]
#   - 输出：node_prob [N,1] (奇异点概率，节点PSL已删除)
#           edge_prob [2E,1] (PSL边概率)
#
# Discriminator: GNN 图分类器
#   - 输入：应力场 + 节点预测 + 边预测
#   - 输出：score [1] (真实 vs 生成的置信度)
#
# 关键设计：
#   1. Spectral Norm 稳定判别器训练
#   2. 边缘预测作为判别器消息传递的注意力权重
#   3. 全局池化捕获空间分布模式
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.utils import dropout_edge


# ============================= Generator =============================

class GeneratorGNN(nn.Module):
    """
    图生成器：从应力场预测连续概率分布。

    架构：
      GNN Encoder (2层) → 节点嵌入
        ├── Node Head: MLP → node_prob [N, 2]  (sing, psl_node)
        └── Edge Head: MLP(src⊕dst) → edge_prob [2E, 1]
    """

    def __init__(self, input_dim=12, hidden_dim=128, hidden_dim2=64,
                 gnn_type='gat', dropout=0.1):
        super().__init__()
        self.gnn_type = gnn_type
        self.dropout_rate = dropout

        # ---- Encoder ----
        if gnn_type == 'gat':
            self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim2, heads=4, concat=False, dropout=dropout)
        elif gnn_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim2)
        elif gnn_type == 'sage':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim2)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)

        # ---- Node Head: embedding → singularity probability ----
        # 节点 PSL 已删除，仅输出奇异点概率
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2 // 2, 1),   # [sing_prob]
            nn.Sigmoid()                       # 概率输出 [0,1]
        )

        # ---- Edge Head: concat(src_emb, dst_emb) → [m1_psl, m2_psl] ----
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, hidden_dim2 // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2 // 2, 2),   # [m1_psl_prob, m2_psl_prob]
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        # Encoder
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)

        # Node predictions
        node_prob = self.node_head(h)  # [N, 2]

        # Edge predictions: 双向边都预测
        src, dst = edge_index[0], edge_index[1]
        edge_feat = torch.cat([h[src], h[dst]], dim=-1)  # [2E, hidden_dim2*2]
        edge_prob = self.edge_head(edge_feat)              # [2E, 1]

        return node_prob, edge_prob


# ============================= Discriminator =============================

class DiscriminatorGNN(nn.Module):
    """
    图判别器：判断 (应力场, 概率分布) 是真实还是生成的。

    输入节点特征 = 应力场特征 ⊕ 节点概率
    边权重 = 边概率（用于消息传递中的注意力调制）

    架构：
      GNN (2层) → 全局池化 (mean+max) → MLP → score
    """

    def __init__(self, input_dim=12, node_label_dim=1, hidden_dim=128,
                 gnn_type='gat', dropout=0.2):
        super().__init__()
        self.node_label_dim = node_label_dim
        total_input_dim = input_dim + node_label_dim  # 12 + 2 = 14

        # ---- GNN Encoder ----
        if gnn_type == 'gat':
            self.conv1 = GATv2Conv(total_input_dim, hidden_dim, heads=4, concat=False,
                                   edge_dim=2, dropout=dropout)
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False,
                                   edge_dim=2, dropout=dropout)
        elif gnn_type == 'gcn':
            self.conv1 = GCNConv(total_input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # ---- 全局池化 + 分类头 ----
        # mean_pool + max_pool → concat → MLP → score
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            # 无 Sigmoid — 使用 BCEWithLogitsLoss（数值更稳定）
        )

        self.dropout_rate = dropout

    def forward(self, data, node_labels, edge_labels):
        """
        Args:
            data: PyG Data with x [N,12], edge_index [2,2E], batch [N]
            node_labels: [N, 2] — 真实或生成的节点概率
            edge_labels: [2E, 1] — 真实或生成的边概率
        Returns:
            score: [B, 1] — 真实度评分 (1=真实, 0=生成)
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        # 拼接应力场特征 + 节点标签
        h = torch.cat([x, node_labels], dim=-1)  # [N, 14]

        # GNN with edge features as attention weights
        if hasattr(self.conv1, 'edge_dim') and self.conv1.edge_dim is not None:
            edge_attr = edge_labels  # [2E, 1]
            h = self.conv1(h, edge_index, edge_attr)
        else:
            h = self.conv1(h, edge_index)

        h = self.bn1(h)
        h = F.leaky_relu(h, 0.2)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        if hasattr(self.conv2, 'edge_dim') and self.conv2.edge_dim is not None:
            h = self.conv2(h, edge_index, edge_attr)
        else:
            h = self.conv2(h, edge_index)

        h = self.bn2(h)
        h = F.leaky_relu(h, 0.2)

        # 全局池化
        if batch is None:
            # 单图
            h_mean = h.mean(dim=0, keepdim=True)
            h_max = h.max(dim=0, keepdim=True)[0]
        else:
            h_mean = global_mean_pool(h, batch)
            h_max = global_max_pool(h, batch)

        h_pool = torch.cat([h_mean, h_max], dim=-1)

        # 分类
        score = self.classifier(h_pool)  # [B, 1]
        return score


# ============================= 权重初始化 =============================

def init_weights(m):
    """He 初始化（对 GNN 和 MLP 均适用）"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (GCNConv, GATv2Conv, SAGEConv)):
        if hasattr(m, 'lin'):
            nn.init.kaiming_normal_(m.lin.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
