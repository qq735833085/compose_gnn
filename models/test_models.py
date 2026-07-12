# test_models.py — 四种基线 GNN 模型 (edge_decoder 统一输出 [E, 2])

import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv, GATConv


def _make_edge_decoder(hidden_dim2):
    """边双任务解码器: 拼接两端特征 → [is_psl_1, is_psl_2]"""
    return nn.Sequential(
        nn.Linear(hidden_dim2 * 2, hidden_dim2 // 2),
        nn.LeakyReLU(0.2), nn.Dropout(0.2),
        nn.Linear(hidden_dim2 // 2, 2),
        nn.Sigmoid()
    )


def _make_node_decoder(hidden_dim2):
    """节点解码器: → is_singularity"""
    return nn.Sequential(
        nn.Linear(hidden_dim2, hidden_dim2 // 2),
        nn.LeakyReLU(0.2), nn.Dropout(0.2),
        nn.Linear(hidden_dim2 // 2, 1),
        nn.Sigmoid()
    )


def _make_embedding(input_dim, hidden_dim1):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim1),
        nn.LeakyReLU(0.2), nn.Dropout(0.2)
    )


# ======================== 1. Pure GraphSAGE ========================
class PureSAGEModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=64):
        super().__init__()
        self.node_embedding = _make_embedding(input_dim, hidden_dim1)
        self.conv1 = SAGEConv(hidden_dim1, hidden_dim1)
        self.conv2 = SAGEConv(hidden_dim1, hidden_dim2)
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)
        self.node_decoder = _make_node_decoder(hidden_dim2)
        self.edge_decoder = _make_edge_decoder(hidden_dim2)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        h0 = self.node_embedding(x)
        h1 = F.leaky_relu(self.conv1(h0, ei), 0.2)
        h2 = F.leaky_relu(self.conv2(h1, ei), 0.2)
        h_final = h2 + self.residual_proj(h0)
        node_preds = self.node_decoder(h_final)
        row, col = ei[0], ei[1]
        edge_preds = self.edge_decoder(torch.cat([h_final[row], h_final[col]], dim=1))
        return node_preds, edge_preds


# ======================== 2. Pure GATv2 ========================
class PureDynamicGATModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=64, heads=8):
        super().__init__()
        self.node_embedding = _make_embedding(input_dim, hidden_dim1)
        self.conv1 = GATv2Conv(hidden_dim1, hidden_dim1 // heads, heads=heads, concat=True, dropout=0.2)
        self.conv2 = GATv2Conv(hidden_dim1, hidden_dim2, heads=1, dropout=0.2)
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)
        self.node_decoder = _make_node_decoder(hidden_dim2)
        self.edge_decoder = _make_edge_decoder(hidden_dim2)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        h0 = self.node_embedding(x)
        h1 = F.leaky_relu(self.conv1(h0, ei), 0.2)
        h2 = F.leaky_relu(self.conv2(h1, ei), 0.2)
        h_final = h2 + self.residual_proj(h0)
        node_preds = self.node_decoder(h_final)
        row, col = ei[0], ei[1]
        edge_preds = self.edge_decoder(torch.cat([h_final[row], h_final[col]], dim=1))
        return node_preds, edge_preds


# ======================== 3. Pure GCN ========================
class PureGCNModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=64):
        super().__init__()
        self.node_embedding = _make_embedding(input_dim, hidden_dim1)
        self.conv1 = GCNConv(hidden_dim1, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)
        self.node_decoder = _make_node_decoder(hidden_dim2)
        self.edge_decoder = _make_edge_decoder(hidden_dim2)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        h0 = self.node_embedding(x)
        h1 = F.leaky_relu(self.conv1(h0, ei), 0.2)
        h2 = F.leaky_relu(self.conv2(h1, ei), 0.2)
        h_final = h2 + self.residual_proj(h0)
        node_preds = self.node_decoder(h_final)
        row, col = ei[0], ei[1]
        edge_preds = self.edge_decoder(torch.cat([h_final[row], h_final[col]], dim=1))
        return node_preds, edge_preds


# ======================== 4. Classic GAT ========================
class PureClassicGATModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=64, heads=8):
        super().__init__()
        self.node_embedding = _make_embedding(input_dim, hidden_dim1)
        self.conv1 = GATConv(hidden_dim1, hidden_dim1 // heads, heads=heads, concat=True, dropout=0.2)
        self.conv2 = GATConv(hidden_dim1, hidden_dim2, heads=1, dropout=0.2)
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)
        self.node_decoder = _make_node_decoder(hidden_dim2)
        self.edge_decoder = _make_edge_decoder(hidden_dim2)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        h0 = self.node_embedding(x)
        h1 = F.leaky_relu(self.conv1(h0, ei), 0.2)
        h2 = F.leaky_relu(self.conv2(h1, ei), 0.2)
        h_final = h2 + self.residual_proj(h0)
        node_preds = self.node_decoder(h_final)
        row, col = ei[0], ei[1]
        edge_preds = self.edge_decoder(torch.cat([h_final[row], h_final[col]], dim=1))
        return node_preds, edge_preds
