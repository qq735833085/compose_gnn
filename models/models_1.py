# test_models.py (重构后)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv, GATConv

# ======================== 1. GraphSAGE 模型 ========================
class PureSAGEModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=64):
        super(PureSAGEModel, self).__init__()
        
        # ---------- 编码器 ----------
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.conv1 = SAGEConv(hidden_dim1, hidden_dim1)
        self.conv2 = SAGEConv(hidden_dim1, hidden_dim2)
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)
        
        # ---------- 解码器 ----------
        # 节点解码器：输入节点特征 (hidden_dim2)，输出奇异点概率 (1)
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )
        # 边解码器：输入两端节点特征拼接 (hidden_dim2 * 2)，输出应力线概率 (1)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )
    
    def encoder(self, x, edge_index):
        """返回编码后的节点特征 h_final (N, hidden_dim2)"""
        h0 = self.node_embedding(x)
        h1 = F.leaky_relu(self.conv1(h0, edge_index), 0.2)
        h2 = F.leaky_relu(self.conv2(h1, edge_index), 0.2)
        h_final = h2 + self.residual_proj(h0)
        return h_final
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h_final = self.encoder(x, edge_index)
        
        node_preds = self.node_decoder(h_final)                 # (N, 1)
        row, col = edge_index[0], edge_index[1]
        edge_feats = torch.cat([h_final[row], h_final[col]], dim=1)  # (E, 2*hidden_dim2)
        edge_preds = self.edge_decoder(edge_feats)              # (E, 1)
        return node_preds, edge_preds


# ======================== 2. GATv2 模型 ========================
class PureDynamicGATModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=64, heads=8):
        super(PureDynamicGATModel, self).__init__()
        
        # 编码器
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.conv1 = GATv2Conv(hidden_dim1, hidden_dim1 // heads, heads=heads, concat=True, dropout=0.2)
        self.conv2 = GATv2Conv(hidden_dim1, hidden_dim2, heads=1, dropout=0.2)
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)
        
        # 解码器
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )
    
    def encoder(self, x, edge_index):
        h0 = self.node_embedding(x)
        h1 = F.leaky_relu(self.conv1(h0, edge_index), 0.2)
        h2 = F.leaky_relu(self.conv2(h1, edge_index), 0.2)
        h_final = h2 + self.residual_proj(h0)
        return h_final
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h_final = self.encoder(x, edge_index)
        
        node_preds = self.node_decoder(h_final)
        row, col = edge_index[0], edge_index[1]
        edge_feats = torch.cat([h_final[row], h_final[col]], dim=1)
        edge_preds = self.edge_decoder(edge_feats)
        return node_preds, edge_preds


# ======================== 3. GCN 模型 ========================
class PureGCNModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=64):
        super(PureGCNModel, self).__init__()
        
        # 编码器
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.conv1 = GCNConv(hidden_dim1, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)
        
        # 解码器
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )
    
    def encoder(self, x, edge_index):
        h0 = self.node_embedding(x)
        h1 = F.leaky_relu(self.conv1(h0, edge_index), 0.2)
        h2 = F.leaky_relu(self.conv2(h1, edge_index), 0.2)
        h_final = h2 + self.residual_proj(h0)
        return h_final
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h_final = self.encoder(x, edge_index)
        
        node_preds = self.node_decoder(h_final)
        row, col = edge_index[0], edge_index[1]
        edge_feats = torch.cat([h_final[row], h_final[col]], dim=1)
        edge_preds = self.edge_decoder(edge_feats)
        return node_preds, edge_preds


# ======================== 4. 经典 GAT 模型 ========================
class PureClassicGATModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=64, heads=8):
        super(PureClassicGATModel, self).__init__()
        
        # 编码器
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.conv1 = GATConv(hidden_dim1, hidden_dim1 // heads, heads=heads, concat=True, dropout=0.2)
        self.conv2 = GATConv(hidden_dim1, hidden_dim2, heads=1, dropout=0.2)
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)
        
        # 解码器
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )
    
    def encoder(self, x, edge_index):
        h0 = self.node_embedding(x)
        h1 = F.leaky_relu(self.conv1(h0, edge_index), 0.2)
        h2 = F.leaky_relu(self.conv2(h1, edge_index), 0.2)
        h_final = h2 + self.residual_proj(h0)
        return h_final
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h_final = self.encoder(x, edge_index)
        
        node_preds = self.node_decoder(h_final)
        row, col = edge_index[0], edge_index[1]
        edge_feats = torch.cat([h_final[row], h_final[col]], dim=1)
        edge_preds = self.edge_decoder(edge_feats)
        return node_preds, edge_preds