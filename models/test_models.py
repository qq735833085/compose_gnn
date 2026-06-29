# test_models.py

# 导入 PyTorch 核心深度学习计算库
import torch
# 导入神经网络底层基础层模块（全连接层、Sequential 容器等）
import torch.nn as nn
# 导入功能性神经网络函数库（包含 LeakyReLU、Dropout、Sigmoid 等不含权重的函数）
import torch.nn.functional as F

# 从 PyTorch Geometric 图神经网络库中导入四大核心算子
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv, GATConv

# =========================================================================
# 1. 第一版模型：纯 GraphSAGE 基准模型 (均权/拼接邻域聚合)
# =========================================================================
class PureSAGEModel(nn.Module):
    """
    基于 GraphSAGE 算子的同构网格图多任务应力流预测网络。
    特点：对邻居特征进行平权聚合（求平均/求和），适合大面积平滑的力学场传导。
    """
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=64):
        # 调用父类 nn.Module 的构造函数，初始化 network 底座
        super(PureSAGEModel, self).__init__()
        
        # 【特征嵌入层】：用前馈感知机将输入的 13 维原始受力特征直接提升到 128 维宽通道空间
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # 【图网络编码层】：连续叠加两层标准的 GraphSAGE 算子
        self.conv1 = SAGEConv(hidden_dim1, hidden_dim1)
        self.conv2 = SAGEConv(hidden_dim1, hidden_dim2)
        
        # 【残差快捷通道】：因为初始特征 h0 (128维) 与图卷积输出 h2 (64维) 维度不同，用线性层投影对齐
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)
        
        # 【节点多任务解码器】：将融合后的 64 维特征映射到 2 维，并行输出 [奇异点概率, 在线概率]
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),  # 64 维 -> 32 维
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),             # 32 维 -> 2 维
            nn.Sigmoid()                                # 用 Sigmoid 约束输出在 [0, 1] 概率场区间
        )
        
        # 【边单任务解码器】：成对拼接两端端点特征（64*2=128维），映射到 1 维输出 [close_to_psl概率]
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, hidden_dim2 // 2), # 128 维 -> 32 维
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),              # 32 维 -> 1 维
            nn.Sigmoid()                                 # 用 Sigmoid 约束在 [0, 1] 概率区间
        )

    def forward(self, data):
        """ 定义向前传播推理数据流 """
        x, edge_index = data.x, data.edge_index
        
        # 步骤一：通过初始特征嵌入层提升特征维度
        h0 = self.node_embedding(x)
        
        # 步骤二：经历两层 GraphSAGE 算子进行均权信息交互与非线性激活
        h1 = F.leaky_relu(self.conv1(h0, edge_index), 0.2)
        h2 = F.leaky_relu(self.conv2(h1, edge_index), 0.2)
        
        # 步骤三：执行跨维度残差相加，将浅层纯净力学先验与深层图特征强行交融
        h_final = h2 + self.residual_proj(h0)
        
        # 步骤四：送入节点解码器，直接产生 [节点数, 2] 的连续概率矩阵
        node_preds = self.node_decoder(h_final)
        
        # 步骤五：提取全图边的拓扑端点索引（起点 row，终点 col）
        row, col = edge_index[0], edge_index[1]
        edge_combined_feats = torch.cat([h_final[row], h_final[col]], dim=1)
        edge_preds = self.edge_decoder(edge_combined_feats)
        
        return node_preds, edge_preds


# =========================================================================
# 2. 第二版模型：纯 GATv2 图注意力模型 (现代动态自注意力机制)
# =========================================================================
class PureDynamicGATModel(nn.Module):
    """
    基于 GATv2 算子的同构网格图多任务应力流预测网络。
    特点：采用 8 个并行注意力头，单头 16 维，拼接后刚好是 128 维，与 SAGE/GCN 实现完美的中间维度对齐。
    """
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=64, heads=8):
        super(PureDynamicGATModel, self).__init__()
        
        # 【特征嵌入层】
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # 【图注意力编码层】
        # 8 个头拼接（concat=True）后，这一层的最终输出刚好又是 128 维
        self.conv1 = GATv2Conv(hidden_dim1, hidden_dim1 // heads, heads=heads, concat=True, dropout=0.2)
        # 第二层采用单头注意力（heads=1），降维提炼到 64 维
        self.conv2 = GATv2Conv(hidden_dim1, hidden_dim2, heads=1, dropout=0.2)
        
        # 【跨维度残差连接快捷通道】
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)
        
        # 【节点多任务解码器】(64 -> 32 -> 2)
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )
        
        # 【边单任务解码器】(128 -> 32 -> 1)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 步骤一：初始特征嵌入
        h0 = self.node_embedding(x)
        
        # 步骤二：经历第一层 8 头 GATv2 算子
        h1 = F.leaky_relu(self.conv1(h0, edge_index), 0.2)
        
        # 步骤三：经历第二层单头 GATv2 算子并激活 (🟢 已删除多余的重复激活行)
        h2 = F.leaky_relu(self.conv2(h1, edge_index), 0.2)
        
        # 步骤四：激活残差连接 [节点数, 64] + [节点数, 64] = [节点数, 64]
        h_final = h2 + self.residual_proj(h0)
        
        # 后端多任务分流解码预测
        node_preds = self.node_decoder(h_final)
        row, col = edge_index[0], edge_index[1]
        edge_combined_feats = torch.cat([h_final[row], h_final[col]], dim=1)
        edge_preds = self.edge_decoder(edge_combined_feats)
        
        return node_preds, edge_preds


# =========================================================================
# 3. 第三版模型：纯 GCN 图卷积模型 (基于度归一化的标准对称卷积)
# =========================================================================
class PureGCNModel(nn.Module):
    """
    基于最经典 GCN 算子的同构网格图多任务应力流预测网络。
    特点：根据节点的度（邻居数量）进行两端拉普拉斯归一化，适合作为最基础的控制变量底座。
    """
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=64):
        super(PureGCNModel, self).__init__()
        
        # 【特征嵌入层】
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # 【图卷积编码层】：连续叠加两层最经典的 GCN 图卷积算子
        self.conv1 = GCNConv(hidden_dim1, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2) 
        
        # 【跨维度残差快捷通道】
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)
        
        # 【节点多任务解码器】(64 -> 32 -> 2)
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )
        
        # 【边单任务解码器】(128 -> 32 -> 1)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 步骤一：初始特征嵌入
        h0 = self.node_embedding(x)
        
        # 步骤二：经历两层 GCN 算子
        h1 = F.leaky_relu(self.conv1(h0, edge_index), 0.2)
        h2 = F.leaky_relu(self.conv2(h1, edge_index), 0.2)
        
        # 步骤三：激活残差连接
        h_final = h2 + self.residual_proj(h0)
        
        # 后端多任务分流解码预测
        node_preds = self.node_decoder(h_final)
        row, col = edge_index[0], edge_index[1]
        edge_combined_feats = torch.cat([h_final[row], h_final[col]], dim=1)
        edge_preds = self.edge_decoder(edge_combined_feats)
        
        return node_preds, edge_preds


# =========================================================================
# 4. 第四版模型：标准 GAT 图注意力模型 (传统静态自注意力机制)
# =========================================================================
class PureClassicGATModel(nn.Module):
    """
    基于 2018 ICLR 经典 GATConv 算子的图神经网络。
    特点：采用传统的静态自注意力机制，邻域注意力的相对大小排序在训练完成后被固定，
          无法随节点力学特征的实时剧烈波动而动态重新排序。用于作为基准对照。
    """
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=64, heads=8):
        super(PureClassicGATModel, self).__init__()
        
        # 【特征嵌入层】
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # 【图注意力编码层】：全部调用标准的 GATConv
        self.conv1 = GATConv(hidden_dim1, hidden_dim1 // heads, heads=heads, concat=True, dropout=0.2)
        self.conv2 = GATConv(hidden_dim1, hidden_dim2, heads=1, dropout=0.2)
        
        # 【跨维度线性残差连接快捷通道】
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)
        
        # 【节点多任务感知机解码器】
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )
        
        # 【边单任务感知机解码器】
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, hidden_dim2 // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2 // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        h0 = self.node_embedding(x)
        
        # 经历第一层 8头 传统老版 GAT 传导
        h1 = F.leaky_relu(self.conv1(h0, edge_index), 0.2)
        # 经历第二层 单头 传统老版 GAT 降维
        h2 = F.leaky_relu(self.conv2(h1, edge_index), 0.2)
        
        # 残差交融
        h_final = h2 + self.residual_proj(h0)
        
        # 多任务解码
        node_preds = self.node_decoder(h_final)
        row, col = edge_index[0], edge_index[1]
        edge_combined_feats = torch.cat([h_final[row], h_final[col]], dim=1)
        edge_preds = self.edge_decoder(edge_combined_feats)
        
        return node_preds, edge_preds