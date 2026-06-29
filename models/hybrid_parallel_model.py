# hybrid_parallel_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# 从 PyTorch Geometric 导入同构图三大骨干算子
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv

# =========================================================================
# 👑 并行自适应混合模型 (三路图算子融合 + 残差投影)
# =========================================================================
class ParallelHybridStressModel(nn.Module):
    """
    自适应融合 GCN、GATv2 与 GraphSAGE 的网格图多任务应力流预测网络。
    输入接口与标准 PyG Data 完全兼容，无需额外的全局宏观特征 u。
    """
    def __init__(self, input_dim=13, hidden_dim1=128, hidden_dim2=64, head=8):
        super(ParallelHybridStressModel, self).__init__()

        # -----------------------------------------------------------------
        # 🔧 核心可调参数注释说明：
        # 1. input_dim (13): 节点原始力学先验维度，由数据集决定，不可随意更改。
        # 2. hidden_dim1 (128): 拓扑高维宽通道。
        #    - 调大: 增强网络对复杂非线性应力场的拟合能力，但显存开销成倍增加，易过拟合。
        #    - 调小: 节省显存，加快训练，但可能导致复杂受力区域（如奇异点）预测欠拟合。
        # 3. hidden_dim2 (64): 图编码层输出/解码层输入维度。
        #    - 调大: 保留更多图拓扑聚合特征，提高解码器上限，但增大了后端全连接层的计算量。
        #    - 调小: 特征高度压缩，强迫网络提炼宏观场特征，但过小会导致细节（边任务）特征丢失。
        # 4. head (8): GATv2 的多头注意力数。
        #    - 调大: 提升多视角并行捕捉奇异应力梯度的能力，但注意 hidden_dim1 必须能被其整除。
        #    - 调小: 弱化各向异性注意力的优势，退化为单一权重的场传导。
        # -----------------------------------------------------------------

        # 【1. 特征嵌入层】：将 13 维原始物理特征线性提升至 128 维高维空间
        self.node_embedding = nn.Sequential(
            # 全连接层：[节点数, 13] -> [节点数, 128]
            nn.Linear(input_dim, hidden_dim1),
            # 激活函数：LeakyReLU (负斜率0.2)，保留微弱的负力学梯度信息
            nn.LeakyReLU(0.2),
            # 随机失活：统一设为 0.2
            # 调大(如0.5): 强迫网络提取更鲁棒特征，抗过拟合强，但收敛变慢
            # 调小(如0.0): 允许全量特征通过，局部拟合快，但在小样本上极易过拟合
            nn.Dropout(0.2)
        )

        # 【2. 运行分支 A】：GCN 图卷积双层通路 (基于度归一化的标准对称拉普拉斯卷积)
        # 第一层卷积：在 128 维空间内执行 1 阶网格邻域各向同性信息粗筛
        self.gcn1 = GCNConv(hidden_dim1, hidden_dim1)
        # 第二层卷积：将特征由 128 维平滑演进演进到 64 维
        self.gcn2 = GCNConv(hidden_dim1, hidden_dim2)

        # 【3. 运行分支 B】：GATv2 现代动态注意力双层通路 (捕获应力突变与奇异点)
        # 第一层：输入 128 维，多头拼接(concat=True)，单头输出 128//8=16 维，拼接后刚好对齐 128 维
        self.gat1 = GATv2Conv(hidden_dim1, hidden_dim1 // head, heads=head, dropout=0.2)
        # 第二层：采用单头注意力（heads=1），直接将 128 维特征动态降维提炼到 64 维
        self.gat2 = GATv2Conv(hidden_dim1, hidden_dim2, heads=1, dropout=0.2)

        # 【4. 运行分支 C】：GraphSAGE 均权采样双层通路 (适合大面积平滑的力学传导)
        # 第一层：在 128 维空间中向外扩展 1 层网格进行均权/拼接聚合
        self.sage1 = SAGEConv(hidden_dim1, hidden_dim1)
        # 第二层：捕捉 2 层跨度的全局特征并演进到 64 维
        self.sage2 = SAGEConv(hidden_dim1, hidden_dim2)

        # 【5. 跨维度残差快捷通道】：因为初始特征 h0 (128维) 与混合图特征 (64维) 维度不同，用线性层投影对齐
        self.residual_proj = nn.Linear(hidden_dim1, hidden_dim2)

        # 【6. 三路并行分支的可学习自注意力决策矩阵】：初始化为 [1, 3] 的张量
        self.attn_weights = nn.Parameter(torch.empty((1, 3)))
        # 使用 Xavier 均匀分布初始化权重，确保初始状态下三路分支权重均衡
        nn.init.xavier_uniform_(self.attn_weights)

        # 【7. 节点多任务解码器】：接收 64 维混合高级特征，并行输出 [奇异点概率, 在线概率]
        self.node_decoder = nn.Sequential(
            # 隐藏层降维：64 维 -> 32 维
            nn.Linear(hidden_dim2, hidden_dim2 // 2),  
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            # 输出层：32 维 -> 2 维 (分别对应两大节点任务目标)
            nn.Linear(hidden_dim2 // 2, 1),             
            # 门控约束：用 Sigmoid 将连续数值约束在 [0, 1] 概率场区间
            nn.Sigmoid()                                
        )
        
        # 【8. 边单任务解码器】：拼接两端端点特征（64*2=128维），输出 [close_to_psl概率]
        self.edge_decoder = nn.Sequential(
            # 边描述矩阵降维：128 维 -> 32 维
            nn.Linear(hidden_dim2 * 2, hidden_dim2 // 2), 
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            # 输出层：32 维 -> 1 维 (边接近主应力线的概率)
            nn.Linear(hidden_dim2 // 2, 1),              
            # 门控约束：约束在 [0, 1] 概率区间
            nn.Sigmoid()                                 
        )

    def forward(self, data):
        """
        前向传播数据推理流
        data: 标准 PyG Data 包，包含 data.x [节点数, 13] 和 data.edge_index [2, 边数*2]
        """
        # 解包图数据的局部节点特征矩阵与静态拓扑边索引
        x, edge_index = data.x, data.edge_index
        
        # Step 1: 通过基础特征嵌入层，将原始力学特征映射到 128 维宽通道
        h0 = self.node_embedding(x)

        # Step 2: 定义内部三路并行分支的演进闭包 (统一使用 0.2 的 LeakyReLU 激活)
        def apply_branch(conv1, conv2):
            # 第一层图卷积激活
            h1 = F.leaky_relu(conv1(h0, edge_index), 0.2)
            # 第二层图卷积激活，输出规整为 64 维
            h2 = F.leaky_relu(conv2(h1, edge_index), 0.2)
            return h2

        # Step 3: 三路并发，各自提取对应的拓扑场特征 [节点数, 64]
        h_gcn  = apply_branch(self.gcn1, self.gcn2)   # 拉普拉斯各向同性特征
        h_gat  = apply_branch(self.gat1, self.gat2)   # 各向异性动态注意力特征
        h_sage = apply_branch(self.sage1, self.sage2) # 邻域平权特征

        # Step 4: 通过 Softmax 对可学习参数自适应归一化，动态生成三路分支的融合权重 (和为1)
        attn_scores = F.softmax(self.attn_weights, dim=1)
        
        # 加权求和：让网络在训练中自己感应在当前应力场传导中谁的贡献最大
        h_combined = (
            attn_scores[0, 0] * h_gcn +
            attn_scores[0, 1] * h_gat +
            attn_scores[0, 2] * h_sage
        )

        # Step 5: 激活跨维度大残差相加，将浅层纯净局部力学先验与深层图混合拓扑特征强行交融
        h_final = h_combined + self.residual_proj(h0)

        # Step 6: 后端节点多任务分流解码，直接产生 [节点数, 2] 的连续概率矩阵
        node_preds = self.node_decoder(h_final)

        # Step 7: 提取全图边的拓扑端点索引（源节点 row，目标节点 col）
        row, col = edge_index[0], edge_index[1]
        
        # 瞬间切片抓取每条边起终点的更新特征并进行水平拼接，融合成 [边数*2, 128] 的边描述矩阵
        edge_combined_feats = torch.cat([h_final[row], h_final[col]], dim=1)
        
        # 送入边解码器，产生并输出 [边数*2, 1] 的靠近主应力线概率向量
        edge_preds = self.edge_decoder(edge_combined_feats)

        # 成对返回多任务预测结果
        return node_preds, edge_preds