# 导入操作系统底层路径和文件处理库
import os
# 导入数据科学核心矩阵和数学计算库
import numpy as np
# 导入用于读取和处理 CSV 表格的 Pandas 库
import pandas as pd
# 导入深度学习核心框架 PyTorch
import torch
# 从 PyTorch Geometric 导入用于构建同构图的数据图结构类 Data
from torch_geometric.data import Data
# 从 sklearn 导入独热编码器，用于将拉压文本类型转换为数字独热向量
from sklearn.preprocessing import OneHotEncoder

def process_vector_field(val):
    """
    辅助函数：专门处理由于受力轴屏蔽导致的【方向分量】空值 '/' 符号。
    如果数据点为 NaN 或 '/'，则在几何上将其安全地定义为 0.0，代表在此正交轴向无方向场投影贡献。
    """
    # 如果值为 Pandas 识别的空值(NaN) 或者去除空格后是斜杠 '/'
    if pd.isna(val) or str(val).strip() == '/':
        # 返回 0.0，防止字符串直接转 float 时程序崩溃
        return 0.0
    # 如果是正常的数字分量字符串，直接转换为标准的浮点数
    return float(val)

def read_stress_graph(node_path, edge_path):
    """
    核心函数：读取单张楼板的节点 CSV 和边 CSV，并将其转换为标准的 PyG 同构图 Data 对象。
    """
    # 1. 使用 pandas 读取节点和边表格数据
    node_df = pd.read_csv(node_path)
    edge_df = pd.read_csv(edge_path)
    
    # 2. 建立节点 ID 的局部映射
    # 提取节点表里的原始 node_id 列并转为 Python 列表
    node_ids = node_df['node_id'].tolist()
    # 创建字典映射 {原始不连续ID : 从0开始的网格内部连续索引}，防止图拓扑边索引越界
    id_map = {orig_id: local_idx for local_idx, orig_id in enumerate(node_ids)}
    
    # 3. 初始化标准的 2 维独热编码器 (OneHotEncoder)
    # 因为 type 列永远只有 '+t'（拉）或 '-c'（压）两种状态，不需要包含 '/'
    type_categories = [['+t', '-c']]
    # 实例化编码器：不生成稀疏矩阵(方便直接转换)，若遇到边缘未知符号则安全忽略不报错
    encoder = OneHotEncoder(categories=type_categories, sparse_output=False, handle_unknown='ignore')
    
    # 4. 提取节点表中的力的类型列，并做文本格式清洗
    # 提取 m1_type 转换为字符串，去除两端空格，并举重轻重地擦除可能存在的草率英文句点 '.'
    m1_types = node_df['m1_type'].astype(str).str.strip().str.replace('.', '', regex=False).values.reshape(-1, 1)
    # 提取 m2_type 执行完全相同的文本清洗（此时它始终是 +t 或 -c，可以直接参与编码）
    m2_types = node_df['m2_type'].astype(str).str.strip().str.replace('.', '', regex=False).values.reshape(-1, 1)
    
    # 5. 执行 One-Hot 编码转换
    # 将整个 m1_type 列批量转换为 (节点数, 2) 维度的独热特征矩阵
    m1_type_encoded = encoder.fit_transform(m1_types)
    # 将整个 m2_type 列批量转换为 (节点数, 2) 维度的独热特征矩阵
    m2_type_encoded = encoder.fit_transform(m2_types)
    
    # 6. 遍历节点表，拼接和构建每个节点 13 维的输入特征矩阵
    node_features = []
    # 逐行迭代遍历读取每一个网格基准点
    for idx, row in node_df.iterrows():
        # 读取几何空间三维归一化坐标特征 (由于是平面，z 通常全为 0)
        x, y, z = float(row['x']), float(row['y']), float(row['z'])
        
        # 调用辅助函数安全解析主弯矩 m1 的 xy 方向分解分量（若为 '/' 会自动安全转为 0.0）
        """
        为什么会有'/'
        """
        m1_vx = process_vector_field(row['m1_vx'])
        m1_vy = process_vector_field(row['m1_vy'])
        # 直接读取主弯矩 m1 的力的绝对值大小（因为必定是纯浮点数，无需防崩函数）
        m1_abs = float(row['m1_abs'])
        
        # 调用辅助函数安全解析最小弯矩 m2 的 xy 方向分解分量（若为 '/' 会自动安全转为 0.0）
        m2_vx = process_vector_field(row['m2_vx'])
        m2_vy = process_vector_field(row['m2_vy'])
        # 直接读取最小弯矩 m2 的力的绝对值大小（因为必定是纯浮点数，无需防崩函数）
        m2_abs = float(row['m2_abs'])
        
        # 从前面计算好的 One-Hot 矩阵中取出当前行对应的 m1 2 维特征列表（例如 [1.0, 0.0]）
        m1_t_oh = m1_type_encoded[idx].tolist()
        # 从前面计算好的 One-Hot 矩阵中取出当前行对应的 m2 2 维特征列表（例如 [0.0, 1.0]）
        m2_t_oh = m2_type_encoded[idx].tolist()
        
        # 严格按照固定顺序拼接单个节点的所有特征：
        # 坐标(3) + m1方向与大小(3) + m1独热(2) + m2方向与大小(3) + m2独热(2) = 共计 13 维
        feat = [x, y, z, m1_vx, m1_vy, m1_abs] + m1_t_oh + [m2_vx, m2_vy, m2_abs] + m2_t_oh
        # 将当前节点的 13 维特征向量追加到特征总列表中
        node_features.append(feat)
        
    # 将包含全图所有节点特征的二维列表转化为 PyTorch 标准的浮点型张量，形状为 [节点数, 13]
    X = torch.tensor(node_features, dtype=torch.float)
    
    # 7. 解析节点的 Ground Truth (真实标签)连续概率标签（用于多任务回归预测预测热力图）
    # 提取节点是否是奇异点的真实概率，并升维变成 (节点数, 1) 的列向量
    y_singularity = torch.tensor(node_df['is_singularity'].values, dtype=torch.float).unsqueeze(1)
    # 提取节点是否在主应力线上的真实概率，并升维变成 (节点数, 1) 的列向量
    y_on_psl = torch.tensor(node_df['is_on_psl'].values, dtype=torch.float).unsqueeze(1)
    # 在第 1 维度（列方向）水平拼接，组装成节点的多任务回归目标张量，形状为 [节点数, 2]
    #Y_node = torch.cat([y_singularity, y_on_psl], dim=1)
    Y_node = y_singularity  # 不再包含 is_on_psl
    
    # 8. 解析图的拓扑连接边以及边的真实标签
    edge_indices = [] # 用于存放边起终点拓扑的临时列表
    edge_labels = []  # 用于存放边对应概率标签的临时列表
    
    # 逐行迭代遍历边表 CSV
    for _, row in edge_df.iterrows():
        # 获取当前行边的原始起始节点 ID 和终止节点 ID
        orig_start = int(row['start_id'])
        orig_end = int(row['end_id'])
        
        # 通过我们之前建立的 id_map 映射字典，将原始 ID 转换为连续的网格内部索引
        u = id_map[orig_start]
        v = id_map[orig_end]
        
        # 因为三角面网格结构是物理无向图，为了图卷积的双向传播，我们需要正向和反向成对添加边
        edge_indices.append([u, v]) # 添加正向边 u -> v
        edge_indices.append([v, u]) # 添加反向边 v -> u
        
        # 获取这条边距离主应力线的 Ground Truth 概率连续值
        close_to_psl = float(row['close_to_psl'])
        # 因为一条无向边被拆分成了两条有向边，所以两条边需要各自共享分配一次相同的概率标签
        edge_labels.append(close_to_psl)
        edge_labels.append(close_to_psl)
        
    # 将边拓扑关系转换为长整型张量，并进行转置操作使其形状符合 PyG 要求的 [2, 边数*2]，确保内存连续
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    # 将边对应的连续概率标签转换为浮点型张量，形状为 [边数*2, 1]
    Y_edge = torch.tensor(edge_labels, dtype=torch.float).unsqueeze(1)
    
    # 9. 封装并打包返回 PyG Data 同构图对象
    data = Data(
        x=X,                        # 注入节点特征张量矩阵 [num_nodes, 13]
        edge_index=edge_index,      # 注入全图拓扑连接邻接关系 [2, num_edges]
        y_node=Y_node,              # 注入节点预测的双任务概率目标 [num_nodes, 1]
        y_edge=Y_edge               # 注入边预测的单任务概率目标 [num_edges, 1]
    )
    
    # 返回构建完成的单张图样本数据对象
    return data

def remove_z_and_global_normalize(graph_list):
    """
    对图列表中每个图执行：
    1. 删除节点特征中的 z 列（原索引 2）
    2. 对 m1_vx, m1_vy, m1_abs, m2_vx, m2_vy, m2_abs（新索引 [2,3,4,7,8,9]）进行全局 Min-Max 归一化
    3. 对 y_node（is_singularity）进行全局 Min-Max 归一化到 [0,1]
    返回新的图列表，其中节点特征维度为 12（无 z），且 y_node 已归一化。
    """
    if not graph_list:
        return []

    # 1. 去除 z 列，创建新图列表
    graphs_no_z = []
    for data in graph_list:
        # x 原始: [x, y, z, m1_vx, m1_vy, m1_abs, m1_oh(2), m2_vx, m2_vy, m2_abs, m2_oh(2)]
        x_no_z = torch.cat([data.x[:, :2], data.x[:, 3:]], dim=1)
        # 保留原始 y_node 的引用（未归一化）
        new_data = Data(
            x=x_no_z,
            edge_index=data.edge_index,
            y_node=data.y_node.clone(),      # 克隆避免修改原数据
            y_edge=data.y_edge
        )
        graphs_no_z.append(new_data)

    # 2. 全局归一化节点特征 (x) 中的选定列
    norm_cols = [2, 3, 4, 7, 8, 9]  # m1_vx, m1_vy, m1_abs, m2_vx, m2_vy, m2_abs
    col_min = [float('inf')] * len(norm_cols)
    col_max = [float('-inf')] * len(norm_cols)

    for data in graphs_no_z:
        x = data.x
        for i, col in enumerate(norm_cols):
            col_data = x[:, col]
            min_val = col_data.min().item()
            max_val = col_data.max().item()
            if min_val < col_min[i]:
                col_min[i] = min_val
            if max_val > col_max[i]:
                col_max[i] = max_val

    for data in graphs_no_z:
        x = data.x
        for i, col in enumerate(norm_cols):
            min_val = col_min[i]
            max_val = col_max[i]
            if max_val - min_val > 1e-8:
                x[:, col] = (x[:, col] - min_val) / (max_val - min_val)
            else:
                x[:, col] = torch.zeros_like(x[:, col])

    # 3. 全局归一化 y_node (is_singularity)
    # 收集所有图中 y_node 的全局 min 和 max
    y_min = float('inf')
    y_max = float('-inf')
    for data in graphs_no_z:
        y = data.y_node
        if y.numel() == 0:
            continue
        cur_min = y.min().item()
        cur_max = y.max().item()
        if cur_min < y_min:
            y_min = cur_min
        if cur_max > y_max:
            y_max = cur_max

    # 应用归一化
    if y_max - y_min > 1e-8:
        for data in graphs_no_z:
            data.y_node = (data.y_node - y_min) / (y_max - y_min)
    else:
        # 若所有 y_node 为常数，则设为 0
        for data in graphs_no_z:
            data.y_node = torch.zeros_like(data.y_node)

    return graphs_no_z