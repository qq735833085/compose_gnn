import torch # 导入 PyTorch 核心库
import torch.optim as optim # 导入优化器模块
from torch_geometric.loader import DataLoader # 导入 PyG 的数据加载器
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score # 导入评估指标
import numpy as np # 导入数值计算库
import os # 导入操作系统接口
import random # 导入随机数控制
from datetime import datetime # 导入时间日期模块
from tqdm import tqdm # 导入进度条插件

# 导入所有待测试的模型类
from models.test_models import PureSAGEModel, PureDynamicGATModel, PureGCNModel, PureClassicGATModel
from models.hybrid_parallel_model import ParallelHybridStressModel # 补全缺失的并行混合模型
from loss import UnifiedStressLoss # 导入自定义的多任务联合损失函数

# 导入数据增强模块
from utils.augment import StressFieldAugmentor, AugmentedGraphList, AUGMENT_CONFIG

def model_val(model, device, val_loader, criterion):
    """验证逻辑：在验证集上进行模型性能评估"""
    model.eval() # 将模型设置为评估模式（关闭 dropout 等）
    total_loss = 0.0 # 初始化累计损失
    all_preds, all_labels = [], [] # 初始化预测结果和标签列表
    
    with torch.no_grad(): # 验证阶段不计算梯度，节省显存
        for batch in val_loader: # 遍历验证集中的每一个 batch
            batch = batch.to(device) # 将数据送入指定设备（GPU）
            n_out, e_out = model(batch) # 模型推理获取节点和边输出
            loss, _ = criterion(n_out, batch.y_node, e_out, batch.y_edge) # 计算综合损失
            total_loss += loss.item() # 累计损失值
            
            # 以节点奇异点预测 (n_out[:,0]) 为核心指标进行评估
            all_preds.append(n_out[:, 0].cpu()) # 保存预测值到列表
            all_labels.append(batch.y_node[:, 0].cpu()) # 保存标签值到列表

    all_preds = torch.cat(all_preds, dim=0).numpy() # 将所有 batch 的预测合并为 numpy 数组
    all_labels = torch.cat(all_labels, dim=0).numpy() # 将所有 batch 的标签合并为 numpy 数组
    
    auc = roc_auc_score(all_labels, all_preds) # 计算 ROC AUC
    preds = (all_preds > 0.5).astype(int) # 使用 0.5 为阈值将概率转为类别
    acc = accuracy_score(all_labels, preds) # 计算准确率
    f1 = f1_score(all_labels, preds, zero_division=0) # 计算 F1 分数
    
    model.train() # 恢复模型为训练模式
    return total_loss / len(val_loader), auc, acc, f1 # 返回平均 Loss 和各项指标

def train_stress_flow(model_class, epochs=200, lr=1e-3, batch_size=2, hidden_dim1=128, hidden_dim2=64, data_path=r'D:\2025-2026 RA\3-sitp 2026 基于图学习的楼板应力线找形\composite\graph_dataset\20260531_44cases.pt', device='cuda', save_root='./trained_model', seed=42):
    # 设置随机种子以保证实验可复现
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    # 自动管理文件夹路径：基于日期和模型名创建实验文件夹
    model_label = f'{model_class.__name__}_h1-{hidden_dim1}_h2-{hidden_dim2}'
    cur_time = datetime.now().strftime("%Y_%m_%d")
    save_dir = os.path.join(save_root, cur_time)
    os.makedirs(save_dir, exist_ok=True)
    exp_index = str(len(os.listdir(save_dir))).zfill(3)
    save_path = os.path.join(save_dir, exp_index)
    os.makedirs(save_path, exist_ok=True)

    print(f"▶ 开始实验: {model_label} | 保存路径: {save_path}")
    
    dataset = torch.load(data_path, weights_only=False) # 加载预处理好的图数据集
    n_train = int(len(dataset) * 0.8) # 划分 80% 训练集

    # 数据增强：仅对训练集应用物理感知增强，验证集保持原样
    augmentor = StressFieldAugmentor(AUGMENT_CONFIG)
    train_dataset = AugmentedGraphList(dataset[:n_train], augmentor)
    val_dataset = AugmentedGraphList(dataset[n_train:])  # 无增强（验证模式）

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 训练数据加载器
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # 验证数据加载器

    # 初始化模型：根据传入的类名和参数动态实例化
    model = model_class(input_dim=12, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) # 使用 Adam 优化器
    criterion = UnifiedStressLoss() # 实例化自定义联合损失函数

    best_auc = 0.0 # 记录历史最佳 AUC
    for epoch in range(epochs): # 循环训练 epoch
        model.train() # 设置为训练模式
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training {model_class.__name__} Ep {epoch}"):
            batch = batch.to(device)
            optimizer.zero_grad() # 清零梯度
            n_out, e_out = model(batch) # 前向传播
            loss, _ = criterion(n_out, batch.y_node, e_out, batch.y_edge) # 计算 Loss
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            total_train_loss += loss.item()

        val_loss, auc, acc, f1 = model_val(model, device, val_loader, criterion) # 执行验证
        
        # 将训练日志追加写入 CSV 文件
        if not os.path.exists(os.path.join(save_path, 'val_metrics.csv')):
            with open(os.path.join(save_path, 'val_metrics.csv'), 'w') as f: f.write("epoch,val_loss,auc,acc,f1\n")
        with open(os.path.join(save_path, 'val_metrics.csv'), 'a') as f:
            f.write(f"{epoch+1},{val_loss},{auc},{acc},{f1}\n")
            
        # 实时保存表现最好的模型权重
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            
        print(f"Epoch {epoch+1} | Loss: {total_train_loss/len(train_loader):.4f} | AUC: {auc:.4f}")

if __name__ == '__main__':
    # 批量跑 5 个模型
    all_models = [PureSAGEModel, PureDynamicGATModel, PureGCNModel, PureClassicGATModel, ParallelHybridStressModel]
    for m in all_models:
        print(f"正在进行模型对比实验: {m.__name__}")
        train_stress_flow(m)