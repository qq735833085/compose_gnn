import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import random
from datetime import datetime
from tqdm import tqdm

# 导入所有待测试的模型类（确保路径正确）
from models.test_models import PureSAGEModel, PureDynamicGATModel, PureGCNModel, PureClassicGATModel

# 导入数据增强模块
from utils.augment import StressFieldAugmentor, AugmentedGraphList, AUGMENT_CONFIG


# ========== 损失函数定义 ==========
def mse_loss(node_pred, node_true, edge_pred, edge_true, node_weight=1.0, edge_weight=1.0):
    """节点和边的加权 MSE 损失"""
    node_loss = torch.nn.functional.mse_loss(node_pred, node_true)
    edge_loss = torch.nn.functional.mse_loss(edge_pred, edge_true)
    total_loss = node_weight * node_loss + edge_weight * edge_loss
    return total_loss, node_loss.item(), edge_loss.item()


# ========== 辅助函数：皮尔逊相关系数 ==========
def pearson_corr(x, y):
    """计算两个 numpy 数组的皮尔逊相关系数"""
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    return r_num / r_den if r_den != 0 else 0.0


# ========== 验证函数（回归指标） ==========
def model_val(model, device, val_loader, node_weight=1.0, edge_weight=1.0):
    """
    验证集评估，返回 (平均总损失, MSE, MAE, Pearson相关系数, R²)
    """
    model.eval()
    total_loss = 0.0
    all_node_preds = []
    all_node_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            node_pred, edge_pred = model(batch)                     # (N,1), (E,1)
            loss, _, _ = mse_loss(node_pred, batch.y_node,
                                  edge_pred, batch.y_edge,
                                  node_weight, edge_weight)
            total_loss += loss.item()

            all_node_preds.append(node_pred.cpu())
            all_node_labels.append(batch.y_node.cpu())

    all_node_preds = torch.cat(all_node_preds, dim=0).numpy().ravel()
    all_node_labels = torch.cat(all_node_labels, dim=0).numpy().ravel()

    # 回归指标
    mse = mean_squared_error(all_node_labels, all_node_preds)
    mae = mean_absolute_error(all_node_labels, all_node_preds)
    pearson = pearson_corr(all_node_labels, all_node_preds)
    r2 = r2_score(all_node_labels, all_node_preds)

    model.train()
    return total_loss / len(val_loader), mse, mae, pearson, r2


# ========== 主训练函数 ==========
def train_stress_flow(model_class,
                      epochs=200,
                      lr=1e-3,
                      batch_size=2,
                      hidden_dim1=128,
                      hidden_dim2=64,
                      node_weight=1.0,
                      edge_weight=1.0,
                      data_path=r'D:\2025-2026 RA\3-sitp 2026 基于图学习的楼板应力线找形\composite\graph_dataset\20260531_44cases.pt',
                      device='cuda',
                      save_root='./trained_model',
                      seed=42,
                      use_augmentation=True):
    # 固定随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 创建保存路径（按模型名+参数自动生成子文件夹）
    model_label = f'{model_class.__name__}_h1-{hidden_dim1}_h2-{hidden_dim2}'
    cur_time = datetime.now().strftime("%Y_%m_%d")
    save_dir = os.path.join(save_root, cur_time, model_label)
    os.makedirs(save_dir, exist_ok=True)
    print(f"▶ 实验开始: {model_label} | 保存路径: {save_dir}")

    # 加载数据集（假设是预处理好的图列表）
    dataset = torch.load(data_path, weights_only=False)
    n_total = len(dataset)
    n_train = int(n_total * 0.9)

    # 数据增强：若使用静态增强数据集则跳过运行时增强（提升训练效率）
    if use_augmentation:
        augmentor = StressFieldAugmentor(AUGMENT_CONFIG)
        train_dataset = AugmentedGraphList(dataset[:n_train], augmentor)
        print(f"  运行时增强已启用 (数据集={n_total} 张图)")
    else:
        train_dataset = AugmentedGraphList(dataset[:n_train])  # 无增强（已静态增强）
        print(f"  使用静态增强数据集 (数据集={n_total} 张图, 无运行时增强)")

    val_dataset = AugmentedGraphList(dataset[n_train:])  # 验证集始终无增强

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型（输入维度固定为12，因为已丢弃z列）
    model = model_class(input_dim=12,
                        hidden_dim1=hidden_dim1,
                        hidden_dim2=hidden_dim2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_r2 = -float('inf')   # 用于保存最佳 R² 模型
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Training {model_class.__name__} Epoch {epoch+1}/{epochs}")
        for batch in loop:
            batch = batch.to(device)
            optimizer.zero_grad()
            node_pred, edge_pred = model(batch)
            loss, node_l, edge_l = mse_loss(node_pred, batch.y_node,
                                            edge_pred, batch.y_edge,
                                            node_weight, edge_weight)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item(), node_loss=node_l, edge_loss=edge_l)

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证
        val_loss, val_mse, val_mae, val_pearson, val_r2 = model_val(
            model, device, val_loader, node_weight, edge_weight
        )

        # 记录日志
        log_path = os.path.join(save_dir, 'metrics.csv')
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write("epoch,train_loss,val_loss,val_mse,val_mae,val_pearson,val_r2\n")
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{val_loss:.6f},{val_mse:.6f},{val_mae:.6f},{val_pearson:.6f},{val_r2:.6f}\n")

        # 保存最佳模型（依据 R²）
        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ 新最佳模型 (R²={val_r2:.4f}) 已保存")

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={val_loss:.4f} | R²={val_r2:.4f} | Pearson={val_pearson:.4f}")

    print(f"✅ 训练完成！最佳验证 R² = {best_r2:.4f}")
    return model


# ========== 批量运行示例 ==========
if __name__ == '__main__':
    # 选择要训练的模型列表
    all_models = [
        PureSAGEModel,
        PureDynamicGATModel,
        PureGCNModel,
        PureClassicGATModel
    ]

    # ★ 选择数据集：静态增强版（推荐）或原始版
    # 方式1: 使用静态增强数据集（训练效率高，无需运行时增强）
    DATA_PATH = r"D:\composite_0602\graph_dataset\20260602_104_pro_augmented_x3.pt"
    USE_AUG = False  # 数据集已增强，关闭运行时增强

    # 方式2: 使用原始数据集 + 运行时增强（每 epoch 不同变体）
    # DATA_PATH = r"D:\composite_0602\graph_dataset\20260602_104_pro.pt"
    # USE_AUG = True

    # 公共训练参数
    common_params = {
        'epochs': 200,
        'lr': 1e-3,
        'batch_size': 2,
        'hidden_dim1': 128,
        'hidden_dim2': 64,
        'node_weight': 1.0,
        'edge_weight': 0.5,          # 边损失权重可适当降低，因为边数量远多于节点
        'data_path': DATA_PATH,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_root': './trained_model',
        'seed': 42,
        'use_augmentation': USE_AUG,  # ★ 控制是否使用运行时增强
    }

    for model_cls in all_models:
        print("\n" + "="*60)
        print(f"开始训练模型: {model_cls.__name__}")
        print("="*60)
        train_stress_flow(model_cls, **common_params)