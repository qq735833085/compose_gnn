import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import random
from datetime import datetime
from tqdm import tqdm

# 导入所有待测试的模型类
from models.test_models import PureSAGEModel, PureDynamicGATModel, PureGCNModel, PureClassicGATModel

# 导入数据增强模块
from utils.augment import StressFieldAugmentor, AugmentedGraphList, AUGMENT_CONFIG

# 导入损失函数
from loss import MultiTaskStressLoss


# ========== 辅助函数：皮尔逊相关系数 ==========
def pearson_corr(x, y):
    """计算两个 numpy 数组的皮尔逊相关系数"""
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    return r_num / r_den if r_den != 0 else 0.0


# ========== 验证函数 ==========
def model_val(model, device, val_loader, criterion):
    """
    验证集评估，返回 (loss, sing_auc, psl1_auc, psl2_auc, log_dict)
    """
    from sklearn.metrics import roc_auc_score
    model.eval()
    total_loss = 0.0
    all_sing_preds, all_sing_labels = [], []
    all_psl1_preds, all_psl1_labels = [], []
    all_psl2_preds, all_psl2_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            node_pred, edge_pred = model(batch)  # [N,1], [E,2]
            loss, log_dict = criterion(node_pred, batch.y_node,
                                       edge_pred, batch.y_edge)
            total_loss += loss.item()

            all_sing_preds.append(node_pred.cpu())
            all_sing_labels.append(batch.y_node.cpu())
            all_psl1_preds.append(edge_pred[:, 0:1].cpu())
            all_psl1_labels.append(batch.y_edge[:, 0:1].cpu())
            all_psl2_preds.append(edge_pred[:, 1:2].cpu())
            all_psl2_labels.append(batch.y_edge[:, 1:2].cpu())

    all_sing_preds = torch.cat(all_sing_preds).numpy().ravel()
    all_sing_labels = torch.cat(all_sing_labels).numpy().ravel()
    all_psl1_preds = torch.cat(all_psl1_preds).numpy().ravel()
    all_psl1_labels = torch.cat(all_psl1_labels).numpy().ravel()
    all_psl2_preds = torch.cat(all_psl2_preds).numpy().ravel()
    all_psl2_labels = torch.cat(all_psl2_labels).numpy().ravel()

    # AUC（对二分类任务）
    def safe_auc(y_true, y_pred):
        if y_true.sum() == 0 or (y_true == 1).all():
            return 0.5
        return roc_auc_score(y_true, y_pred)

    sing_auc = safe_auc(all_sing_labels, all_sing_preds)
    psl1_auc = safe_auc(all_psl1_labels, all_psl1_preds)
    psl2_auc = safe_auc(all_psl2_labels, all_psl2_preds)

    # Dice 系数
    def dice(y_true, y_pred, th=0.5):
        pred_bin = (y_pred > th).astype(int)
        inter = (pred_bin * y_true).sum()
        return 2 * inter / (pred_bin.sum() + y_true.sum() + 1e-8)

    sing_dice = dice(all_sing_labels, all_sing_preds)
    psl1_dice = dice(all_psl1_labels, all_psl1_preds)
    psl2_dice = dice(all_psl2_labels, all_psl2_preds)

    model.train()
    return (total_loss / len(val_loader),
            sing_auc, psl1_auc, psl2_auc,
            sing_dice, psl1_dice, psl2_dice)


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
    print(f"Experiment: {model_label} | Save: {save_dir}")

    # 加载数据集（假设是预处理好的图列表）
    dataset = torch.load(data_path, weights_only=False)
    n_total = len(dataset)
    n_train = int(n_total * 0.9)

    # 数据增强：若使用静态增强数据集则跳过运行时增强（提升训练效率）
    if use_augmentation:
        augmentor = StressFieldAugmentor(AUGMENT_CONFIG)
        train_dataset = AugmentedGraphList(dataset[:n_train], augmentor)
        print(f"  On-the-fly augmentation enabled (n={n_total})")
    else:
        train_dataset = AugmentedGraphList(dataset[:n_train])
        print(f"  Static augmented dataset (n={n_total}, no runtime aug)")

    val_dataset = AugmentedGraphList(dataset[n_train:])  # 验证集始终无增强

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型（输入维度固定为12）
    model = model_class(input_dim=12,
                        hidden_dim1=hidden_dim1,
                        hidden_dim2=hidden_dim2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = MultiTaskStressLoss()

    best_psl1_auc = 0.0
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Training {model_class.__name__} Epoch {epoch+1}/{epochs}")
        for batch in loop:
            batch = batch.to(device)
            optimizer.zero_grad()
            node_pred, edge_pred = model(batch)
            loss, log_dict = criterion(node_pred, batch.y_node,
                                       edge_pred, batch.y_edge)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            loop.set_postfix(loss=f'{loss.item():.4f}',
                            sing=f'{log_dict["sing/total"]:.4f}',
                            psl1=f'{log_dict["psl1/total"]:.4f}')

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证
        val_loss, sing_auc, psl1_auc, psl2_auc, sing_dice, psl1_dice, psl2_dice = model_val(
            model, device, val_loader, criterion
        )

        # 记录日志
        log_path = os.path.join(save_dir, 'metrics.csv')
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write("epoch,train_loss,val_loss,sing_auc,psl1_auc,psl2_auc,sing_dice,psl1_dice,psl2_dice\n")
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{val_loss:.6f},{sing_auc:.4f},{psl1_auc:.4f},{psl2_auc:.4f},{sing_dice:.4f},{psl1_dice:.4f},{psl2_dice:.4f}\n")

        # 保存最佳模型（依据 PSL1 AUC + PSL2 AUC 均值）
        avg_psl_auc = (psl1_auc + psl2_auc) / 2
        if avg_psl_auc > best_psl1_auc:
            best_psl1_auc = avg_psl_auc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"  New best (avg PSL AUC={avg_psl_auc:.4f}) saved")

        print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f} Val={val_loss:.4f} | "
              f"Sing AUC={sing_auc:.3f} Dice={sing_dice:.3f} | "
              f"PSL1 AUC={psl1_auc:.3f} Dice={psl1_dice:.3f} | "
              f"PSL2 AUC={psl2_auc:.3f} Dice={psl2_dice:.3f}")

    print(f"Training done. Best PSL AUC={best_psl1_auc:.4f}")
    return model


# ========== 批量运行示例 ==========
if __name__ == '__main__':
    # 选择要训练的模型列表
    from models.hybrid_parallel_model import ParallelHybridStressModel
    all_models = [
        PureSAGEModel,
        PureDynamicGATModel,
        PureGCNModel,
        PureClassicGATModel,
        ParallelHybridStressModel,
    ]

    # ★ 数据集路径
    DATA_PATH = r"D:\composite_0602\datasets\03_graph\merged_25cases_augmented_x7.pt"
    USE_AUG = False  # 静态增强数据集，无需运行时增强

    # 公共训练参数
    common_params = {
        'epochs': 200,
        'lr': 1e-3,
        'batch_size': 2,
        'hidden_dim1': 128,
        'hidden_dim2': 64,
        'data_path': DATA_PATH,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_root': './trained_model',
        'seed': 42,
        'use_augmentation': USE_AUG,
    }

    for model_cls in all_models:
        print("\n" + "="*60)
        print(f"Start training: {model_cls.__name__}")
        print("="*60)
        train_stress_flow(model_cls, **common_params)