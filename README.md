# Compose-GNN: 基于 GAN 的图神经网络楼板应力场预测

使用 GAN + GNN 架构，从楼板有限元网格中同时预测**奇异点（应力集中）**和 **PSL 主应力线**。

## 项目概览

```
compose_gnn/
├── train_pure_gat.py              # Exp7: 纯 GAT 基线（无 GAN）
├── train_edge_aware_gan.py        # Exp8: Edge-Aware 判别器 GAN
├── train_focal_gan.py             # Exp9: Focal + Dice Loss GAN
├── train_combined_gan.py          # Exp10: Edge-Aware + Focal/Dice 组合
├── loss.py                        # 多任务损失函数
├── models/                        # 模型定义
│   ├── gan_models.py              # Generator / Discriminator GNN
│   ├── cgan.py                    # cGAN 变体
│   └── v3_hinge/ v4_hinge/       # 历史最佳 checkpoint
├── notebooks/                     # Jupyter 分析笔记
├── dashboard/                     # 训练监控面板
├── utils/                         # 数据增强 & 可视化工具
├── legacy/                        # 早期训练脚本（参考）
├── datasets/                      # PyG 图数据集
├── trained_model/                 # 训练产出
├── logs/                          # 运行日志
└── experiment_log.md              # 详细实验记录
```

## 数据集

| 文件 | 大小 | 说明 |
|------|------|------|
| `datasets/merged_25cases_continuous.pt` | 53MB | 25 个楼板案例，连续概率标签 |
| `datasets/merged_25cases_continuous_augmented_x7.pt` | 422MB | 增强版（7x 数据增强），200 图 |

每个图是一个 `torch_geometric.data.Data` 对象：

| 属性 | 维度 | 含义 |
|------|------|------|
| `x` | `[N, 12]` | 节点特征（坐标、主应力方向/幅值等） |
| `y_node` | `[N, 1]` | 奇异点连续概率 |
| `y_edge` | `[E, 2]` | PSL 连续概率（M1/M2 两向） |
| `edge_index` | `[2, E]` | FEM 网格邻接 |

## 实验进展

### 当前最佳 (E7-E10, BatchSize=2)

| 实验 | 脚本 | Sing Dice | Sing AUC | Edge Dice | M1 Dice |
|------|------|-----------|----------|-----------|---------|
| **Exp9** Focal+Dice GAN | `train_focal_gan.py` | **0.386** | 0.922 | 0.327 | 0.345 |
| Exp10 Combined | `train_combined_gan.py` | 0.358 | **0.956** | **0.328** | **0.351** |
| Exp8 Edge-Aware GAN | `train_edge_aware_gan.py` | 0.334 | 0.924 | 0.319 | 0.339 |
| Exp7 Pure GAT | `train_pure_gat.py` | 0.000 | 0.596 | 0.320 | 0.346 |

> 💡 Pure GAT 的 Sing Dice = 0.000，验证了 GAN 框架对奇异点预测的必要性。

### 历史版本演进

| 版本 | 方案 | Sing Dice | Edge Dice | 要点 |
|------|------|-----------|-----------|------|
| v2 Hinge | GAT GAN | 0.170 | 0.262 | D 崩塌 |
| v3 Hinge | +R1 + Label Smooth + 不平衡LR | 0.386 | 0.322 | ✅ 稳定 |
| v3 LSGAN | LS loss 对比 | 0.000 | 0.325 | Sing 坍塌 |
| v4 Hinge | +EMA + 架构升级 | — | — | 实验阶段 |
| v5 GAT | GNN 架构对比 | **0.409** | 0.316 | 历史最优 Sing |
| v5 SAGE/GCN | 消融实验 | 0.165–0.240 | 0.304–0.316 | GAT 显著优于其他 |

完整记录见 [`experiment_log.md`](experiment_log.md)。

## 快速开始

### 环境

```bash
pip install torch torch-geometric
# 或使用 conda 环境 torch24_py312
```

### 训练

```bash
# BS=2 并行训练（4 实验同时运行，约需 A10 24GB）
export PYTHONPATH=.
python train_pure_gat.py       --batch_size 2 --epochs 300  # Exp7
python train_edge_aware_gan.py --batch_size 2 --epochs 300  # Exp8
python train_focal_gan.py      --batch_size 2 --epochs 300  # Exp9
python train_combined_gan.py   --batch_size 2 --epochs 300  # Exp10
```

### 推理

```python
import torch
from models.gan_models import GeneratorGNN

# 加载数据
dataset = torch.load('datasets/merged_25cases_continuous_augmented_x7.pt', weights_only=False)

# 加载模型
ckpt = torch.load('models/v3_hinge/best_sing.pth', weights_only=False)
model = GeneratorGNN(input_dim=12, hidden_dim=128, hidden_dim2=64, gnn_type='gat')
model.load_state_dict(ckpt['G'])
model.eval()

# 推理
data = dataset[0]
with torch.no_grad():
    sing_prob, edge_prob = model(data)
```

### 监控面板

```bash
python dashboard/monitor_server.py --port 8080
# 浏览器打开 http://localhost:8080
```

## 模型权重

| 权重 | 路径 | Sing Dice |
|------|------|-----------|
| V3 Hinge — 最佳 Sing | `models/v3_hinge/best_sing.pth` | 0.386 |
| V3 Hinge — 最佳 Edge | `models/v3_hinge/best_edge.pth` | 0.322 |
| V4 Hinge — 最佳 Sing (EMA) | `models/v4_hinge/best_sing_ema.pth` | — |
| V4 Hinge — 最佳 Edge (EMA) | `models/v4_hinge/best_edge_ema.pth` | — |

## 引用

```bibtex
@misc{compose-gnn,
  title={Compose-GNN: GAN-based Stress Field Prediction for Concrete Slabs},
  author={Karos},
  year={2026},
}
```
