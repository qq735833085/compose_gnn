# Compose-GNN: 基于图神经网络的楼板应力场预测

基于 GAN 架构的 GNN 模型，从楼板有限元应力场中预测：
- **奇异点 (Singularity)**：应力集中的节点位置
- **PSL 边 (Principal Stress Line)**：主应力线方向标注

## 数据集

| 文件 | 大小 | 说明 |
|------|------|------|
| `merged_25cases_continuous.pt` | 53MB | 25 个楼板案例，连续概率标签 |
| `merged_25cases_continuous_augmented_x7.pt` | 422MB | 增强版（7x 数据增强），200 个图 |

数据格式：PyTorch Geometric `Data` 对象
- `x [N, 12]`：应力场特征（坐标、主应力方向、幅值等）
- `y_node [N, 1]`：奇异点连续概率
- `y_edge [E, 2]`：PSL 连续概率（M1/M2 两个主应力方向）
- `edge_index [2, E]`：有限元网格邻接关系

## 模型

| 模型 | 说明 | Sing Dice | Edge Dice |
|------|------|-----------|-----------|
| `v3_hinge/best_sing.pth` | Hinge GAN 稳定版 | 0.386 | — |
| `v3_hinge/best_edge.pth` | Hinge GAN 稳定版 | — | 0.322 |
| `v5_compare/best_sing.pth` | **GAT — 当前最佳 (Sing)** | **0.409** | — |
| `v5_compare/best_edge.pth` | GAT — 当前最佳 (Edge) | — | 0.316 |
| `v5_physics/S0.0_M0.0_C0.0/` | 物理基线 | 0.363 | 0.316 |
| `v5_physics/S0.1_M1.0_C0.5/` | 全部物理约束 (最佳物理) | 0.382 | 0.316 |

## 使用方法

```python
import torch
from models.gan_models import GeneratorGNN, DiscriminatorGNN

# 加载数据集
dataset = torch.load('datasets/merged_25cases_continuous_augmented_x7.pt', weights_only=False)

# 加载模型
checkpoint = torch.load('models/v5_compare/best_sing.pth', weights_only=False)
generator = GeneratorGNN(input_dim=12, hidden_dim=128, hidden_dim2=64, gnn_type='gat')
generator.load_state_dict(checkpoint['G'])
generator.eval()

# 推理
data = dataset[0]
with torch.no_grad():
    sing_prob, edge_prob = generator(data)
```

## 实验历程

| 版本 | 方案 | Sing Dice | Edge Dice | 稳定性 |
|------|------|-----------|-----------|--------|
| v2 Hinge | GAT GAN | 0.170 | 0.262 | D 崩塌 |
| v3 Hinge | +R1+Label Smooth+不平衡LR | 0.386 | 0.322 | 稳定 |
| v3 LSGAN | LS loss | 0.000 | 0.325 | Sing 坍塌 |
| **v5 GAT** | **GNN 架构对比 (最佳)** | **0.409** | 0.316 | 稳定 |
| v5 SAGE | SAGE 对比 | 0.240 | 0.316 | 稳定 |
| v5 GCN | GCN 对比 | 0.165 | 0.304 | 稳定 |
| v5 Physics C | GAT + 物理约束 | 0.382 | 0.316 | 稳定 |

详见 [experiment_log.md](https://github.com/karos1214/compose-gnn)

## 训练代码

```bash
git clone https://github.com/karos1214/compose-gnn
cd compose-gnn
pip install torch torch-geometric modelscope
python train_gan_v3.py          # 稳定版 Hinge GAN
python train_gan_v5_compare.py  # GNN 架构对比
python train_gan_v5_physics.py  # 物理约束实验
```

## 监控面板

```bash
python monitor_server.py --port 8080
# 浏览器打开 http://localhost:8080
```

## 引用

```
@misc{compose-gnn,
  title={Compose-GNN: GAN-based Stress Field Prediction for Concrete Slabs},
  author={Karos},
  year={2026},
}
```
