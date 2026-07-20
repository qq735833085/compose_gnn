# GAN 实验日志

> 项目：基于 GNN 的楼板应力场预测（奇异点 + PSL 边）
> 数据集：`merged_25cases_continuous_augmented_x7.pt`（200 图，25 cases × 8 增强）
> 环境：`torch24_py312` (PyTorch 2.8.0+cu129, PyG 2.7.0)
> GPU：CUDA available

---

## 实验 1：GAN v2 — Hinge GAN (batch_size=2)

**日期**：2026-07-17
**脚本**：`train_gan_v2.py`
**保存目录**：`trained_model/2026_07_17_1728/GANv2_gat_hinge_bs2_h128_rec10.0/`

### 超参数

| 参数 | 值 |
|------|-----|
| batch_size | 2 |
| epochs | 300 |
| gnn_type | gat |
| hidden_dim (G) | 128 |
| hidden_dim (D) | 128 |
| gan_mode | hinge |
| lambda_recon | 10.0 |
| lambda_adv | 1.0 |
| lambda_gp | 0.0 |
| d_updates | 2 |
| lr_g / lr_d | 1e-4 / 1e-4 |
| G params | 87,011 |
| D params | 191,489 |

### 结果

| 指标 | Epoch 1 | 最佳 | 最佳 Epoch | Epoch 300 |
|------|---------|------|-----------|-----------|
| Sing AUC | 0.676 | **0.767** | E85 | 0.756 |
| Sing Dice | 0.067 | **0.170** | E85 | 0.124 |
| M1 AUC | 0.441 | **0.617** | E265 | 0.609 |
| M1 Dice | 0.126 | **0.293** | E265 | 0.244 |
| M2 AUC | 0.553 | 0.562 | E300 | 0.562 |
| Edge Dice | 0.235 | **0.262** | E265 | 0.223 |

### 问题分析

1. **判别器过强 → mode collapse**：D loss 从 12.9 → 0.03，G adv loss 从 -1.4 → +9.1。D 完美区分真假，G 无法获得有效梯度。
2. **M2 任务退化**：M2 Dice 最佳出现在 epoch 1 (0.276)，后续持续下降。
3. **Sing Dice 偏低**：AUC 0.76 尚可但 Dice 仅 0.12，说明概率分布校准不足，定位精度差。
4. **D 参数量为 G 的 2.2 倍**（191k vs 87k），天然不平衡。

### 结论

Hinge GAN 在 batch_size=2 下可运行，但训练不稳定。需要：
- 降低 D 能力/提高 G 能力
- 添加正则化（R1 梯度惩罚、label smoothing）
- 不平衡学习率

---

## 实验 2：GAN v2 — WGAN-GP (batch_size=2)

**日期**：2026-07-17
**状态**：❌ 失败，提前终止

### 超参数

| 参数 | 值 |
|------|-----|
| gan_mode | wgan-gp |
| lambda_gp | 10.0 |
| d_updates | 3 |
| 其他 | 同实验 1 |

### 结果

Epoch 1 开始损失即爆炸（D loss: -5000+, G adv loss: ±数千）。WGAN-GP 在无 Spectral Normalization 的情况下在图级别判别器上极不稳定。

### 结论

WGAN-GP 需要 Spectral Norm 或更强的约束。跳过此配置。

---

## 实验 3：GAN v3 — Hinge GAN 稳定版 (batch_size=2)

**日期**：2026-07-17
**脚本**：`train_gan_v3.py`
**保存目录**：`trained_model/2026_07_17_2013/GANv3_gat_hinge_bs2_Gh128_Dh64/`

### 超参数（相比 v2 的变化）

| 参数 | v2 值 | v3 值 | 目的 |
|------|-------|-------|------|
| hidden_dim (D) | 128 | **64** | 降低 D 容量 |
| G/D 参数比 | 0.45 | **1.65** | G 比 D 大 65% |
| lr_g / lr_d | 1e-4 / 1e-4 | **2e-4 / 5e-5** | G 比 D 快 4 倍 |
| lambda_adv | 1.0 | **0.5** | 减半对抗压力 |
| lambda_r1 | 0 | **1.0** | R1 梯度惩罚 |
| label_smooth | 1.0 | **0.9** | 标签平滑 |
| d_noise | 0 | **0.05** | D 输入噪声 |
| d_updates | 2 | **1** | 降低 D 更新频率 |
| g_clip_norm | 无 | **5.0** | G 梯度裁剪 |

### 结果

| 指标 | Epoch 1 | 最佳 | 最佳 Epoch | Epoch 300 |
|------|---------|------|-----------|-----------|
| Sing AUC | 0.674 | **0.906** | E179 | 0.925 |
| Sing Dice | 0.063 | **0.386** | E179 | 0.163 |
| M1 AUC | 0.478 | **0.673** | E271 | 0.660 |
| M1 Dice | — | **0.337** | E232 | 0.293 |
| M2 AUC | 0.571 | 0.621 | E263 | 0.620 |
| Edge Dice | 0.168 | **0.322** | E232 | 0.293 |

### 稳定性分析

| 指标 | v2 (E300) | v3 (E300) | 改善 |
|------|-----------|-----------|------|
| D loss | 0.03 | **1.89** | D 不再过强 ✅ |
| G adv | +9.08 | **+0.07** | 对抗平衡 ✅ |
| D loss 范围 (末30轮) | ~0.02-0.05 | **1.75-2.07** | 稳定 ✅ |

### 关键发现

1. **Sing AUC 大幅提升**：0.767 → **0.925** (+21%)，v3 的奇异点排序能力显著增强
2. **Edge Dice 提升**：0.262 → **0.322** (+23%)
3. **训练稳定**：D loss 维持 1.9 左右（v2 崩塌到 0.03），G adv 接近零
4. **Sing Dice 波动大**：最佳 0.386 (E179) 但最终 0.163，说明定位精度仍需改善

### 结论

Hinge GAN v3 在所有指标上大幅超越 v2，且训练稳定无 mode collapse。R1 惩罚 + 标签平滑 + 不平衡 LR 组合有效。

---

## 实验 4：GAN v3 — LSGAN 稳定版 (batch_size=2)

**日期**：2026-07-17
**脚本**：`train_gan_v3.py`
**保存目录**：`trained_model/2026_07_17_2148/GANv3_gat_lsgan_bs2_Gh128_Dh64/`

### 超参数

同实验 3，仅 `gan_mode='lsgan'`

### 结果

| 指标 | Epoch 1 | 最佳 | 最佳 Epoch | Epoch 300 |
|------|---------|------|-----------|-----------|
| Sing AUC | 0.519 | 0.577 | E300 | 0.577 |
| Sing Dice | 0.022 | **0.024** | E9 | **0.000** |
| M1 AUC | 0.450 | **0.672** | E271 | 0.670 |
| M1 Dice | — | **0.342** | E271 | 0.340 |
| M2 AUC | 0.552 | 0.618 | E200 | 0.618 |
| Edge Dice | 0.239 | **0.325** | E224 | 0.301 |

### 问题分析

1. **Sing 任务完全坍塌**：Sing Dice 从 0.024 → 0.000，G 完全放弃了奇异点预测
2. **Edge 任务正常**：M1 Dice=0.342, M2 Dice=0.317, Edge Dice=0.325 — 与 Hinge 版本持平
3. **D loss 偏低**：0.43-0.60（LSGAN 的 MSE 特性导致 D loss 天然较小）

### 结论

LSGAN 在 edge 预测上与 Hinge GAN 持平，但导致了 singularity 任务的灾难性遗忘。G 发现放弃 sing 任务可以更容易骗过 D（因为 edge 任务有更多样本）。**不推荐 LSGAN 用于多任务 GAN**。

---

## 汇总对比

| 实验 | Sing AUC | Sing Dice | Edge Dice | 训练稳定性 |
|------|----------|-----------|-----------|-----------|
| v2 Hinge | 0.767 | 0.170 | 0.262 | ❌ D 崩塌 |
| v2 WGAN-GP | — | — | — | ❌ 损失爆炸 |
| **v3 Hinge** | **0.925** | **0.386** | **0.322** | ✅ 稳定 |
| v3 LSGAN | 0.577 | 0.000 | 0.325 | ⚠️ Sing 坍塌 |

**最佳方案：Hinge GAN v3** — Sing AUC 0.925, Edge Dice 0.322, 训练稳定。

### 下一步方向

1. 进一步改善 Sing Dice（AUC 高但 Dice 低 → 概率阈值校准问题）
2. 尝试在 G 中添加注意力机制或更强的边特征编码
3. 考虑 Post-processing：对 G 输出做温度缩放 (temperature scaling) 校准概率分布
4. 增大 batch_size 到 4，配合更小的 D hidden_dim (32) 测试

---
