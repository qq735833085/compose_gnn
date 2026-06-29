# utils/augment.py
# =============================================================================
# 物理感知的图数据增强模块 — 用于楼板应力场 GNN 训练
# =============================================================================
# 五种增强技术（按物理安全性从高到低排列）：
#
#   1. 镜像 (Reflection)    ★★★★★ 绝对安全：对称几何的严格物理对称性
#   2. 旋转 (Rotation)      ★★★★☆ 高度安全：应力张量在坐标旋转下严格协变
#   3. 力值缩放 (Scale)     ★★★☆☆ 线弹性安全：等比缩放保持相对分布，保守范围 0.7~1.4
#   4. 平移 (Translation)   ★★★☆☆ 低幅安全：σ=0.01 模拟网格离散化原点偏移
#   5. 特征抖动 (Jitter)    ★★☆☆☆ 仅坐标+力值：方向向量不参与（保护 m1⊥m2 正交性）
#
# 核心物理不变量：
#   - m1 ⊥ m2 正交性：方向向量不做独立扰动
#   - 力值非负性：m1_abs, m2_abs ≥ 0 始终成立
#   - 标签协变性：is_singularity 和 close_to_psl 随应力场同步变换
# =============================================================================

import math
import random
import copy

import torch
from torch_geometric.data import Data


# =============================================================================
# 特征索引常量（与 read_oridata.py 中 remove_z 后的 12 维布局一致）
# =============================================================================
# [0] x,  [1] y,
# [2] m1_vx,  [3] m1_vy,  [4] m1_abs,  [5] m1_oh_0,  [6] m1_oh_1,
# [7] m2_vx,  [8] m2_vy,  [9] m2_abs,  [10] m2_oh_0,  [11] m2_oh_1

IDX_X, IDX_Y = 0, 1
IDX_M1_VX, IDX_M1_VY, IDX_M1_ABS = 2, 3, 4
IDX_M1_OH_0, IDX_M1_OH_1 = 5, 6
IDX_M2_VX, IDX_M2_VY, IDX_M2_ABS = 7, 8, 9
IDX_M2_OH_0, IDX_M2_OH_1 = 10, 11


# =============================================================================
# 默认增强配置（推荐：平衡增强强度与物理一致性）
# =============================================================================
AUGMENT_CONFIG = {
    'rotation': {
        'prob': 0.8,                     # 旋转增强触发概率
        'angle_mode': 'continuous',       # 'discrete': 90°/180°/270° 严格方形边界
                                          # 'continuous'（当前）: 任意角度+等比缩放适配
                                          #   'continuous'（谨慎）: 任意角度 + 缩放到 [0,1]，几何比例不变
        'angle_range': (0, 2 * math.pi),  # 连续角度范围（仅 angle_mode='continuous' 时生效）
        # ★ 连续模式不 clamp 越界坐标，而是等比缩放回 [0,1]，保持楼板几何形状
        #   代价：节点间距离变化，应力场受尺寸效应影响。离散模式无此问题。
    },
    'reflection': {
        'prob': 0.6,                      # 镜像增强触发概率
        'axes': ['x', 'y', 'both'],       # 随机选择翻转轴
    },
    'scale': {
        'prob': 0.7,                      # 力值缩放触发概率
        'alpha_min': 0.7,                 # 最小缩放因子（保守：仅 ±30%）
        'alpha_max': 1.4,                 # 最大缩放因子
        'mode': 'log_uniform',            # 'log_uniform' 或 'uniform'
        # 物理依据：线弹性假设下等比缩放保持奇异点位置不变。
        # 保守范围 0.7~1.4 兼顾混凝土非线性可能导致的应力重分布。
    },
    'translate': {
        'prob': 0.3,                      # 平移增强触发概率（降低：仅模拟离散化噪声）
        'sigma': 0.01,                    # 平移量标准差 ≈ 1% 楼板跨度
        # 物理依据：模拟有限元网格生成的微小原点偏移，不改变结构拓扑。
    },
    'jitter': {
        'prob': 0.3,                      # 抖动增强触发概率（降低：控制噪声引入量）
        'coord_sigma': 0.005,             # 坐标噪声标准差
        # 注意：方向向量不参与抖动 —— 独立噪声会破坏 m1⊥m2 正交性，
        # 引入物理上不存在的应力状态，故刻意排除。
        'abs_sigma_ratio': 0.02,          # 力值噪声比例（相对当前值）
        'abs_sigma_min': 0.001,           # 力值噪声最小标准差（防止零值无噪声）
    },
}


# =============================================================================
# 保守增强配置（高物理保真度，仅使用严格物理对称的增强）
# =============================================================================
CONSERVATIVE_CONFIG = {
    'rotation': {
        'prob': 0.6,
        'angle_mode': 'discrete',         # 仅 90°/180°/270°，零边界失真
        'angle_range': None,
    },
    'reflection': {
        'prob': 0.5,
        'axes': ['x', 'y', 'both'],
    },
    'scale': {
        'prob': 0.3,                       # 极保守：仅低频触发
        'alpha_min': 0.85,
        'alpha_max': 1.15,
        'mode': 'log_uniform',
    },
    'translate': {
        'prob': 0.0,                       # 关闭平移
        'sigma': 0.0,
    },
    'jitter': {
        'prob': 0.0,                       # 关闭抖动
        'coord_sigma': 0.0,
        'abs_sigma_ratio': 0.0,
        'abs_sigma_min': 0.0,
    },
}


# =============================================================================
# 应力场增强器
# =============================================================================
class StressFieldAugmentor:
    """
    物理感知的图数据增强器。

    对输入的 PyG Data 对象依次应用可选的增强变换。
    每次调用 __call__ 时，根据配置中的概率随机决定是否触发各增强。

    用法:
        augmentor = StressFieldAugmentor(AUGMENT_CONFIG)
        augmented_data = augmentor(original_data)
    """

    def __init__(self, config=None):
        """
        Args:
            config: 增强参数字典，使用 AUGMENT_CONFIG 中的默认值覆盖
        """
        self.config = copy.deepcopy(AUGMENT_CONFIG)
        if config is not None:
            self._deep_update(self.config, config)

    @staticmethod
    def _deep_update(base, override):
        """递归合并配置字典"""
        for key, value in override.items():
            if isinstance(value, dict) and key in base:
                StressFieldAugmentor._deep_update(base[key], value)
            else:
                base[key] = value

    # -------------------------------------------------------------------------
    # 主入口
    # -------------------------------------------------------------------------
    def __call__(self, data):
        """
        对单个 PyG Data 对象执行增强。

        Args:
            data: PyG Data 对象（会被 clone，原对象不修改）

        Returns:
            增强后的 PyG Data 对象
        """
        data = data.clone()  # 不修改原始数据

        # 按概率链式调用各增强
        data = self._maybe_rotate(data)
        data = self._maybe_reflect(data)
        data = self._maybe_scale(data)
        data = self._maybe_translate(data)
        data = self._maybe_jitter(data)

        return data

    # -------------------------------------------------------------------------
    # 1. 旋转增强
    # -------------------------------------------------------------------------
    def _maybe_rotate(self, data):
        """
        以 80% 概率对坐标和方向向量做同步旋转。

        - discrete 模式（推荐）：仅 90°/180°/270°，方形区域完美自映射，
          零边界失真，结合反射可生成 D4 群全部 8 种对称方向。
        - continuous 模式（谨慎）：任意角度旋转后等比缩放回 [0,1]，
          几何形状不变但节点间距被压缩（最大压缩 30%@45°），
          可能影响尺寸相关的应力特征。
        """
        cfg = self.config['rotation']
        if random.random() > cfg['prob']:
            return data

        if cfg.get('angle_mode', 'discrete') == 'discrete':
            # D4 对称群：仅使用严格保持方形边界的旋转角度
            theta = random.choice([math.pi / 2, math.pi, 3 * math.pi / 2])
            return self._apply_rotation_discrete(data, theta)
        else:
            lo, hi = cfg['angle_range']
            theta = random.uniform(lo, hi)
            return self._apply_rotation_continuous(data, theta)

    def _apply_rotation_discrete(self, data, theta):
        """
        离散旋转（90°/180°/270°）：方形 [0,1]² 完美自映射，无需 clamp 或缩放。

        坐标变换：
          - 90°:  (x-0.5, y-0.5) → (-(y-0.5), x-0.5) → 仍在 [-0.5, 0.5] → 完美回 [0,1]
          - 180°: (x-0.5, y-0.5) → (-(x-0.5), -(y-0.5))
          - 270°: (x-0.5, y-0.5) → (y-0.5, -(x-0.5))
        方向向量同步旋转，力值和标签不变。
        """
        x = data.x
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=x.dtype, device=x.device)

        # 坐标：以 (0.5, 0.5) 为中心旋转 → 方形完美映射，无需 clamp
        coords = x[:, [IDX_X, IDX_Y]] - 0.5
        coords_rot = coords @ R.T
        x[:, [IDX_X, IDX_Y]] = coords_rot + 0.5  # 必在 [0,1] 内

        # 方向向量同步旋转
        x[:, [IDX_M1_VX, IDX_M1_VY]] = self._normalize_dir(
            x[:, [IDX_M1_VX, IDX_M1_VY]] @ R.T)
        x[:, [IDX_M2_VX, IDX_M2_VY]] = self._normalize_dir(
            x[:, [IDX_M2_VX, IDX_M2_VY]] @ R.T)

        return data

    def _apply_rotation_continuous(self, data, theta):
        """
        连续角度旋转 + 等比缩放适配。

        问题：方板旋转后轴对齐包围盒超出 [0,1]，clamp 会压扁角部。
        方案：旋转后按包围盒等比缩放回 [0,1]，楼板几何形状不变，
        但节点间距被压缩（45° 时约 30%），属于近似增强。
        """
        x = data.x
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=x.dtype, device=x.device)

        # 旋转坐标（以中心为原点）
        coords = x[:, [IDX_X, IDX_Y]] - 0.5
        coords_rot = coords @ R.T

        # 计算旋转后的轴对齐包围盒
        x_min, x_max = coords_rot[:, 0].min().item(), coords_rot[:, 0].max().item()
        y_min, y_max = coords_rot[:, 1].min().item(), coords_rot[:, 1].max().item()

        # 等比缩放：取两个方向中较大的超出比例，保持正方形形状
        half_range = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
        if half_range > 0.5:
            scale = 0.5 / half_range
            coords_rot = coords_rot * scale

        # 平移回 [0, 1]
        x[:, [IDX_X, IDX_Y]] = coords_rot + 0.5

        # 方向向量同步旋转（方向是单位向量，不缩放）
        x[:, [IDX_M1_VX, IDX_M1_VY]] = self._normalize_dir(
            x[:, [IDX_M1_VX, IDX_M1_VY]] @ R.T)
        x[:, [IDX_M2_VX, IDX_M2_VY]] = self._normalize_dir(
            x[:, [IDX_M2_VX, IDX_M2_VY]] @ R.T)

        return data

    # -------------------------------------------------------------------------
    # 2. 镜像增强
    # -------------------------------------------------------------------------
    def _maybe_reflect(self, data):
        """以 60% 概率随机选择轴做镜像翻转"""
        cfg = self.config['reflection']
        if random.random() > cfg['prob']:
            return data

        axis = random.choice(cfg['axes'])
        return self._apply_reflection(data, axis)

    def _apply_reflection(self, data, axis):
        """
        对 data 执行镜像翻转。

        axis='x': 关于 y 轴翻转 (x -> 1 - x), m1_vx 和 m2_vx 取反
        axis='y': 关于 x 轴翻转 (y -> 1 - y), m1_vy 和 m2_vy 取反
        axis='both': 关于原点翻转，两组方向分量均取反
        """
        x = data.x

        if axis == 'x':
            x[:, IDX_X] = 1.0 - x[:, IDX_X]
            x[:, IDX_M1_VX] = -x[:, IDX_M1_VX]
            x[:, IDX_M2_VX] = -x[:, IDX_M2_VX]

        elif axis == 'y':
            x[:, IDX_Y] = 1.0 - x[:, IDX_Y]
            x[:, IDX_M1_VY] = -x[:, IDX_M1_VY]
            x[:, IDX_M2_VY] = -x[:, IDX_M2_VY]

        elif axis == 'both':
            x[:, IDX_X] = 1.0 - x[:, IDX_X]
            x[:, IDX_Y] = 1.0 - x[:, IDX_Y]
            x[:, [IDX_M1_VX, IDX_M1_VY]] = -x[:, [IDX_M1_VX, IDX_M1_VY]]
            x[:, [IDX_M2_VX, IDX_M2_VY]] = -x[:, [IDX_M2_VX, IDX_M2_VY]]

        return data

    # -------------------------------------------------------------------------
    # 3. 力值缩放增强
    # -------------------------------------------------------------------------
    def _maybe_scale(self, data):
        """
        以 70% 概率等比缩放 m1_abs 和 m2_abs。

        物理依据：在线弹性分析中，所有内力与外荷载成线性比例。
        m1_abs 和 m2_abs 同步缩放，方向、类型、标签不变。
        """
        cfg = self.config['scale']
        if random.random() > cfg['prob']:
            return data

        if cfg.get('mode', 'log_uniform') == 'log_uniform':
            # 对数均匀：在乘法空间中均匀采样
            log_min = math.log(cfg['alpha_min'])
            log_max = math.log(cfg['alpha_max'])
            alpha = math.exp(random.uniform(log_min, log_max))
        else:
            alpha = random.uniform(cfg['alpha_min'], cfg['alpha_max'])

        return self._apply_scale(data, alpha)

    def _apply_scale(self, data, alpha):
        """缩放 m1_abs 和 m2_abs"""
        x = data.x
        x[:, IDX_M1_ABS] = torch.clamp(x[:, IDX_M1_ABS] * alpha, min=0.0)
        x[:, IDX_M2_ABS] = torch.clamp(x[:, IDX_M2_ABS] * alpha, min=0.0)
        return data

    # -------------------------------------------------------------------------
    # 4. 平移增强
    # -------------------------------------------------------------------------
    def _maybe_translate(self, data):
        """
        以 40% 概率对坐标做微小随机平移。

        模拟不同网格原点离散化。方向向量和力值不变。
        """
        cfg = self.config['translate']
        if random.random() > cfg['prob']:
            return data

        dx = random.gauss(0, cfg['sigma'])
        dy = random.gauss(0, cfg['sigma'])

        return self._apply_translation(data, dx, dy)

    def _apply_translation(self, data, dx, dy):
        """平移坐标并 clamp 到 [0, 1]"""
        x = data.x
        x[:, IDX_X] = torch.clamp(x[:, IDX_X] + dx, 0.0, 1.0)
        x[:, IDX_Y] = torch.clamp(x[:, IDX_Y] + dy, 0.0, 1.0)
        return data

    # -------------------------------------------------------------------------
    # 5. 特征抖动增强
    # -------------------------------------------------------------------------
    def _maybe_jitter(self, data):
        """
        以 50% 概率对特征注入小量高斯噪声。

        模拟测量误差和标注不确定性。
        - 坐标：小量高斯噪声
        - 方向向量：小量噪声后重新归一化
        - 力值：比例噪声（相对当前值）
        - 独热编码和标签：不扰动
        """
        cfg = self.config['jitter']
        if random.random() > cfg['prob']:
            return data

        return self._apply_jitter(data, cfg)

    def _apply_jitter(self, data, cfg):
        """
        仅对坐标和力值注入微小高斯噪声。

        方向向量 (m1_vx/vy, m2_vx/vy) 不参与抖动：
        独立噪声会打破主应力方向的正交性 (m1 ⟂ m2)，
        生成物理上不存在的应力状态，对模型训练有害。
        """
        x = data.x
        N = x.shape[0]
        device = x.device
        dtype = x.dtype

        # --- 坐标噪声 ---
        coord_noise = torch.randn(N, 2, device=device, dtype=dtype) * cfg['coord_sigma']
        x[:, [IDX_X, IDX_Y]] = torch.clamp(
            x[:, [IDX_X, IDX_Y]] + coord_noise, 0.0, 1.0
        )

        # 方向向量: 刻意不做抖动（保护 m1⊥m2 正交性）

        # --- 力值噪声（比例噪声 + 最小标准差） ---
        m1_abs_val = x[:, IDX_M1_ABS]
        m1_abs_noise = torch.randn(N, device=device, dtype=dtype) * torch.clamp(
            m1_abs_val * cfg['abs_sigma_ratio'], min=cfg['abs_sigma_min']
        )
        x[:, IDX_M1_ABS] = torch.clamp(m1_abs_val + m1_abs_noise, min=0.0)

        m2_abs_val = x[:, IDX_M2_ABS]
        m2_abs_noise = torch.randn(N, device=device, dtype=dtype) * torch.clamp(
            m2_abs_val * cfg['abs_sigma_ratio'], min=cfg['abs_sigma_min']
        )
        x[:, IDX_M2_ABS] = torch.clamp(m2_abs_val + m2_abs_noise, min=0.0)

        # 独热编码和标签不扰动（保持不变）
        return data

    # -------------------------------------------------------------------------
    # 辅助函数：方向向量归一化
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_dir(vec, eps=1e-8):
        """
        将方向向量归一化到单位长度。

        零向量（原始为 '/' 的占位符）保持为零，不进行归一化。

        Args:
            vec: [N, 2] 方向向量张量
            eps: 判定零向量的阈值

        Returns:
            归一化后的 [N, 2] 张量
        """
        norms = vec.norm(dim=1, keepdim=True)  # [N, 1]
        # 仅对范数 > eps 的行做归一化，零向量保持为零
        mask = (norms > eps).squeeze(1)  # [N]
        if mask.any():
            vec[mask] = vec[mask] / norms[mask]
        # 零向量行保持全零（对应原始数据中 m1 或 m2 不存在的节点）
        return vec


# =============================================================================
# 增强图列表包装器
# =============================================================================
class AugmentedGraphList:
    """
    轻量 Dataset 包装器，在 __getitem__ 时自动对图执行增强。

    仅在训练模式下调用增强；验证模式原样返回数据。

    用法:
        dataset = torch.load('data.pt', weights_only=False)
        train_set = AugmentedGraphList(dataset[:n_train], StressFieldAugmentor())
        val_set = AugmentedGraphList(dataset[n_train:])  # 无增强

        train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=2, shuffle=False)
    """

    def __init__(self, data_list, augmentor=None):
        """
        Args:
            data_list: PyG Data 对象的列表
            augmentor: StressFieldAugmentor 实例，为 None 时不增强（验证模式）
        """
        self.data_list = data_list
        self.augmentor = augmentor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.augmentor is not None:
            data = self.augmentor(data)
        return data

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# =============================================================================
# 批量增强工具函数
# =============================================================================
def augment_dataset(data_list, config=None, n_augment=1):
    """
    对图列表进行批量增强，返回增强后的扩展列表。

    适用于"静态增强"场景：一次生成增强数据并保存为 .pt 文件，
    后续训练直接加载增强后的数据集。

    Args:
        data_list: 原始 PyG Data 列表
        config: 增强配置
        n_augment: 每个原始样本生成多少个增强变体

    Returns:
        原始列表 + 增强变体的合并列表
    """
    augmentor = StressFieldAugmentor(config)
    augmented = list(data_list)  # 保留原始数据
    for _ in range(n_augment):
        for data in data_list:
            augmented.append(augmentor(data.clone()))
    return augmented


# =============================================================================
# 调试与可视化辅助
# =============================================================================
def summarize_augmentation(original_data, augmented_data):
    """
    打印原始数据和增强后数据的统计摘要，用于调试验证。
    """
    print("=" * 60)
    print("数据增强前后对比")
    print("=" * 60)

    orig_x = original_data.x
    aug_x = augmented_data.x

    # 坐标
    print(f"\n{'指标':<20} {'原始':>18} {'增强后':>18}")
    print("-" * 56)
    for name, idx in [('x 均值', IDX_X), ('y 均值', IDX_Y),
                       ('m1_vx 均值', IDX_M1_VX), ('m1_vy 均值', IDX_M1_VY),
                       ('m1_abs 均值', IDX_M1_ABS),
                       ('m2_vx 均值', IDX_M2_VX), ('m2_vy 均值', IDX_M2_VY),
                       ('m2_abs 均值', IDX_M2_ABS)]:
        print(f"{name:<20} {orig_x[:, idx].mean().item():>18.6f} {aug_x[:, idx].mean().item():>18.6f}")

    # 标签不变性检查
    node_label_match = torch.allclose(original_data.y_node, augmented_data.y_node)
    edge_label_match = torch.allclose(original_data.y_edge, augmented_data.y_edge)
    edge_index_match = torch.equal(original_data.edge_index, augmented_data.edge_index)

    print(f"\n标签不变性检查:")
    print(f"  y_node 不变:  {node_label_match}")
    print(f"  y_edge 不变:  {edge_label_match}")
    print(f"  edge_index 不变: {edge_index_match}")
    print("=" * 60)


# =============================================================================
# 自检测试（直接运行此文件时执行）
# =============================================================================
if __name__ == '__main__':
    print("utils/augment.py - augment module self-check")
    print("=" * 60)

    # Construct a dummy PyG Data object
    N = 100
    E = 300

    dummy_x = torch.rand(N, 12)
    # Ensure direction vectors are non-zero (simulate real data)
    dummy_x[:, [IDX_M1_VX, IDX_M1_VY]] = torch.randn(N, 2)
    dummy_x[:, [IDX_M2_VX, IDX_M2_VY]] = torch.randn(N, 2)
    # Normalize direction vectors
    for cols in [(IDX_M1_VX, IDX_M1_VY), (IDX_M2_VX, IDX_M2_VY)]:
        norms = dummy_x[:, cols].norm(dim=1, keepdim=True)
        dummy_x[:, cols] = dummy_x[:, cols] / norms.clamp(min=1e-8)

    # Random edge index (bidirectional edges)
    half_edges = torch.randint(0, N, (2, E // 2))
    rev_edges = half_edges.flip(0)
    edge_index = torch.cat([half_edges, rev_edges], dim=1)

    data = Data(
        x=dummy_x,
        edge_index=edge_index,
        y_node=torch.rand(N, 1),
        y_edge=torch.rand(E, 1),
    )

    print(f"Original data: {N} nodes, {E} edges, x shape={data.x.shape}")

    # Test augmentor
    augmentor = StressFieldAugmentor()

    # Run multiple times to ensure no crashes
    for i in range(5):
        aug_data = augmentor(data.clone())
        assert aug_data.x.shape == data.x.shape, f"x shape mismatch: {aug_data.x.shape}"
        assert aug_data.y_node.shape == data.y_node.shape, "y_node shape mismatch"
        assert aug_data.y_edge.shape == data.y_edge.shape, "y_edge shape mismatch"
        assert torch.equal(aug_data.edge_index, data.edge_index), "edge_index was modified"
        # Check magnitude non-negativity
        assert (aug_data.x[:, IDX_M1_ABS] >= 0).all(), "m1_abs has negative values"
        assert (aug_data.x[:, IDX_M2_ABS] >= 0).all(), "m2_abs has negative values"
        # Check label invariance
        assert torch.allclose(aug_data.y_node, data.y_node), f"y_node modified in run {i}"
        assert torch.allclose(aug_data.y_edge, data.y_edge), f"y_edge modified in run {i}"
        print(f"  Test {i+1}/5 PASSED")

    # Test AugmentedGraphList
    data_list = [data.clone() for _ in range(5)]
    train_set = AugmentedGraphList(data_list, augmentor)
    val_set = AugmentedGraphList(data_list)  # no augmentation

    assert len(train_set) == 5
    assert train_set[0].x.shape == (N, 12)

    val_sample = val_set[0]
    # Validation set should return original data unchanged
    assert torch.allclose(val_sample.x, data_list[0].x), "val set should not be augmented"

    print("\nAll tests passed! Augmentation module is working correctly.")
