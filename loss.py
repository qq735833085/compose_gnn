# loss.py — 多任务联合损失函数
# =============================================================================
# 三个预测任务：
#   1. 节点奇异点 (is_singularity)  — 极端失衡 1:6319
#   2. PSL边 Set1  (is_psl_1)       — 中度失衡 1:10
#   3. PSL边 Set2  (is_psl_2)       — 中度失衡 1:11
#
# 损失组合: Focal Loss (处理失衡) + Dice Loss (优化重叠度)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: -α(1-p)^γ * log(p)

    自动降低"容易"样本的权重，聚焦"困难"样本。
    γ 越大，对简单样本的压制越强。

    Args:
        gamma: 聚焦参数 (γ=0 → 标准 BCE, γ=4 → 极端聚焦)
        alpha: 正样本权重 (处理类别失衡)
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred:  [*, 1] 预测概率 (sigmoid 后)
        target: [*, 1] 真实标签 (0/1)
        """
        # BCE loss per element
        bce = F.binary_cross_entropy(pred, target, reduction='none')

        # 聚焦因子: (1 - p_t)^γ
        p_t = pred * target + (1 - pred) * (1 - target)  # p if t=1 else 1-p
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha 加权
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)

        loss = alpha_weight * focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss: 1 - 2|P∩T| / (|P| + |T|)

    直接优化预测与真实标签的重叠度，对小目标敏感。
    对极端失衡任务尤其有效（不依赖大量负样本）。

    Args:
        smooth: 平滑项，防止除零
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred:  [*, 1] 预测概率
        target: [*, 1] 真实标签
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class MultiTaskStressLoss(nn.Module):
    """
    楼板应力场多任务联合损失。

    任务权重:
        L_total = w_sing * L_sing + w_psl1 * L_psl1 + w_psl2 * L_psl2

    每项 = Focal Loss + λ * Dice Loss
    """

    def __init__(self,
                 # 奇异点 (极端失衡 → 高 gamma)
                 sing_gamma=4.0, sing_alpha=0.95, sing_dice_weight=0.3,
                 # PSL边 (中度失衡 → 低 gamma)
                 psl_gamma=1.0, psl_alpha=0.75, psl_dice_weight=0.5,
                 # 任务权重
                 w_sing=2.0, w_psl1=1.0, w_psl2=1.0):
        super().__init__()

        # 奇异点损失
        self.focal_sing = FocalLoss(gamma=sing_gamma, alpha=sing_alpha)
        self.dice_sing = DiceLoss()
        self.sing_dice_w = sing_dice_weight

        # PSL Set1 损失
        self.focal_psl1 = FocalLoss(gamma=psl_gamma, alpha=psl_alpha)
        self.dice_psl1 = DiceLoss()
        self.psl1_dice_w = psl_dice_weight

        # PSL Set2 损失
        self.focal_psl2 = FocalLoss(gamma=psl_gamma, alpha=psl_alpha)
        self.dice_psl2 = DiceLoss()
        self.psl2_dice_w = psl_dice_weight

        # 任务权重
        self.w_sing = w_sing
        self.w_psl1 = w_psl1
        self.w_psl2 = w_psl2

    def forward(self, node_pred, node_target, edge_pred, edge_target):
        """
        Args:
            node_pred:   [N, 1] 奇异点预测概率
            node_target: [N, 1] 奇异点真实标签 (0/1)
            edge_pred:   [E, 2] PSL边预测概率 [is_psl_1, is_psl_2]
            edge_target: [E, 2] PSL边真实标签
        Returns:
            total_loss: 标量
            log_dict:   各子损失值 (用于日志)
        """
        # ---- 1. 奇异点损失 ----
        loss_sing_focal = self.focal_sing(node_pred, node_target)
        loss_sing_dice = self.dice_sing(node_pred, node_target)
        loss_sing = loss_sing_focal + self.sing_dice_w * loss_sing_dice

        # ---- 2. PSL Set1 损失 ----
        loss_psl1_focal = self.focal_psl1(edge_pred[:, 0:1], edge_target[:, 0:1])
        loss_psl1_dice = self.dice_psl1(edge_pred[:, 0:1], edge_target[:, 0:1])
        loss_psl1 = loss_psl1_focal + self.psl1_dice_w * loss_psl1_dice

        # ---- 3. PSL Set2 损失 ----
        loss_psl2_focal = self.focal_psl2(edge_pred[:, 1:2], edge_target[:, 1:2])
        loss_psl2_dice = self.dice_psl2(edge_pred[:, 1:2], edge_target[:, 1:2])
        loss_psl2 = loss_psl2_focal + self.psl2_dice_w * loss_psl2_dice

        # ---- 总损失 ----
        total_loss = (self.w_sing * loss_sing +
                      self.w_psl1 * loss_psl1 +
                      self.w_psl2 * loss_psl2)

        log_dict = {
            'total': total_loss.item(),
            'sing/focal': loss_sing_focal.item(),
            'sing/dice': loss_sing_dice.item(),
            'sing/total': loss_sing.item(),
            'psl1/focal': loss_psl1_focal.item(),
            'psl1/dice': loss_psl1_dice.item(),
            'psl1/total': loss_psl1.item(),
            'psl2/focal': loss_psl2_focal.item(),
            'psl2/dice': loss_psl2_dice.item(),
            'psl2/total': loss_psl2.item(),
        }

        return total_loss, log_dict


# =============================================================================
# 兼容旧接口（简化版 MSE，用于快速测试）
# =============================================================================
def mse_loss(node_pred, node_true, edge_pred, edge_true, node_weight=1.0, edge_weight=1.0):
    """节点和边的加权 MSE 损失（兼容旧代码）"""
    node_loss = F.mse_loss(node_pred, node_true)
    edge_loss = F.mse_loss(edge_pred, edge_true)
    total_loss = node_weight * node_loss + edge_weight * edge_loss
    return total_loss, node_loss.item(), edge_loss.item()
