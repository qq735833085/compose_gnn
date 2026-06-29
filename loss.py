# loss.py
# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(node_preds, edge_preds, data, node_weight=1.0, edge_weight=0.5):
    """
    node_preds: (N, 1)
    edge_preds: (E, 1)
    data: 包含 y_node (N,1) 和 y_edge (E,1)
    """
    node_loss = F.mse_loss(node_preds, data.y_node)
    edge_loss = F.mse_loss(edge_preds, data.y_edge)
    total_loss = node_weight * node_loss + edge_weight * edge_loss
    return total_loss, node_loss.item(), edge_loss.item()



class StressTaskLoss(nn.Module):
    def __init__(self):
        super(StressTaskLoss, self).__init__()
        # 使用 MSELoss 进行概率场拟合
        self.mse = nn.MSELoss()

    def forward(self, node_preds, node_targets, edge_preds, edge_targets):
        """
        不再合并 Loss，直接返回两个独立的 MSE 结果。
        """
        # 1. 计算奇异点回归损失
        loss_node = self.mse(node_preds, node_targets.float())
        
        # 2. 计算边概率回归损失
        loss_edge = self.mse(edge_preds, edge_targets.view(-1, 1).float())
        
        # 3. 封装日志字典 (方便训练循环直接记录)
        log_dict = {
            "loss/node_singularity": loss_node.item(),
            "loss/edge_psl": loss_edge.item()
        }
        
        return loss_node, loss_edge, log_dict



# import torch
# import torch.nn as nn

# class UnifiedStressLoss(nn.Module):
#     def __init__(self, w_node1=5.0, w_node2=1.0, w_edge=1.0):
#         super(UnifiedStressLoss, self).__init__()
#         # 只有奇异点任务需要高权重
#         self.loss_n1_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
#         # 另外两个任务使用标准的 BCEWithLogitsLoss (pos_weight=1.0)
#         self.loss_n2_fn = nn.BCEWithLogitsLoss()
#         self.loss_e3_fn = nn.BCEWithLogitsLoss()
        
#         self.w_node1 = w_node1
#         self.w_node2 = w_node2
#         self.w_edge  = w_edge

#     def forward(self, node_preds, node_targets, edge_preds, edge_targets):
#         # 分别使用对应的损失函数
#         loss_n1 = self.loss_n1_fn(node_preds[:, 0], node_targets[:, 0].float())
#         loss_n2 = self.loss_n2_fn(node_preds[:, 1], node_targets[:, 1].float())
#         loss_e3 = self.loss_e3_fn(edge_preds, edge_targets.view(-1, 1).float())
        
#         total_loss = (self.w_node1 * loss_n1) + (self.w_node2 * loss_n2) + (self.w_edge * loss_e3)

#         # 5. 返回总损失与日志字典
#         loss_log = {
#             "total": total_loss.item(),
#             "n1_singularity": loss_n1.item(),
#             "n2_online": loss_n2.item(),
#             "e3_psl": loss_e3.item()
#         }
        
#         return total_loss, loss_log