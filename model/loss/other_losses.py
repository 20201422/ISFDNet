import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    """
    重建损失函数，结合 L1 损失和余弦相似度损失。
    """
    def __init__(self, l1_weight=0.7, cos_weight=0.3):
        super(ReconstructionLoss, self).__init__()
        self.l1_weight = l1_weight
        self.cos_weight = cos_weight

    def forward(self, f_recon, f_mixed_target):
        def _channel_norm(feat: torch.Tensor) -> torch.Tensor:
            b, c, h, w = feat.shape
            flat = feat.view(b, c, -1)
            mean = flat.mean(dim=2, keepdim=True)
            std = flat.std(dim=2, keepdim=True) + 1e-6
            normed = (flat - mean) / std
            return normed.view_as(feat)

        f_recon_n = _channel_norm(f_recon)
        f_mixed_n = _channel_norm(f_mixed_target)

        l1_normed = torch.mean(torch.abs(f_recon_n - f_mixed_n))
        cos_sim = F.cosine_similarity(f_recon_n.view(f_recon_n.size(0), -1),
                                      f_mixed_n.view(f_mixed_n.size(0), -1), dim=1)
        cos_loss = torch.mean(1.0 - cos_sim)

        return self.l1_weight * l1_normed + self.cos_weight * cos_loss


class CycleConsistencyLoss(nn.Module):
    """
    循环一致性损失函数。
    """
    def forward(self, f_id_cycle, f_id_target):
        id_cycle_cos = 1.0 - torch.sum(F.normalize(f_id_cycle, dim=1) * F.normalize(f_id_target, dim=1), dim=1).mean()

        return id_cycle_cos