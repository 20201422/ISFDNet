"""
@file_name  grl
@author     24
@date       2025/8/11 14:42
@version    1.0.0
freedom is the oxygen of the soul.
"""
import torch
from torch import nn



class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GRL(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversal.apply(x, self.lambd)


"""
@file_name  grl
Created by 24 on 2025/8/11
"""
