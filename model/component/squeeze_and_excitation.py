"""
@FileName   squeeze_and_excitation
@Author  24
@Date    2024/1/26 22:32
@Version 1.0.0
freedom is the oxygen of the soul.
"""
from torch import nn


class SEModule(nn.Module):
    """SE（Squeeze-and-Excitation）模块::
        INPUTS:
            channel：输入特征图的通道数
            reduction：压缩比例
    """
    def __init__(self, channel, reduction=1):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x)
        return x * se_weight


'''  
may the force be with you.
@FileName   squeeze_and_excitation
Created by 24 on 2024/1/26.
'''
