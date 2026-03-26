"""
@FileName   gabor
@Author  24
@Date    2024/1/26 22:30
@Version 1.0.0
freedom is the oxygen of the soul.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


# 滤波卷积层 Learnable Gabor Convolution (LGC) Layer
class GaborConv2d(nn.Module):
    """滤波卷积层 Learnable Gabor Convolution (LGC) Layer::
        INPUTS:
            channel_in：输入通道数
            channel_out：输出通道数
            kernel_size：卷积核大小
            stride：步长
            padding：填充大小
            init_ratio：初始参数（感受野）的缩放因子
    """
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_ratio = init_ratio

        self.kernel = 0

        self.SIGMA = 9.2 * self.init_ratio
        self.GAMMA = 2.0
        self.FREQUENCY = 0.057 / self.init_ratio

        self.sigma = nn.Parameter(torch.FloatTensor([self.SIGMA]), requires_grad=True)
        self.gamma = nn.Parameter(torch.FloatTensor([self.GAMMA]), requires_grad=True)
        self.theta = nn.Parameter(torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
                                  requires_grad=False)
        self.frequency = nn.Parameter(torch.FloatTensor([self.FREQUENCY]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def forward(self, x):

        self.kernel = self.get_gabor()

        out = F.conv2d(x, self.kernel, stride=self.stride, padding=self.padding)

        return out

    def get_gabor(self):
        x_max = self.kernel_size // 2
        y_max = self.kernel_size // 2
        x_min = -x_max
        y_min = -y_max

        k_size = x_max - x_min + 1
        x_0 = torch.arange(x_min, x_max + 1).float()
        y_0 = torch.arange(y_min, y_max + 1).float()

        x = x_0.view(-1, 1).repeat(self.channel_out, self.channel_in, 1, k_size)
        y = y_0.view(1, -1).repeat(self.channel_out, self.channel_in, k_size, 1)

        x = x.float().to(self.sigma.device)
        y = y.float().to(self.sigma.device)

        x_theta = x * torch.cos(self.theta.view(-1, 1, 1, 1)) + y * torch.sin(
            self.theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(self.theta.view(-1, 1, 1, 1)) + y * torch.cos(
            self.theta.view(-1, 1, 1, 1))

        gabor = -torch.exp(
            -0.5 * ((self.gamma * x_theta) ** 2 + y_theta ** 2) / (8 * self.sigma.view(-1, 1, 1, 1) ** 2)) \
                * torch.cos(2 * math.pi * self.frequency.view(-1, 1, 1, 1) * x_theta +
                            self.psi.view(-1, 1, 1, 1))

        gabor = gabor - gabor.mean(dim=[2, 3], keepdim=True)

        return gabor


'''  
may the force be with you.
@FileName   gabor
Created by 24 on 2024/1/26.
'''
