"""
@FileName   arcface
@Author  24
@Date    2024/1/26 22:35
@Version 1.0.0
freedom is the oxygen of the soul.
"""
import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """实现大边距弧度距离的类 Implement of large margin arc distance::
        Args:
            in_features: size of each input sample  每个输入样本的大小
            out_features: size of each output sample    每个输出样本的大小
            s: norm of input feature    输入特征的范数
            m: margin   边距

            cos(theta + m)

        From: https://github.com/ronghuaiyang/arcface-pytorch
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(self.out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if self.training:
            assert label is not None

            if input.size(1) != self.in_features:
                raise ValueError(f"Input feature size ({input.size(1)}) does not match ArcFace in_features ({self.in_features}).")
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

            one_hot = torch.zeros(cosine.size(), device=cosine.device)



            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            output = self.s * cosine

        return output


'''  
may the force be with you.
@FileName   arcface
Created by 24 on 2024/1/26.
'''
