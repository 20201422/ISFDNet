"""
@file_name  isfdnet
@author     24
@date       2025/6/21 16:33
@version    1.0.0
freedom is the oxygen of the soul.
"""

import torch
import torch.nn.functional as F
from torch import nn

from model.ccnet import ccnet
from model.co3net import co3net
from model.component.squeeze_and_excitation import SEModule
from model.loss.arcface import ArcMarginProduct

class SpatiallyAwareSeparationModule(nn.Module):
    """
    空间感知分离模块(SASM)：将混合特征分离为身份特征和风格特征。
    INPUTS:
        in_channels：输入特征的通道数
    """
    def __init__(self, in_channels):
        super(SpatiallyAwareSeparationModule, self).__init__()

        # 生成可学习的注意力掩码
        self.mask_generator = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1)
        )

    def forward(self, mixed_feature):
        attention_logits = self.mask_generator(mixed_feature)

        masks = F.softmax(attention_logits, dim=1)
        mask_id = masks[:, 0:1, :, :]
        mask_sty = masks[:, 1:2, :, :]

        id_input = mixed_feature * mask_id
        sty_input = mixed_feature * mask_sty

        mask_overlap = torch.mean(torch.min(mask_id, mask_sty))
        mask_coverage = torch.mean(torch.max(mask_id, mask_sty))

        return id_input, sty_input, mask_id, mask_sty, mask_overlap, mask_coverage


class ChannelAttention(nn.Module):
    """
    通道注意力模块：结合通道和空间注意力来增强特征表示。
    INPUTS:
        in_channels：输入特征的通道数
    """
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.se = SEModule(channel=in_channels)

    def forward(self, x):

        feat = self.channel_attention(x)
        feat = self.se(feat)

        return feat


class FeatureDecouplingModule(nn.Module):
    """
        特征解耦模块(FDM)：接收一个混合特征向量，并将其解耦为身份特征和风格特征。
        INPUTS:
            backbone_channels：骨干网络的通道数    sf2net:64   ccnet: 96   co3net: 48
            backbone_flat_dim：骨干网络展平后的维度    sf2net:16384   ccnet:24576   co3net:17328
            id_dim：身份特征的维度，默认为2048
            style_dim：风格特征的维度，默认为2048
    """

    def __init__(self, backbone_channels, backbone_flat_dim, id_dim=2048, style_dim=2048):
        super(FeatureDecouplingModule, self).__init__()

        self.feature_complementary_separation = SpatiallyAwareSeparationModule(in_channels=backbone_channels)

        self.id_channel_attention = ChannelAttention(in_channels=backbone_channels)
        self.sty_channel_attention = ChannelAttention(in_channels=backbone_channels)

        self.fully_connection_id_1 = torch.nn.Linear(backbone_flat_dim, 4096)
        self.fully_connection_id_2 = torch.nn.Linear(4096, id_dim)
        self.fully_connection_sty_1 = torch.nn.Linear(backbone_flat_dim, 4096)
        self.fully_connection_sty_2 = torch.nn.Linear(4096, style_dim)

    def forward(self, mixed_feature):

        id_input, sty_input, mask_id, mask_sty, mask_overlap, mask_coverage = (
            self.feature_complementary_separation(mixed_feature))

        id_feat = self.id_channel_attention(id_input)
        sty_feat = self.sty_channel_attention(sty_input)

        id_feat = id_feat.view(id_feat.size(0), -1)
        sty_feat = sty_feat.view(sty_feat.size(0), -1)

        id_feat = self.fully_connection_id_1(id_feat)
        id_feat = self.fully_connection_id_2(id_feat)
        sty_feat = self.fully_connection_sty_1(sty_feat)
        sty_feat = self.fully_connection_sty_2(sty_feat)

        return id_feat, sty_feat, mask_id, mask_sty, mask_overlap, mask_coverage


class Generator(nn.Module):
    """
    生成器/解码器。
    接收身份和风格特征，重建出混合特征。
    INPUTS:
        target_channels: 目标混合特征的通道数     sf2net:64   ccnet: 96   co3net: 48
        target_size: 目标混合特征的空间尺寸    sf2net:16   ccnet: 16   co3net: 19
        id_dim：身份特征的维度，2048
        style_dim：风格特征的维度，2048
    """
    def __init__(self,  target_channels=64, target_size=16, id_dim=2048, style_dim=2048):
        super(Generator, self).__init__()

        self.target_channels = target_channels
        self.target_size = target_size

        target_flat_dim = self.target_channels * self.target_size * self.target_size  # 16384

        self.id_decoder = nn.Sequential(
            nn.Linear(id_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, target_flat_dim),
            nn.Unflatten(1, (self.target_channels, self.target_size, self.target_size))
        )

        self.style_decoder = nn.Sequential(
            nn.Linear(style_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, target_flat_dim),
            nn.Unflatten(1, (self.target_channels, self.target_size, self.target_size))
        )

        self.feature_complementary_separation = SpatiallyAwareSeparationModule(in_channels=target_channels)

        self.id_multi_dimension_attention = ChannelAttention(in_channels=target_channels)
        self.sty_multi_dimension_attention = ChannelAttention(in_channels=target_channels)

        self.refine = nn.Sequential(
            nn.Conv2d(self.target_channels, self.target_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.target_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.target_channels, self.target_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.target_channels),
        )

    def forward(self, f_id, f_sty):
        id_feature_map = self.id_decoder(f_id)
        style_feature_map = self.style_decoder(f_sty)

        fused_feature = id_feature_map + style_feature_map

        id_reconstructed, sty_reconstructed, _, _, _, _ = self.feature_complementary_separation(fused_feature)

        id_reconstructed = self.id_multi_dimension_attention(id_reconstructed)
        sty_reconstructed = self.sty_multi_dimension_attention(sty_reconstructed)

        reconstructed_feature = id_reconstructed + sty_reconstructed
        reconstructed_feature = reconstructed_feature + self.refine(reconstructed_feature)

        return reconstructed_feature

class IdentifyAndStyleFeatureDecouplingNet(nn.Module):
    """
    身份特征与风格特征解耦网络(ISFDNet)：对骨干网络的特征进行解耦。
    INPUTS:
        backbone_net：传入的掌纹识别网络作为骨干网络
        label_num：数据集中标签的个数
        weight：身份特征和风格特征的权重
        id_dim：身份特征的维度
        style_dim：风格特征的维度
    """

    def __init__(self, backbone, backbone_name: str, label_num, weight=1.0, id_dim=2048, style_dim=2048):
        super(IdentifyAndStyleFeatureDecouplingNet, self).__init__()

        _backbone_channels = {
            'ccnet': 96,
            'co3net': 48
        }
        _backbone_size = {
            'ccnet': 16,
            'co3net': 19
        }
        _backbone_flat_dim = {
            'ccnet': 24576,
            'co3net': 17328
        }

        self.backbone = backbone
        self.backbone_name = backbone_name
        if self.backbone_name not in _backbone_channels:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        self.backbone_channels = _backbone_channels[self.backbone_name]
        if self.backbone_name not in _backbone_size:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        self.backbone_size = _backbone_size[self.backbone_name]
        if self.backbone_name not in _backbone_flat_dim:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        self.backbone_flat_dim = _backbone_flat_dim[self.backbone_name]

        self.label_num = label_num
        self.weight = weight

        self.disentangler = FeatureDecouplingModule(backbone_channels=self.backbone_channels,
                                                    backbone_flat_dim=self.backbone_flat_dim,
                                                    id_dim=id_dim, style_dim=style_dim)
        self.generator = Generator(target_channels=self.backbone_channels, target_size=self.backbone_size,
                                   id_dim=id_dim, style_dim=style_dim)

        self.dropout = torch.nn.Dropout(p=0.5)

        self.arcface_id = ArcMarginProduct(in_features=id_dim, out_features=self.label_num)
        self.arcface_adv = ArcMarginProduct(in_features=id_dim, out_features=self.label_num)

    def decoupling(self, x):
        """
        完整的编码过程：图像 -> 骨干网络提特征 -> 解耦。
        """
        mixed_feature = self.backbone.processing(x)

        f_id, f_sty, mask_id, mask_sty, mask_overlap, mask_coverage = self.disentangler(mixed_feature)

        return f_id, f_sty, mixed_feature, mask_id, mask_sty, mask_overlap, mask_coverage

    def decode(self, f_id, f_sty):
        """
        解码过程。
        """
        return self.generator(f_id, f_sty)

    def forward(self, feature_tensor, target=None):
        """
        用于身份识别的前向传播。
        在训练和推理时，我们最终需要的是身份分类结果。
        """
        f_id, f_sty, mixed_feature, mask_id, mask_sty, mask_overlap, mask_coverage = self.decoupling(feature_tensor)

        embedding = f_id * self.weight + f_sty * (1 - self.weight)

        embedding_norm = F.normalize(embedding, dim=-1)

        embedding = self.dropout(embedding)

        logits = self.arcface_id(embedding, target)

        return f_id, f_sty, logits, embedding_norm, mixed_feature, mask_id, mask_sty, mask_overlap, mask_coverage

    def cycle_encode(self, mixed_feature):
        """
        循环编码：将身份特征和风格特征重新编码为混合特征。
        :param mixed_feature: 输入的混合特征张量，形状为 [batch_size, 64, 16, 16]
        :return: f_id, f_sty
        """
        f_id, f_sty, _, _, _, _ = self.disentangler(mixed_feature)

        return f_id, f_sty

    def get_feature_vector(self, feature_tensor):

        f_id, f_sty, mixed_feature, mask_id, mask_sty, mask_overlap, mask_coverage = self.decoupling(feature_tensor)

        merge_feature = f_id * self.weight + f_sty * (1 - self.weight)

        return merge_feature / torch.norm(merge_feature, p=2, dim=1, keepdim=True)


if __name__ == "__main__":
    input = torch.randn(256, 1, 128, 128)

    backbone_name = 'ccnet'  # 可选 'ccnet', 'co3net'

    backbone_map = {
        'ccnet': lambda: ccnet(num_classes=600, weight=0.8),
        'co3net': lambda: co3net(num_classes=600)
    }
    if backbone_name not in backbone_map:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    backbone_net = backbone_map[backbone_name]()

    net = IdentifyAndStyleFeatureDecouplingNet(backbone=backbone_net, backbone_name=backbone_name, label_num=600, weight=1.0)

    id, sty, o, fe, m, _, _, _, _ = net(input, target=torch.tensor([1]))
    print(id.shape, sty.shape, o.shape, fe.shape, m.shape)



"""
@file_name  isfdnet
Created by 24 on 2025/6/21
"""
