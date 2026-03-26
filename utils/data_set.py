# -*- coding:utf-8 -*-
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils import data
from torchvision import transforms as T

from utils.low_frequency_disturbance import *


class NormSingleROI(object):
    """
    Normalize the input image (exclude the black region) with 0 mean and 1 std.
    [c,h,w]
    """

    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):

        # if not T.functional._is_tensor_image(tensor):
        #     raise TypeError('tensor is not a torch image.')

        c, h, w = tensor.size()

        if c != 1:
            raise TypeError('only support graysclae image.')

        # print(tensor.size)

        tensor = tensor.view(c, h * w)
        idx = tensor > 0
        t = tensor[idx]

        # print(t)
        m = t.mean()
        s = t.std()
        t = t.sub_(m).div_(s + 1e-6)
        tensor[idx] = t

        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats=self.outchannels, dim=0)

        return tensor


class MyDataset(data.Dataset):
    """
    Load and process the ROI images::

    INPUT::
    txt: a text file containing pathes & labels of the input images \n
    transforms: None
    train: True for a training set, and False for a testing set
    imside: the image size of the output image [imside x imside]
    outchannels: 1 for grayscale image, and 3 for RGB image

    OUTPUT::
    [batch, outchannels, imside, imside]
    """

    def __init__(self, txt, database_name, label_num,
                 transforms=None, train=True, image_size=128, out_channels=1):
        """
            初始化函数，用于创建Dataset对象

            参数：
            txt：str，文本文件路径
            transforms：torchvision.transforms对象，数据预处理的转换操作，默认为None
            train：bool，是否为训练模式，默认为True
            imside：int，图像的尺寸，默认为128
            out_channels：int，输出图像的通道数，默认为1
        """
        self.train = train

        self.image_size = image_size  # 128, 224
        self.out_channels = out_channels  # 1, 3

        self.text_path = txt
        self.database_name = database_name
        self.label_num = label_num

        self.transforms = transforms

        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.Resize(self.image_size),
                    T.ToTensor(),
                    NormSingleROI(outchannels=self.out_channels)
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(self.image_size),
                    T.RandomChoice(transforms=[
                        T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),  # 0.3 0.35
                        T.RandomResizedCrop(size=self.image_size, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                        T.RandomPerspective(distortion_scale=0.15, p=1),  # (0.1, 0.2) (0.05, 0.05)
                        T.RandomChoice(transforms=[
                            T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BICUBIC, expand=False,
                                             center=(0.5 * self.image_size, 0.0)),
                            T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BICUBIC, expand=False,
                                             center=(0.0, 0.5 * self.image_size)),
                        ]),
                    ]),

                    T.ToTensor(),
                    NormSingleROI(outchannels=self.out_channels)
                ])

        self._read_txt_file()

    def _read_txt_file(self):
        self.images_path = []
        self.images_label = []

        txt_file = self.text_path

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_label.append(item[1])

    def __getitem__(self, index):
        img_path_anchor = self.images_path[index]
        label_anchor = self.images_label[index]

        positive_indices = np.where(np.array(self.images_label) == label_anchor)[0]
        negative_indices = np.where(np.array(self.images_label) != label_anchor)[0]

        if self.train:
            positive_index = np.random.choice(positive_indices[positive_indices != index])
            negative_index = np.random.choice(negative_indices[negative_indices != index])
        else:
            positive_index = index
            negative_index = index

        img_path_positive = self.images_path[positive_index]

        anchor_img = Image.open(img_path_anchor).convert('L')
        anchor = self.transforms(anchor_img)

        positive_img = Image.open(img_path_positive).convert('L')
        positive = self.transforms(positive_img)

        negative_img = Image.open(self.images_path[negative_index]).convert('L')
        negative = self.transforms(negative_img)

        if self.train:
            low_freq_shifted, high_freq_shifted = fft_decompose(positive_img)
            perturbed_low_freq_shifted = perturb_low_frequency(low_freq_shifted, mode='scale_noise_illumination')
            stylized_anchor_np = fft_reconstruct(perturbed_low_freq_shifted, high_freq_shifted)
            stylized_anchor = Image.fromarray(stylized_anchor_np.astype(np.uint8)).convert('L')
            stylized_anchor = self.transforms(stylized_anchor)
        else:
            stylized_anchor = Image.open(img_path_anchor).convert('L')
            stylized_anchor = self.transforms(stylized_anchor)

        data = [anchor, positive, negative, stylized_anchor]
        label = [int(label_anchor), int(self.images_label[positive_index]), int(self.images_label[negative_index]),
                 int(label_anchor)]
        # print(data)
        # print(label)

        return data, label, img_path_anchor

    # 用于获取数据集的长度
    def __len__(self):
        return len(self.images_path)
