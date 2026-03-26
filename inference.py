"""
@file_name  inference
@author     24
@date       2025/11/13 15:50
@version    1.0.0
freedom is the oxygen of the soul.
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms as T
import numpy as np

from model.ccnet import ccnet
from model.co3net import co3net
from model.isfdnet import IdentifyAndStyleFeatureDecouplingNet
from utils.data_set import NormSingleROI


def main(args):
    """
    主推理函数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    transform = T.Compose([
        T.Resize(args.image_size),
        T.ToTensor(),
        NormSingleROI(outchannels=1)
    ])

    backbone_map = {
        'ccnet': lambda: ccnet(num_classes=args.label_num, weight=0.7),
        'co3net': lambda: co3net(num_classes=args.label_num)
    }
    if args.backbone not in backbone_map:
        raise ValueError(f"不支持的骨干网络: {args.backbone}, 可选项: 'ccnet', 'co3net'")

    backbone_net = backbone_map[args.backbone]()

    model = IdentifyAndStyleFeatureDecouplingNet(
        backbone=backbone_net,
        backbone_name=args.backbone,
        label_num=args.label_num
    ).to(device)

    if args.model_path:
        try:
            state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
            if 'module.' in list(state_dict.keys())[0]:
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print(f"成功加载模型权重: {args.model_path}")
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            return
    else:
        print("警告: 未提供模型路径，将使用随机初始化的模型进行推理。")

    model.eval()

    try:
        image1 = Image.open(args.image_path1).convert('L')
        image2 = Image.open(args.image_path2).convert('L')
    except FileNotFoundError as e:
        print(f"错误: 找不到图像文件 {e.filename}")
        return

    image_tensor1 = transform(image1).unsqueeze(0).to(device)
    image_tensor2 = transform(image2).unsqueeze(0).to(device)

    with torch.no_grad():
        feature_vector1 = model.get_feature_vector(image_tensor1)
        feature_vector2 = model.get_feature_vector(image_tensor2)

    feature1_np = feature_vector1.cpu().numpy().flatten()
    feature2_np = feature_vector2.cpu().numpy().flatten()

    matching_score = np.arccos(np.clip(np.dot(feature1_np, feature2_np), -1, 1)) / np.pi

    print("\n--- 推理结果 ---")
    print(f"图像 1: {args.image_path1}")
    print(f"图像 2: {args.image_path2}")
    print(f"匹配得分: {matching_score}")
    print("(得分越接近0，表示匹配度越高)")  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ISFDNet 模型推理脚本')
    parser.add_argument('--model_path', type=str,
                        default='./results/9.4/ccnet/PolyU/model/best_model_params.pth',
                        help='预训练模型的路径 (.pth文件)')
    parser.add_argument('--image_path1', type=str,
                        default='/disk01/lyl/datasets/PolyU/PalmBigDataBase_zq/P_F_102_7.bmp',
                        help='第一张需要推理的图像的路径')
    parser.add_argument('--image_path2', type=str,
                        default='/disk01/lyl/datasets/PolyU/PalmBigDataBase_zq/P_F_112_5.bmp',
                        help='第二张需要推理的图像的路径')
    parser.add_argument('--backbone', type=str, default='ccnet', choices=['ccnet', 'co3net'],
                        help='模型使用的骨干网络')
    parser.add_argument('--label_num', type=int, default=386,
                        help="Tongji: 600 PolyU 386 IITD: 460 Multi-Spec 500")
    parser.add_argument('--image_size', type=int, default=128, help='输入图像的尺寸')

    args = parser.parse_args()

    main(args)


"""
@file_name  inference
Created by 24 on 2025/11/13
"""