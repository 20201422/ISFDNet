"""
@FileName   main
@Author  24
@Date    2024/1/22 23:01
@Version 1.0.0
freedom is the oxygen of the soul.
"""
import argparse
import datetime
import io

import math
import torch
import torch.optim as optim
from pytorch_lightning import seed_everything
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from model.ccnet import ccnet
from model.co3net import co3net
from model.isfdnet import IdentifyAndStyleFeatureDecouplingNet

from train import Train
from utils.chart import Chart
from utils.data_set import MyDataset
from utils.file_util import *
from utils.reproducibility import get_dataloader_generator, seed_worker
from verify import Verify

parser = argparse.ArgumentParser(
    description="ISFDNet"
)
parser.add_argument("--database_name", type=str, default="Tongji")
parser.add_argument("--label_num", type=int, default=600,
                    help="Tongji: 600 PolyU 386 IITD: 460 Multi-Spec 500")
parser.add_argument("--one_label_intra_class_num", type=int, default=20)
parser.add_argument("--vit_floor_num", type=int, default=10)
parser.add_argument("--weight_for_backbone", type=float, default=0.7)
parser.add_argument("--weight_for_id_and_sty", type=float, default=1.0)
parser.add_argument("--loss_ce", type=float, default=0.8)
parser.add_argument("--loss_tl", type=float, default=0.2)
parser.add_argument("--loss_id_weight", type=float, default=1.0)
parser.add_argument("--loss_rec_weight", type=float, default=0.05)
parser.add_argument("--loss_cycle_weight", type=float, default=0.03)
parser.add_argument("--loss_ortho_weight", type=float, default=0.12)
parser.add_argument("--loss_id_adversarial_weight", type=float, default=0.4)
parser.add_argument("--loss_id_consistency_weight", type=float, default=0.02)
parser.add_argument("--backbone_name", type=str, default="ccnet", help="ccnet, co3net")
parser.add_argument("--run_type", type=str, default="train")
parser.add_argument("--train_file", type=str, default="./data/Tongji/train_Tongji.txt")
parser.add_argument("--test_file", type=str, default="./data/Tongji/test_Tongji.txt")
parser.add_argument("--verify_file", type=str, default="./data/Tongji/verify_Tongji.txt")
parser.add_argument("--save_path", type=str, default="./results/")
args = parser.parse_args()

'''
参数配置

gpu_id：训练图形处理器的id
database_name：数据集名称
train_file：训练文本文件路径
test_file：测试文本文件路径
verify_file：验证文本文件路径
all_database_path：所有数据集的验证文本文件路径
model_path：保存模型的路径
file_path：保存文件的路径
chart_path：保存图表的路径
label_num：数据集中标签的个数 Tongji: 600 PolyU 386 IITD: 460 Multi-Spec 500
one_label_intra_class_num：数据集中一个标签（一组类内）的数量 Tongji: 20 PolyU 10 IITD: 6 Multi-Spec 12
loss_ce_weight：交叉熵损失函数权重
loss_tl_weight：三元组损失函数权重
loss_id_weight：身份损失函数权重
loss_rec_weight：重建损失函数权重
loss_cycle_weight：循环一致性损失函数权重
loss_ortho_weight：正交损失函数权重
loss_id_adversarial_weight：身份对抗损失函数权重
loss_id_consistency_weight：身份表征一致性损失权重
backbone_name：backbone的名称，支持'ccnet'和'co3net'
vit_floor_num：vit层数
weight_for_backbone：backbone中的权重
weight_for_id_and_sty：身份特征和风格特征的权重
epochs：训练迭代次数
batch_size：批量大小，指每次迭代训练时所使用的样本数量
lr：初始学习率
step：学习调度的步长
run_type：运行类型 “train”、“verify”和“train_with_existing_model”，为训练+验证、仅验证和用已有模型训练，默认为“train”
'''
gpu_id = '0'  # 训练图形处理器的id
database_name = args.database_name  # args.database_name    # 数据集名称
train_file = args.train_file  # 训练文本文件路径
test_file = args.test_file  # 测试文本文件路径
verify_file = args.verify_file  # 验证文本文件路径
model_path = args.save_path + database_name + '/model/'  # 保存模型的路径
file_path = args.save_path + database_name + '/file/'  # 保存文件的路径
chart_path = args.save_path + database_name + '/chart/'  # 保存图表的路径
all_database_path = './data/all_verify.txt'  # 所有数据集的验证文本文件路径
label_num = args.label_num  # 数据集中标签的个数（更改数据集必须修改这个）
one_label_intra_class_num = args.one_label_intra_class_num  # 数据集中一个标签（一组类内类内）的数量（更改数据集必须修改这个）
weight_for_backbone = args.weight_for_backbone  # backbone中的权重
weight_for_id_and_sty = args.weight_for_id_and_sty  # 身份特征和风格特征的权重
loss_ce_weight = args.loss_ce  # 交叉熵损失函数权重
loss_tl_weight = args.loss_tl    # 三元组损失函数权重
loss_id_weight = args.loss_id_weight  # 身份损失函数权重
loss_rec_weight = args.loss_rec_weight  # 重建损失函数权重
loss_cycle_weight = args.loss_cycle_weight  # 循环一致性损失函数权重
loss_ortho_weight = args.loss_ortho_weight  # 正交损失函数权重
loss_id_adversarial_weight = args.loss_id_adversarial_weight  # 身份对抗损失函数权重
loss_id_consistency_weight = args.loss_id_consistency_weight    # 身份表征一致性损失权重
backbone_name = args.backbone_name  # backbone的名称，支持'ccnet'和'co3net'
epochs = 1000  # 训练迭代次数
batch_size = 256  # 批大小
lr = 0.0005  # 初始学习率
step = 50  # 学习调度的步长
run_type = args.run_type  # 运行类型
seed = 3407    # 种子

if __name__ == '__main__':
    print('\n----Start Of Program----')
    print("gpu_id: {} \tdatabase_name: {} \ntrain_file: {} \ntest_file: {} \nverify_file: {} \nmodel_path: {} "
          "\nfile_path: {} \nchart_path: {} \nlabel_num: {} \tone_label_intra_class_num: {} "
          "\nloss_ce_weight: {} \tloss_tl_weight: {} \tweight_for_backbone: {} "
          "\t weight_for_id_and_sty: {}"
          "\nloss_id_weight: {} \tloss_rec_weight: {} \tloss_cycle_weight: {} \tloss_ortho_weight: {} "
          "\tloss_id_adversarial_weight: {} \tloss_id_consistency_weight: {}"
          "\nbackbone: {} \tepochs: {} \tbatch_size: {} \tlr: {} \tstep: {} \n"
          .format(gpu_id, database_name, train_file, test_file, verify_file, model_path, file_path, chart_path,
                  label_num, one_label_intra_class_num, loss_ce_weight, loss_tl_weight,
                  weight_for_backbone, weight_for_id_and_sty, loss_id_weight,  loss_rec_weight, loss_cycle_weight,
                  loss_ortho_weight, loss_id_adversarial_weight, loss_id_consistency_weight,
                  backbone_name, epochs, batch_size, lr, step))
    print('Now time is', datetime.datetime.now())

    seed_everything(seed)

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(file_path, exist_ok=True)
    os.makedirs(chart_path, exist_ok=True)

    print('Load data...')
    train_data = MyDataset(txt=train_file, database_name=database_name,
                           label_num=label_num, transforms=None, train=True, image_size=128, out_channels=1)
    test_data = MyDataset(txt=test_file, database_name=database_name,
                          label_num=label_num, transforms=None, train=False, image_size=128, out_channels=1)

    g = get_dataloader_generator(seed)
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=4, shuffle=True,
                                   worker_init_fn=seed_worker, generator=g
                                   )
    test_data_loader = DataLoader(dataset=test_data, batch_size=128, num_workers=2, shuffle=False,
                                  worker_init_fn=seed_worker, generator=g
                                  )
    print('Load data end!\n')

    print('Get model..')
    backbone_map = {
        'ccnet': lambda: ccnet(num_classes=label_num, weight=weight_for_backbone),
        'co3net': lambda: co3net(num_classes=label_num)
    }
    if backbone_name not in backbone_map:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    backbone_net = backbone_map[backbone_name]()
    model = IdentifyAndStyleFeatureDecouplingNet(backbone=backbone_net, backbone_name=backbone_name,
                                                 label_num=label_num, weight=weight_for_id_and_sty)

    if run_type == 'verify':
        temporary_model_path = model_path + '/best_model.pth'
        check_file_exist(temporary_model_path)
        model = torch.load(temporary_model_path)
    if run_type == 'train_with_existing_model':
        temporary_model_path = model_path + '/best_model_params.pth'
        check_file_exist(temporary_model_path)
        model.load_state_dict(torch.load(temporary_model_path))

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    model.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.8)
    print('End of preparation!')

    if run_type == 'train' or run_type == 'train_with_existing_model':
        print('\n----Begin Train----')
        train_start_time = datetime.datetime.now()
        print('Now time is', train_start_time)
        _, _ = Train(model=model, epochs=epochs, optimizer=optimizer, scheduler=scheduler,
                     train_data_loader=train_data_loader, test_data_loader=test_data_loader, model_path=model_path,
                     loss_ce_weight=loss_ce_weight, loss_tl_weight=loss_tl_weight, loss_id_weight=loss_id_weight,
                     loss_rec_weight=loss_rec_weight, loss_cycle_weight=loss_cycle_weight,
                     loss_ortho_weight=loss_ortho_weight,
                     loss_id_adversarial_weight=loss_id_adversarial_weight,
                     loss_id_consistency_weight=loss_id_consistency_weight).epoch_train()
        train_end_time = datetime.datetime.now()
        print('Now time is', train_end_time)

        time_diff = train_end_time - train_start_time
        print('Training total time is:', time_diff)
        print('\n----Train End----')

        temporary_model_path = model_path + '/best_model.pth'
        check_file_exist(temporary_model_path)
        model = torch.load(temporary_model_path, weights_only=False)

    print('\n----Verify With The Best Model----')
    verify_start_time = datetime.datetime.now()
    print('Now time is', verify_start_time)
    model.cuda()
    Verify(model=model, train_data=train_data_loader, test_data=test_data_loader,path=file_path).verify()
    verify_end_time = datetime.datetime.now()
    print('Now time is', verify_end_time)
    time_diff = verify_end_time - verify_start_time
    print('Verify total time is:', time_diff)
    print('\n----Verify End----')

    print('\n----Get Char And EER And Threshold Value----')
    genuine = read_file_for_npy(file_path, 'genuine_matching_score.npy')
    imposter = read_file_for_npy(file_path, 'imposter_matching_score.npy')
    losses = read_file_for_txt(model_path, 'train_losses.txt')
    accuracies = read_file_for_txt(model_path, 'train_accuracies.txt')
    Chart(genuine_list=genuine, imposter_list=imposter, loss=losses, accuracy=accuracies,
          chart_path=chart_path, file_path=file_path).get_chart()
    print('\n----End Charting----')

    print('\nNow time is', datetime.datetime.now())
    print('\n----End Of Program----')

'''  
may the force be with you.
@FileName   main
Created by 24 on 2024/1/22.
'''
