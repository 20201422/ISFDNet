"""
@file_name  cross_experiment
@author     24
@date       2024/10/12 11:27
@version    1.0.0
freedom is the oxygen of the soul.
"""

import argparse
import datetime

import torch
from torch.utils.data import DataLoader

from model.isfdnet import IdentifyAndStyleFeatureDecouplingNet
from utils.chart import Chart
from utils.file_util import *
from utils.data_set import MyDataset
from verify import Verify


parser = argparse.ArgumentParser(
    description="ISFDNet Cross Experiment"
)
parser.add_argument("--source_id", type=str, default="Tongji")
parser.add_argument("--target_id", type=str, default="IITD")
parser.add_argument("--label_num", type=int, default=600)
parser.add_argument("--save_path", type=str, default="./results/")
parser.add_argument("--source_model", type=str, default="./results/isfdnet/Tongji/model/best_model.pth")
args = parser.parse_args()

'''
参数配置

gpu_id：训练图形处理器的id
source_id：源id
target_id：目标id
label_num：数据集中标签的个数 Tongji: 600 PolyU 386 IITD: 460 Multi-Spec 500
train_file：训练文本文件路径
test_file：测试文本文件路径
verify_file：验证文本文件路径
source_model_path：源模型路径
model_path：保存模型的路径
file_path：保存文件的路径
chart_path：保存图表的路径
batch_size：批量大小，指每次迭代训练时所使用的样本数量
'''
gpu_id = '0'  # 训练图形处理器的id
source_id = args.source_id  # 源id
target_id = args.target_id    # 目标id
label_num = args.label_num  # 数据集中标签的个数（更改数据集必须修改这个）
source_model = args.source_model  # 源模型路径
train_file = './data/' + target_id + '/train_' + target_id + '.txt'  # 训练文本文件路径
test_file = './data/' + target_id + '/test_' + target_id + '.txt'  # 测试文本文件路径
verify_file = './data/' + target_id + '/verify_' + target_id + '.txt'  # 验证文本文件路径
all_database_path = './data/all_verify.txt'  # 所有数据集的验证文本文件路径
model_path = args.save_path + source_id + '-' + target_id + '/model/'  # 保存模型的路径
file_path = args.save_path + source_id + '-' + target_id + '/file/'  # 保存文件的路径
chart_path = args.save_path + source_id + '-' + target_id + '/chart/'  # 保存图表的路径
batch_size = 500  # 批量大小，指每次迭代训练时所使用的样本数量


if __name__ == '__main__':
    print('\n----Start Of Program----')
    print("source id is {}, target id is {}."
          .format(source_id, target_id))
    print('Now time is', datetime.datetime.now())

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(file_path, exist_ok=True)
    os.makedirs(chart_path, exist_ok=True)

    print('Load data...')
    train_data = MyDataset(txt=train_file, database_name=target_id,
                           label_num=label_num, transforms=None, train=True, image_size=128, out_channels=1)
    test_data = MyDataset(txt=test_file, database_name=target_id,
                          label_num=label_num, transforms=None, train=False, image_size=128, out_channels=1)
    verify_data = MyDataset(txt=verify_file, database_name=target_id,
                            label_num=label_num, transforms=None, train=False, image_size=128, out_channels=1)

    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=2, shuffle=True)
    test_data_loader = DataLoader(dataset=test_data, batch_size=128, num_workers=2, shuffle=True)
    verify_data_loader = DataLoader(dataset=verify_data, batch_size=1024, num_workers=2)
    print('Load data end!\n')

    print('Get model..')
    check_file_exist(source_model)
    model = torch.load(source_model)

    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    model.cuda()

    print('\n----Verify With The Best Model----')
    verify_start_time = datetime.datetime.now()
    print('Now time is', verify_start_time)
    Verify(model=model, train_data=train_data_loader, test_data=test_data_loader, path=file_path).verify()
    verify_end_time = datetime.datetime.now()
    print('Now time is', verify_end_time)
    print('Verify total time is:', verify_end_time - verify_start_time)
    print('\n----Verify End----')

    print('\n----Get Char And EER And Threshold Value----')
    genuine = read_file_for_npy(file_path, 'genuine_matching_score.npy')
    imposter = read_file_for_npy(file_path, 'imposter_matching_score.npy')
    Chart(genuine_list=genuine, imposter_list=imposter, loss=None, accuracy=None,
          chart_path=chart_path, file_path=file_path).get_chart()
    print('\n----End Charting----')

    print('\nNow time is', datetime.datetime.now())
    print('\n----End Of Program----')


"""
@file_name  cross_experiment
Created by 24 on 2024/10/12
"""
