"""
@FileName   verify
@Author  24
@Date    2024/1/22 23:01
@Version 1.0.0
freedom is the oxygen of the soul.
"""
from torch import nn
from tqdm import tqdm

from utils.file_util import *


# 模型验证
class Verify(nn.Module):
    """模型验证::
        INPUTS:
            model：要验证的网络模型
            train_data：训练的数据集
            test_data：要测试的数据集
            path：验证结果的存放路径
    """
    def __init__(self, model, train_data, test_data, path):
        super(Verify, self).__init__()
        self.model = model  # 要验证的网络模型
        self.train_data = train_data    # 训练的数据集
        self.test_data = test_data    # 要验证的数据集
        self.path = path    # 验证结果的存放路径

    def verify(self):
        self.model.eval()
        self.model.cuda()

        features_verify, targets_verify = [], []
        genuine_matching_score, imposter_matching_score = [], []

        print('Begin feature extraction...')

        for batch_id, (datas, target, _) in enumerate(self.test_data):
            # 获取图像张量
            data = datas[0]
            data = data.cuda()
            # 得到图像的标签
            target = target[0]

            feature = self.model.get_feature_vector(data)
            feature = feature.cpu().detach().numpy()

            features_verify = np.concatenate([features_verify, feature]) if len(features_verify) > 0 else feature
            targets_verify = np.concatenate([targets_verify, target]) if len(targets_verify) > 0 else target
        print('Completed feature extraction!')

        print('Begin verifying the model for feature matching...')
        for i in tqdm(range(features_verify.shape[0]), desc="feature matching", unit=""):
            for j in range(i + 1, features_verify.shape[0]):

                matching_score = np.arccos(np.clip(np.dot(features_verify[i], features_verify[j]), -1, 1)) / np.pi

                if targets_verify[i] == targets_verify[j]:
                    genuine_matching_score.append(matching_score)
                else:
                    imposter_matching_score.append(matching_score)

        write_file_for_txt(self.path, 'genuine_matching_score.txt', genuine_matching_score)
        write_file_for_txt(self.path, 'imposter_matching_score.txt', imposter_matching_score)
        write_file_for_npy(self.path, 'genuine_matching_score.npy', genuine_matching_score)
        write_file_for_npy(self.path, 'imposter_matching_score.npy', imposter_matching_score)
        print('End verifying the model for feature matching!')


'''  
may the force be with you.
@FileName   verify
Created by 24 on 2024/1/22.
'''
