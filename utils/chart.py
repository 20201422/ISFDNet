"""
@FileName   chart
@Author  24
@Date    2024/1/22 23:05
@Version 1.0.0
freedom is the oxygen of the soul.
"""
import math
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics
from sklearn.metrics import auc

import matplotlib

from utils.file_util import *

matplotlib.use('Agg')  # 使用非交互式后端

import matplotlib.pyplot as plt

from utils import *


# 图表生成（汉明距离图、ROC曲线图）以及获取阈值和Equal Error Rate(EER)
class Chart:
    """图表生成（汉明距离图、ROC曲线图）以及获取阈值和Equal Error Rate(EER)::
        INPUTS:
            genuine_list：类内匹配分数
            imposter_list：类间匹配分数
            losses：训练的损失值数组
            accuracies：训练的精确度数组
            chart_path：保存图表路径
            file_path：保存文件路径
    """
    def __init__(self, genuine_list=None, imposter_list=None, chart_path=None, file_path=None, loss=None, accuracy=None):
        super(Chart, self).__init__()
        self.genuine_list = genuine_list  # 类内匹配分数
        self.imposter_list = imposter_list    # 类间匹配分数
        self.losses = loss    # 训练的损失值数组
        self.accuracies = accuracy    # 训练的精确度数组
        self.chart_path = chart_path    # 保存图表路径
        self.file_path = file_path    # 保存文件路径

    # 绘制汉明距离图
    def get_hamming_distance_chart(self):
        genuine_min = np.min(self.genuine_list)
        imposter_min = np.min(self.imposter_list)
        genuine_max = np.max(self.genuine_list)
        imposter_max = np.max(self.imposter_list)
        genuine_mean = np.mean(self.genuine_list)
        imposter_mean = np.mean(self.imposter_list)
        genuine_std = np.std(self.genuine_list)
        imposter_std = np.std(self.imposter_list)

        print('genuine  (min, max, mean, std): [%f, %f] [%f +- %f]' % (
        genuine_min, genuine_max, genuine_mean,genuine_std))
        print('imposter (min, max, mean, std): [%f, %f] [%f +- %f]' % (
        imposter_min, imposter_max, imposter_mean, imposter_std))
        print('scores loading done!')

        samples = 100

        genuine_score = (self.genuine_list - genuine_min) / (genuine_max - genuine_min) * samples
        imposter_score = (self.imposter_list - imposter_min) / (imposter_max - imposter_min) * samples

        genuine_y = np.zeros((samples + 1, 1), dtype='int32')
        imposter_y = np.zeros((samples + 1, 1), dtype='int32')
        genuine_y = genuine_y[:, 0]
        imposter_y = imposter_y[:, 0]
        for i in genuine_score:
            i = int(round(i))
            genuine_y[i] += 1
        for i in imposter_score:
            i = int(round(i))
            imposter_y[i] += 1
        genuine_y = np.array(genuine_y)
        imposter_y = np.array(imposter_y)

        genuine_sum = np.sum(genuine_y)
        genuine_y = genuine_y / genuine_sum * 100
        imposter_sum = np.sum(imposter_y)
        imposter_y = imposter_y / imposter_sum * 100

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.linspace(0, 1, samples + 1) * (genuine_max - genuine_min) + genuine_min, genuine_y, 'b',
                 label='Genuine')
        ax.plot(np.linspace(0, 1, samples + 1) * (imposter_max - imposter_min) + imposter_min, imposter_y, 'r',
                 label='Impostor')

        ax.legend(loc='upper right', fontsize=13)
        ax.set_xlabel('Matching Score', fontsize=13)
        ax.set_ylabel('Percentage (%)', fontsize=13)
        ax.set_ylim([0, 1.2 * np.max([genuine_y.max(), imposter_y.max()])])
        ax.grid(True)
        save_fp = os.path.join(self.chart_path, 'hamming_distance_chart.svg')
        fig.savefig(save_fp, bbox_inches='tight')
        plt.close(fig)

        with open(os.path.join(self.file_path, 'matching_score_distr.txt'), 'w') as f:
            f.writelines('[min, max] [mean +- std]\n')
            f.writelines('genuine: [%.10f, %.10f] [%.10f +- %.10f]\n' % (genuine_min, genuine_max, genuine_mean, genuine_std))
            f.writelines('imposter: [%.10f, %.10f] [%.10f +- %.10f]\n' % (imposter_min, imposter_max, imposter_mean, imposter_std))
            f.writelines('number of genuine matching:  %d\n' % self.genuine_list.shape)
            f.writelines('number of impostor matching: %d\n' % self.imposter_list.shape)

    def get_far_gar_eer_thresholds_accuracy(self):
        # 计算平均值
        mean_genuine = self.genuine_list.mean()
        mean_imposter = self.imposter_list.mean()
        if mean_genuine < mean_imposter:
            self.genuine_list = -self.genuine_list
            self.imposter_list = -self.imposter_list

        y = np.vstack((np.ones((len(self.genuine_list), 1)), np.zeros((len(self.imposter_list), 1))))
        scores = np.vstack((self.genuine_list.reshape(-1, 1), self.imposter_list.reshape(-1, 1)))
        false_acceptance_rates, genuine_acceptance_rates, thresholds = metrics.roc_curve(y, scores, pos_label=1)

        roc_auc = auc(false_acceptance_rates, genuine_acceptance_rates) * 100

        equal_error_rate = brentq(lambda x: 1. - x - interp1d(false_acceptance_rates, genuine_acceptance_rates)(x), 0.,
                                  1.)
        thresh = interp1d(false_acceptance_rates, thresholds)(equal_error_rate)

        false_acceptance_rates = false_acceptance_rates * 100
        genuine_acceptance_rates = genuine_acceptance_rates * 100
        equal_error_rate = equal_error_rate * 100

        if mean_genuine < mean_imposter:
            thresh = -thresh
            thresholds = -thresholds

        with open(os.path.join(self.file_path, 'eer_th_auc.txt'), 'w') as f:
            f.writelines('%.10f %.4f %.10f\n' % (equal_error_rate, thresh, roc_auc))
        write_file_for_npy(self.file_path, 'false_acceptance_rates.npy', false_acceptance_rates)
        write_file_for_npy(self.file_path, 'genuine_acceptance_rates.npy', genuine_acceptance_rates)
        print('eer: %f%% thresh: %f accuracy: %f%%' % (equal_error_rate, thresh, roc_auc))

        return false_acceptance_rates, genuine_acceptance_rates

    def get_gar_at_far(self, far, gar, target_far=0.00001):
        """
        计算在指定 FAR（如 0.001%）下的 GAR 值     GAR@0.001%FAR
        Args:
            far: 错误接受率数组（FAR）
            gar: 真实接受率数组（GAR）
            target_far: 目标 FAR（默认 0.001%）
        Returns:
            gar_at_far: 在目标 FAR 下的 GAR 值
        """
        far, gar = zip(*sorted(zip(far, gar)))
        far = np.array(far)
        gar = np.array(gar)

        if target_far < far.min():
            gar_at_far = gar[far.argmin()]
        elif target_far > far.max():
            gar_at_far = gar[far.argmax()]
        else:
            interp_func = interp1d(far, gar, kind='linear')
            gar_at_far = float(interp_func(target_far))

        print('GAR@' + str(target_far * 100) + '%FAR: ', gar_at_far)
        write_file_for_txt_one(self.file_path, 'gar@' + str(target_far * 100) + '%far.txt', gar_at_far)

    def get_receiver_operating_characteristic(self, false_acceptance_rates, genuine_acceptance_rates):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(false_acceptance_rates, genuine_acceptance_rates, label='ROC')
        ax.set_ylim(96, 101)
        ax.set_xscale('log')
        ax.legend(loc='best')
        ax.set_xlabel('False Acceptance Rate(%)')
        ax.set_ylabel('Genuine Acceptance Rate(%)')
        save_fp = os.path.join(self.chart_path, 'roc.svg')
        fig.savefig(save_fp, bbox_inches='tight')
        plt.close(fig)

    def get_losses_change(self):
        if self.losses is None:
            return
        epochs = list(range(1, len(self.losses) + 1))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, self.losses, label='Loss')
        ax.legend()
        ax.set_title('Loss over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Value')
        ax.set_yscale('symlog', linthresh=0.1)
        save_fp = os.path.join(self.chart_path, 'loss_over_epochs.svg')
        fig.savefig(save_fp, bbox_inches='tight')
        plt.close(fig)

    def get_accuracies_change(self):
        if self.accuracies is None:
            return
        epochs = list(range(1, len(self.accuracies) + 1))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, self.accuracies, label='Accuracy')
        ax.legend()
        ax.set_title('Accuracy over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Value')
        save_fp = os.path.join(self.chart_path, 'accuracy_over_epochs.svg')
        fig.savefig(save_fp, bbox_inches='tight')
        plt.close(fig)

    def get_chart(self):
        os.makedirs(self.chart_path, exist_ok=True)
        os.makedirs(self.file_path, exist_ok=True)

        plt.close('all')

        if self.losses is not None:
            self.get_losses_change()
        if self.accuracies is not None:
            self.get_accuracies_change()

        self.get_hamming_distance_chart()
        false_acceptance_rates, genuine_acceptance_rates = self.get_far_gar_eer_thresholds_accuracy()
        self.get_gar_at_far(false_acceptance_rates, genuine_acceptance_rates, target_far=0.00001)
        self.get_receiver_operating_characteristic(false_acceptance_rates, genuine_acceptance_rates)

    def get_chart_for_train(self):
        # 训练过程调用时也确保独立图
        plt.close('all')
        self.get_losses_change()  # 绘制损失值曲线图
        self.get_accuracies_change()  # 绘制精确度曲线图


if __name__ == '__main__':
    genuine, imposter = (np.load('/home/hipeson/lyl/Program/ISFDNet/results/IITD/file/genuine_matching_score.npy'),
                         np.load('/home/hipeson/lyl/Program/ISFDNet/results/IITD/file/imposter_matching_score.npy'))
    losses = read_file_for_txt('/home/hipeson/lyl/Program/ISFDNet/results/IITD/model/', 'train_losses.txt')
    accuracies = read_file_for_txt('/home/hipeson/lyl/Program/ISFDNet/results/IITD/model/', 'train_accuracies.txt')

    Chart(genuine_list=genuine, imposter_list=imposter, loss=losses, accuracy=accuracies,
          chart_path='/home/hipeson/lyl/Program/ISFDNet/results/IITD/chart/',
          file_path='/home/hipeson/lyl/Program/ISFDNet/results/IITD/file/').get_chart_for_train()

'''  
may the force be with you.
@FileName   chart
Created by 24 on 2024/1/22.
'''
