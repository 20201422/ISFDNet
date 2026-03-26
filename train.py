"""
@FileName   train
@Author  24
@Date    2024/1/22 23:01
@Version 1.0.0
freedom is the oxygen of the soul.
"""
import copy

import math

from model.ccnet import ccnet
from model.co3net import co3net
from model.isfdnet import IdentifyAndStyleFeatureDecouplingNet
from model.loss.other_losses import *
from model.loss.triplet_loss import TripletLoss
from utils.file_util import *
from utils.grl import GRL


# 模型训练
class Train(nn.Module):
    """模型训练::
        INPUTS:
            model：网络模型
            epochs：训练迭代次数
            optimizer：优化器
            scheduler：学习率调度器
            train_data_loader：用于训练的数据加载器
            test_data_loader：用于测试的数据加载器
            model_path：保存模型的路径
            loss_se_weight：交叉熵损失函数权重
            loss_con_weight：对比损失函数权重
            loss_id_weight：身份损失权重
            loss_rec_weight：重建损失权重
            loss_cycle_weight：循环损失权重
            loss_ortho_weight：正交损失权重
            loss_id_adversarial_weight：对抗性身份损失权重
            loss_id_consistency_weight：身份表征一致性损失权重
    """
    def __init__(self, model, epochs, optimizer, scheduler, train_data_loader, test_data_loader, model_path,
                 loss_ce_weight, loss_tl_weight, loss_id_weight, loss_rec_weight, loss_cycle_weight, loss_ortho_weight,
                 loss_id_adversarial_weight, loss_id_consistency_weight, trial=None):

        super(Train, self).__init__()

        self.model = model  # 网络模型
        self.epochs = epochs  # 训练迭代次数
        self.optimizer = optimizer  # 优化器
        self.scheduler = scheduler  # 学习率调度器
        self.train_data_loader = train_data_loader  # 用于训练的数据加载器
        self.test_data_loader = test_data_loader    # 用于测试的数据加载器
        self.model_path = model_path    # 保存模型的路径

        self.loss_id_weight = loss_id_weight    # 身份损失权重
        self.loss_ce_weight = loss_ce_weight  # 交叉熵损失函数权重
        self.loss_tl_weight = loss_tl_weight  # 三元组损失函数权重
        self.loss_rec_weight = loss_rec_weight  # 重建损失权重
        self.loss_cycle_weight = loss_cycle_weight  # 循环损失权重
        self.loss_ortho_weight = loss_ortho_weight  # 正交损失权重
        self.loss_id_adversarial_weight = loss_id_adversarial_weight  # 对抗性身份损失权重
        self.loss_id_consistency_weight = loss_id_consistency_weight  # 身份表征一致性损失权重

        self.loss_ce = nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.loss_tl = TripletLoss(distance="SRT")  # 三元组损失函数
        self.loss_rec = ReconstructionLoss()  # 重建损失函数
        self.loss_cycle = CycleConsistencyLoss()   # 循环损失函数
        self.loss_id_adversarial = nn.CrossEntropyLoss()  # 对抗性身份损失函数

        # 总 step 数（用于 step 级调度）
        self.total_steps = max(1, self.epochs * len(self.train_data_loader))
        self.grl = GRL(lambd=1.0)   # 梯度反转层
        # 试验句柄
        self.trial = trial

    def match(self, match_type, data_loader, epoch):
        losses = 0
        losses_id, losses_rec, losses_cycle, losses_ortho, losses_id_adversarial, losses_id_consistency = 0, 0, 0, 0, 0, 0
        total_mask_overlap, total_mask_coverage = 0, 0
        right_num = 0

        for batch_id, (datas, target, paths) in enumerate(data_loader):

            # 锚样本
            anchor_data = datas[0].cuda()
            anchor_target = target[0].cuda()
            anchor_path = paths  # 锚样本的路径
            # 正样本
            positive_data = datas[1].cuda()
            positive_target = target[1].cuda()
            # 负样本
            negative_data = datas[2].cuda()
            negative_target = target[2].cuda()
            # 风格扰动样本
            stylized_anchor_data = datas[3].cuda()
            stylized_anchor_target = target[3].cuda()

            if match_type == 'test':
                with (torch.no_grad()):
                    (f_id_1, f_sty_1, output_1, fe1, f_mixed_1,
                     mask_id_1, mask_sty_1, mask_overlap_1, mask_coverage_1) = self.model(anchor_data, None)
                    (f_id_2, f_sty_2, output_2, fe2, f_mixed_2,
                     mask_id_2, mask_sty_2, mask_overlap_2, mask_coverage_2) = self.model(positive_data, None)
                    (f_id_3, f_sty_3, output_3, fe3, f_mixed_3,
                     mask_id_3, mask_sty_3, mask_overlap_3, mask_coverage_3) = self.model(negative_data, None)
                    (f_id_4, f_sty_4, output_4, fe4, f_mixed_4,
                     mask_id_4, mask_sty_4, mask_overlap_4, mask_coverage_4) = self.model(stylized_anchor_data, None)
            else:   # 如果是训练模型
                # 使用优化器将模型的梯度清零
                self.optimizer.zero_grad()
                # 使用模型对输入数据和目标进行前向传播，得到输出和特征编码
                (f_id_1, f_sty_1, output_1, fe1, f_mixed_1,
                 mask_id_1, mask_sty_1, mask_overlap_1, mask_coverage_1) = self.model(anchor_data, anchor_target)
                (f_id_2, f_sty_2, output_2, fe2, f_mixed_2,
                 mask_id_2, mask_sty_2, mask_overlap_2, mask_coverage_2) = self.model(positive_data, positive_target)
                (f_id_3, f_sty_3, output_3, fe3, f_mixed_3,
                 mask_id_3, mask_sty_3, mask_overlap_3, mask_coverage_3) = self.model(negative_data, negative_target)
                (f_id_4, f_sty_4, output_4, fe4, f_mixed_4,
                 mask_id_4, mask_sty_4, mask_overlap_4, mask_coverage_4) = self.model(stylized_anchor_data, stylized_anchor_target)

            loss_ce_anchor = self.loss_ce(output_1, anchor_target)
            loss_ce_stylized = self.loss_ce(output_4, stylized_anchor_target)
            loss_ce = 0.5 * (loss_ce_anchor + loss_ce_stylized)
            loss_tl, _, _, _ = self.loss_tl(fe1, fe4, fe3)
            loss_id = loss_ce * self.loss_ce_weight + loss_tl * self.loss_tl_weight

            f_recon = self.model.decode(f_id_1, f_sty_1)
            loss_rec = self.loss_rec(f_recon, f_mixed_1.detach())

            f_swap = self.model.decode(f_id_1, f_sty_4)
            f_swap_dp = F.dropout(f_swap, p=0.2, training=(match_type == 'train'))
            f_id_1_cycle, _ = self.model.cycle_encode(f_swap_dp)
            loss_cycle = self.loss_cycle(f_id_1_cycle, f_id_1.detach())

            norm_id_1 = F.normalize(f_id_1, dim=1)
            norm_sty_1 = F.normalize(f_sty_1, dim=1)
            pair_cos = torch.sum(norm_id_1 * norm_sty_1, dim=1)
            loss_ortho_pair = torch.mean(pair_cos.pow(2))
            C = torch.matmul(norm_id_1.t(), norm_sty_1) / norm_id_1.size(0)
            loss_ortho_batch = torch.sum(C.pow(2))
            loss_ortho = 0.5 * loss_ortho_pair + 0.5 * loss_ortho_batch

            sty_logits_1 = self.model.arcface_adv(self.grl(f_sty_1), anchor_target)
            sty_logits_4 = self.model.arcface_adv(self.grl(f_sty_4), stylized_anchor_target)
            loss_id_adversarial = (self.loss_id_adversarial(sty_logits_1, anchor_target)
                                   + self.loss_id_adversarial(sty_logits_4, stylized_anchor_target)) / 2.0

            norm_f_id_1 = F.normalize(f_id_1, dim=1)
            norm_f_id_4 = F.normalize(f_id_4, dim=1)
            cos_sim = torch.sum(norm_f_id_1 * norm_f_id_4, dim=1)
            cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
            angle_distance = torch.acos(cos_sim) / math.pi
            loss_id_consistency = angle_distance.mean()

            loss = (loss_id * self.loss_id_weight
                    + loss_rec * self.loss_rec_weight
                    + loss_cycle * self.loss_cycle_weight
                    + loss_ortho * self.loss_ortho_weight
                    + loss_id_adversarial * self.loss_id_adversarial_weight
                    + loss_id_consistency * self.loss_id_consistency_weight
                    )

            if match_type == 'train':
                loss.backward(retain_graph=None)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            losses += loss.data.cpu().numpy()
            losses_id += loss_id.data.cpu().numpy()
            losses_rec += loss_rec.data.cpu().numpy()
            losses_cycle += loss_cycle.data.cpu().numpy()
            losses_ortho += loss_ortho.data.cpu().numpy()
            losses_id_adversarial += loss_id_adversarial.data.cpu().numpy()
            losses_id_consistency += loss_id_consistency.data.cpu().numpy()

            total_mask_overlap += mask_overlap_1.data.cpu().numpy()
            total_mask_coverage += mask_coverage_1.data.cpu().numpy()

            prediction = output_1.data.max(dim=1, keepdim=True)[1]

            right_num += prediction.eq(anchor_target.data.view_as(prediction)).cpu().sum().numpy()

        total = len(data_loader.dataset)
        losses = losses / total
        losses_id = losses_id / total
        losses_rec = losses_rec / total
        losses_cycle = losses_cycle / total
        losses_ortho = losses_ortho / total
        losses_id_adversarial = losses_id_adversarial / total
        losses_id_consistency = losses_id_consistency / total

        num_batches = len(data_loader)
        avg_mask_overlap = total_mask_overlap / num_batches
        avg_mask_coverage = total_mask_coverage / num_batches

        accuracy = right_num / total

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
            print('epoch=', epoch + 1, '\ttype=', match_type,
                  '\tlosses_id=', losses_id,
                  '\tlosses_rec=', losses_rec,
                  '\tlosses_cycle=', losses_cycle,
                  '\tlosses_ortho=', losses_ortho,
                  '\tlosses_id_adversarial=', losses_id_adversarial,
                  '\tlosses_id_consistency=', losses_id_consistency,
                  '\tright num=', right_num, '/', total, '\taccuracy=', accuracy)

        return losses, accuracy, avg_mask_overlap, avg_mask_coverage

    # 训练
    def trainings(self, data_loader, epoch):
        self.model.train()
        average_loss, accuracy, mask_overlap, mask_coverage = self.match('train', data_loader, epoch)
        return average_loss, accuracy, mask_overlap, mask_coverage

    # 测试
    def testings(self, data_loader, epoch):
        self.model.eval()
        average_loss, accuracy, mask_overlap, mask_coverage = self.match('test', data_loader, epoch)
        return average_loss, accuracy, mask_overlap, mask_coverage

    # 迭代训练
    def epoch_train(self):
        train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []
        train_mask_overlaps, train_mask_coverages = [], []
        test_mask_overlaps, test_mask_coverages = [], []

        max_accuracy, best_train_loss, best_test_loss = 0, 0, 0

        segment_interval = 200
        segment_best_accuracy = -1.0
        segment_best_state_dict = None
        segment_start_epoch = 0

        for epoch in range(self.epochs):

            epoch_train_loss, epoch_train_accuracy, train_overlap, train_coverage = self.trainings(self.train_data_loader, epoch)
            epoch_test_loss, epoch_test_accuracy, test_overlap, test_coverage = self.testings(self.test_data_loader, epoch)

            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_accuracy)
            test_losses.append(epoch_test_loss)
            test_accuracies.append(epoch_test_accuracy)

            train_mask_overlaps.append(train_overlap)
            train_mask_coverages.append(train_coverage)
            test_mask_overlaps.append(test_overlap)
            test_mask_coverages.append(test_coverage)

            self.scheduler.step()

            if epoch_train_accuracy >= max_accuracy:
                max_accuracy = epoch_train_accuracy
                best_train_loss = epoch_train_loss
                best_test_loss = epoch_test_loss
                # 保存模型以及模型参数
                torch.save(self.model, os.path.join(self.model_path, 'best_model.pth'))
                torch.save(self.model.state_dict(), os.path.join(self.model_path, 'best_model_params.pth'))

            if epoch_train_accuracy >= segment_best_accuracy:
                segment_best_accuracy = epoch_train_accuracy
                segment_best_state_dict = copy.deepcopy(self.model.state_dict())

            end_of_segment = ((epoch + 1) % segment_interval == 0) or (epoch == self.epochs - 1)
            if end_of_segment and segment_best_state_dict is not None:
                segment_end_epoch = epoch + 1
                segment_file_prefix = f'segment_{segment_start_epoch + 1}_{segment_end_epoch}'
                state_dict_path = os.path.join(self.model_path, f'{segment_file_prefix}_best_state_dict.pth')
                torch.save(segment_best_state_dict, state_dict_path)
                segment_start_epoch = epoch + 1
                segment_best_accuracy = -1.0
                segment_best_state_dict = None

            if self.trial is not None:
                try:
                    self.trial.report(epoch_train_accuracy, step=epoch)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
                except NameError:
                    import optuna
                    self.trial.report(epoch_train_accuracy, step=epoch)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()

        print('Max accuracy is', max_accuracy)

        torch.save(self.model, os.path.join(self.model_path, 'last_model.pth'))
        torch.save(self.model.state_dict(), os.path.join(self.model_path, 'last_model_params.pth'))

        write_file_for_txt(self.model_path, 'train_losses.txt', train_losses)
        write_file_for_txt(self.model_path, 'train_accuracies.txt', train_accuracies)
        write_file_for_txt(self.model_path, 'test_losses.txt', test_losses)
        write_file_for_txt(self.model_path, 'test_accuracies.txt', test_accuracies)
        write_file_for_txt_one(self.model_path, 'max_accuracy.txt', max_accuracy)
        write_file_for_txt_one(self.model_path, 'best_train_loss.txt', best_train_loss)
        write_file_for_txt_one(self.model_path, 'best_test_loss.txt', best_test_loss)

        write_file_for_txt(self.model_path, 'train_mask_overlaps.txt', train_mask_overlaps)
        write_file_for_txt(self.model_path, 'train_mask_coverages.txt', train_mask_coverages)
        write_file_for_txt(self.model_path, 'test_mask_overlaps.txt', test_mask_overlaps)
        write_file_for_txt(self.model_path, 'test_mask_coverages.txt', test_mask_coverages)

        return max_accuracy, best_train_loss


'''  
may the force be with you.
@FileName   train
Created by 24 on 2024/1/22.
'''
