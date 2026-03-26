"""
@FileName   auto_run.py
@Author  24
@Date    2024/2/28 0031
@Version 1.0.0
freedom is the oxygen of the soul.
"""

import os
import torch

# 查看cuda是否可用
print(torch.cuda.is_available())
# 设置GPU设备
torch.cuda.empty_cache()


root_path = './results/'
use_backbone = 'ccnet'  # 可选 'ccnet' 'co3net'

save_path = root_path + use_backbone + '/'
# 数据库配置
databases = [
    {"name": "Tongji", "label_num": 600, "intra_class_num": 20},
    {"name": "PolyU", "label_num": 386, "intra_class_num": 9},
    {"name": "IITD", "label_num": 460, "intra_class_num": 5},
    {"name": "MSRed", "label_num": 500, "intra_class_num": 12},
    {"name": "MSGreen", "label_num": 500, "intra_class_num": 12},
    {"name": "MSBlue", "label_num": 500, "intra_class_num": 12},
    {"name": "MSNIR", "label_num": 500, "intra_class_num": 12},
]

# 通用参数 4 7   5 8
common_params = ("--loss_ce 0.8 --loss_tl 0.2 --vit_floor_num 10  --run_type train --backbone_name " + use_backbone + " "
                 "--weight_for_backbone 0.7 --weight_for_id_and_sty 1.0 --loss_id_weight 1.0 "
                 "--loss_rec_weight 0.05 --loss_cycle_weight 0.03 --loss_ortho_weight 0.12 "
                 "--loss_id_adversarial_weight 0.4 --loss_id_consistency_weight 0.02")

# 遍历数据库配置并运行命令
def run():
    for db in databases:
        command = (
            "python main.py --database_name {name} --label_num {label_num} "
            "--one_label_intra_class_num {intra_class_num} {common_params} "
            "--train_file ./data/{name}/train_{name}.txt "
            "--test_file ./data/{name}/test_{name}.txt "
            "--verify_file ./data/{name}/verify_{name}.txt --save_path {save_path}"
        ).format(
            name=db['name'],
            label_num=db['label_num'],
            intra_class_num=db['intra_class_num'],
            common_params=common_params,
            save_path=save_path
        )
        print(command)
        os.system(command)


# 正常运行
run()

# 结果保存路径
save_path_for_cross = root_path + use_backbone + "/cross_experiment/"
# 数据库配置
databases_for_cross = [
    {"name": "PolyU", "label_num": 386, "intra_class_num": 9},
    {"name": "Tongji", "label_num": 600, "intra_class_num": 20},
    {"name": "IITD", "label_num": 460, "intra_class_num": 5},
    {"name": "MSRed", "label_num": 500, "intra_class_num": 12},
    {"name": "MSGreen", "label_num": 500, "intra_class_num": 12},
    {"name": "MSBlue", "label_num": 500, "intra_class_num": 12},
    {"name": "MSNIR", "label_num": 500, "intra_class_num": 12},
]


# 遍历所有数据集组合方式并运行命令
def cross_experiment():
    for source in databases_for_cross:
        for target in databases_for_cross:
            if source["name"] == "Tongji" and source["name"] != target["name"]:  # 排除源和目标相同的情况
                command = (
                    "python cross_experiment.py --source_id {source_name} --target_id {target_name} "
                    "--label_num {source_label_num} --save_path {save_path_for_cross} "
                    "--source_model {source_model}"
                ).format(
                    source_name=source["name"],
                    target_name=target["name"],
                    source_label_num=source["label_num"],
                    save_path_for_cross=save_path_for_cross,
                    source_model= root_path + use_backbone + "/" + source["name"] + "/model/best_model.pth"
                )
                print(f"正在运行命令: {command}")
                os.system(command)


# 交叉实验
cross_experiment()

'''  
may the force be with you.
@FileName   auto_run.py
Created by 24 on 2024/2/28.
'''
