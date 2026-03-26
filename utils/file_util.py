"""
@ClassName   util
@Author  24
@Date    2024/1/25 03:41
@Version 1.0.0
freedom is the oxygen of the soul.
"""
import os
import sys

import scipy.io as sio

import numpy as np


# 存入txt文件
def write_file_for_txt(path, file_name, texts):
    os.makedirs(path, exist_ok=True)  # 检查并创建文件夹

    # with open(os.path.join(path, file_name), 'w') as file:
    #     for text in texts:
    #         file.write(str(text) + '\n')

    np.savetxt(str(os.path.join(path, file_name)), texts, fmt='%s', delimiter='\n', newline='\n')


# 存入txt文件（非数组）
def write_file_for_txt_one(path, file_name, texts):
    os.makedirs(path, exist_ok=True)  # 检查并创建文件夹
    with open(os.path.join(path, file_name), 'w') as file:
        file.write(str(texts))


# 存入npy文件
def write_file_for_npy(path, file_name, texts):
    os.makedirs(path, exist_ok=True)  # 检查并创建文件夹

    np.save(path + file_name, texts)


# 读取txt文件
def read_file_for_txt(path, file_name):
    return np.loadtxt(str(os.path.join(path, file_name)))


# 读取npy文件
def read_file_for_npy(path, file_name):
    try:
        return np.load(str(os.path.join(path, file_name)))
    except FileNotFoundError:
        print("Not found path or file:", path + file_name)
        return None


# 检查文件是否存在
def check_file_exist(path):
    if not os.path.exists(path):
        print(f"文件 '{path}' 不存在。")
        sys.exit()  # 终止程序运行


# 将文件由npy转为mat
def npy_to_mat(path, file_name, mat_file_name=None):
    if mat_file_name is None:
        mat_file_name = file_name.replace('.npy', '.mat')  # 修改文件扩展名
    sio.savemat(os.path.join(path, mat_file_name), {'data': read_file_for_npy(path, file_name)})    # 保存为mat文件


# 新的 helper: 将npy转换为mat并指定mat里的变量名
def npy_to_mat_with_key(path, file_name, key_name, mat_file_name=None):
    """Load path/file_name (.npy) and save as .mat with variable name key_name.
    Returns True if saved, False if source not found.
    """
    if mat_file_name is None:
        mat_file_name = file_name.replace('.npy', '.mat')
    data = read_file_for_npy(path, file_name)
    if data is None:
        return False
    sio.savemat(os.path.join(path, mat_file_name), {key_name: data})
    return True


# 将原来 main 的逻辑封装为一个函数，按目录遍历包含 scores_EER_test.txt 文件的子文件夹并保存 DisInter/DisIntra.mat
def convert_scores_txt_to_mat(directory):
    """Traverse immediate subfolders of `directory`, read scores_EER_test.txt in each folder and save DisInter.mat and DisIntra.mat.

    Behavior preserved from the original script. This function raises FileNotFoundError if a folder lacks the expected file.
    """
    # 列出所有文件夹
    folders = [f.path for f in os.scandir(directory) if f.is_dir()]

    converted = []

    # 遍历每个文件夹
    for folder in folders:
        # 初始化两个数组来存储数据
        inter = []
        intra = []

        # 读取文件
        # 定义文件路径
        # veri_eer_path = os.path.join(folder, 'scores_VeriEER.txt')
        eer_test_path = os.path.join(folder, 'scores_EER_test.txt')

        # 检查哪个文件存在并读取
        if os.path.exists(eer_test_path):
            file_path = eer_test_path
        # if os.path.exists(veri_eer_path):
        #     file_path = veri_eer_path
        else:
            raise FileNotFoundError("Neither scores_VeriEER.txt nor scores_EER_test.txt exists.")
        with open(file_path, 'r') as file:
            for line in file:
                # 分割每行并转换为浮点数
                score, label = map(float, line.split())
                # 根据标签将数据添加到相应的数组
                if label == -1:
                    inter.append(score)
                elif label == 1:
                    intra.append(score)

        # 将列表转换为NumPy数组
        inter = np.array(inter)
        intra = np.array(intra)

        # 保存数组到MAT文件
        sio.savemat(os.path.join(folder, 'DisInter.mat'), {'DisInter': inter})
        sio.savemat(os.path.join(folder, 'DisIntra.mat'), {'DisIntra': intra})

        converted.append(folder)

    return converted


# 新函数：递归查找 results 目录下的 genuine_matching_score.npy 和 imposter_matching_score.npy 并转换为 DisIntra.mat / DisInter.mat
def convert_results_npy_to_mat(root_path):
    """Recursively scan `root_path` for `genuine_matching_score.npy` and `imposter_matching_score.npy`.

    For each folder containing these files, create:
      - DisIntra.mat  (from genuine_matching_score.npy, variable name 'DisIntra')
      - DisInter.mat  (from imposter_matching_score.npy, variable name 'DisInter')

    Returns a dict with lists of created files: {'DisIntra': [...], 'DisInter': [...]}.
    """
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Root path does not exist: {root_path}")

    created = {'DisIntra': [], 'DisInter': []}

    for dirpath, dirnames, filenames in os.walk(root_path):
        # genuine -> DisIntra
        if 'genuine_matching_score.npy' in filenames:
            ok = npy_to_mat_with_key(dirpath, 'genuine_matching_score.npy', 'DisIntra', 'DisIntra.mat')
            if ok:
                created['DisIntra'].append(os.path.join(dirpath, 'DisIntra.mat'))
                print(f"Saved DisIntra.mat in {dirpath}")
        # imposter -> DisInter
        if 'imposter_matching_score.npy' in filenames:
            ok = npy_to_mat_with_key(dirpath, 'imposter_matching_score.npy', 'DisInter', 'DisInter.mat')
            if ok:
                created['DisInter'].append(os.path.join(dirpath, 'DisInter.mat'))
                print(f"Saved DisInter.mat in {dirpath}")

    return created


if __name__ == "__main__":
    # 示例用法
    root_dir = r"D:\Education\master\项目&工作\研究工作\ISFDNet(身份与风格特征解耦网络)\results"  # 替换为你的根目录路径
    converted_files = convert_results_npy_to_mat(root_dir)
    print("转换完成，生成的文件列表：", converted_files)


'''
may the force be with you.
@ClassName   util
Created by 24 on 2024/1/25.
'''
