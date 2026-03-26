"""
@文件名    reproducibility
@作者      AI
@说明      统一的随机种子与可复现设置
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_cudnn: bool = True) -> None:
    """为 Python、NumPy、PyTorch 设置统一随机种子。

    参数
    ----
    seed: int
        统一使用的随机种子数值。
    deterministic_cudnn: bool
        是否启用 CuDNN 的确定性模式（更慢但结果可复现）。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN 确定性设置
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """为每个 DataLoader worker 单独设置随机种子。

    这能保证多进程数据增强与采样在不同 worker 中的可复现性。
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_dataloader_generator(seed: int) -> torch.Generator:
    """创建带固定种子的 `torch.Generator`，用于 DataLoader 的确定性打乱。"""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


