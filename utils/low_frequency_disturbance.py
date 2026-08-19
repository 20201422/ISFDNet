"""
@file_name  low_frequency_disturbance
@author     24
@date       2025/7/6 16:31
@version    1.0.0
freedom is the oxygen of the soul.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter


def fft_decompose(img):
    """
    对图像执行傅里叶变换并分离低频和高频成分。
    """
    img_np = np.array(img)
    H, W = img_np.shape

    f_transform = np.fft.fft2(img_np)
    f_transform_shifted = np.fft.fftshift(f_transform)

    center_h, center_w = H // 2, W // 2
    radius = int(0.1 * min(H, W))
    y, x = np.ogrid[:H, :W]
    mask = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2) <= radius

    low_freq_shifted = f_transform_shifted * mask
    high_freq_shifted = f_transform_shifted * (1 - mask)

    return low_freq_shifted, high_freq_shifted

def fft_reconstruct(low_freq_shifted, high_freq_shifted):
    """
    使用低频和高频复数频谱重建图像。
    """
    reconstructed_f_transform_shifted = low_freq_shifted + high_freq_shifted

    reconstructed_img_np = np.real(np.fft.ifft2(np.fft.ifftshift(reconstructed_f_transform_shifted)))

    return reconstructed_img_np


def perturb_low_frequency(low_freq_shifted, mode='scale_noise_illumination', random_state=None,
                          scale_range=(0.9, 1.1), noise_ratio=(0.01, 0.03),
                          illumination_strength_range=(0.01, 0.03),
                          clip_dc_ratio=(0.9, 1.1),
                          enforce_symmetry=True):
    """对低频复数频谱(fftshift 后)进行可控幅度扰动, 模拟光照/风格变化 (保持相位以保结构)。

    参数:
        low_freq_shifted: ndarray(complex), 与原图同尺寸, 已 fftshift 的低频复谱(其余频率通常为 0)。
        mode: 由 'scale','noise','illumination' 以 '_' 组合的字符串, 如 'scale_noise', 'noise', 'scale_noise_illumination'。
        random_state: 随机种子(可选, 使用独立 RNG 不污染全局)。
        scale_range: (min,max) 全局幅度乘性缩放区间。
        noise_ratio: (min,max) 加性噪声强度, 按平均幅度比例定义。
        illumination_strength_range: (min,max) 乘性平滑光照纹理强度区间。
        clip_dc_ratio: 限制 DC(中心) 幅度相对原来的缩放比例范围, 防止全局亮度漂移过大。
        enforce_symmetry: 是否在扰动后强制共轭对称以保证 ifft 结果实值。

    返回:
        perturbed_low_freq_shifted: 扰动后的低频复谱(fftshift 后)。
    """

    rng = np.random.default_rng(random_state)

    amp = np.abs(low_freq_shifted)
    phase = np.angle(low_freq_shifted)

    if amp.size == 0 or np.all(amp == 0):
        return low_freq_shifted

    amp_mean = amp[amp > 0].mean() if np.any(amp > 0) else 0.0

    perturbed_amp = amp.copy()

    if 'scale' in mode:
        s = rng.uniform(*scale_range)
        perturbed_amp = perturbed_amp * s

    if 'noise' in mode and amp_mean > 0:
        noise_ratio_val = rng.uniform(*noise_ratio)
        noise_std = noise_ratio_val * amp_mean
        noise = rng.normal(0.0, noise_std, size=amp.shape)
        perturbed_amp = np.clip(perturbed_amp + noise, 0, None)

    if 'illumination' in mode:
        random_field = rng.normal(0, 1, amp.shape)
        smooth_field = gaussian_filter(random_field, sigma=2.0)
        sf_std = smooth_field.std() + 1e-8
        smooth_field = smooth_field / sf_std

        strength = rng.uniform(*illumination_strength_range)
        illum_map = strength * smooth_field

        perturbed_amp = perturbed_amp * (1.0 + illum_map)
        perturbed_amp = np.clip(perturbed_amp, 0, None)

    perturbed = perturbed_amp * np.exp(1j * phase)

    mask = (np.abs(low_freq_shifted) > 0).astype(float)
    perturbed = perturbed * mask

    return perturbed


"""
@file_name  low_frequency_disturbance
Created by 24 on 2025/7/6
"""
