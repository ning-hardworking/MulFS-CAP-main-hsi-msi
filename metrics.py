# metrics.py - 完整的HSI-MSI评估指标模块

import torch
import torch.nn.functional as F
import numpy as np


def calculate_psnr(pred, target, data_range=1.0):
    """
    计算峰值信噪比 (PSNR)

    Args:
        pred: (B, C, H, W) - 预测图像
        target: (B, C, H, W) - 真实图像
        data_range: 数据范围（默认1.0表示归一化到[0,1]）

    Returns:
        psnr: 标量，单位dB
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(data_range ** 2 / mse)
    return psnr.item()


def calculate_sam(pred, target):
    """
    计算光谱角距离 (Spectral Angle Mapper, SAM)
    衡量光谱相似度，值越小越好

    Args:
        pred: (B, C, H, W) - 预测的高光谱图像
        target: (B, C, H, W) - 真实的高光谱图像

    Returns:
        sam: 标量，单位度（°）
    """
    # 将空间维度展平: (B, C, H, W) -> (B, C, H*W)
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)

    # 计算内积
    dot_product = torch.sum(pred_flat * target_flat, dim=1)  # (B, H*W)

    # 计算模长
    pred_norm = torch.norm(pred_flat, dim=1) + 1e-8  # 避免除零
    target_norm = torch.norm(target_flat, dim=1) + 1e-8

    # 计算cos值
    cos_theta = dot_product / (pred_norm * target_norm)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 防止数值误差

    # 计算角度（弧度 -> 度数）
    sam = torch.acos(cos_theta).mean() * 180 / np.pi

    return sam.item()


def calculate_ssim(pred, target, data_range=1.0):
    """
    计算结构相似度 (SSIM)

    Args:
        pred: (B, C, H, W)
        target: (B, C, H, W)
        data_range: 数据范围

    Returns:
        ssim: 标量，范围[0,1]，越大越好
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_pred = F.avg_pool2d(pred, 3, 1, 1)
    mu_target = F.avg_pool2d(target, 3, 1, 1)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred * target, 3, 1, 1) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return ssim_map.mean().item()


def calculate_rmse(pred, target):
    """
    计算均方根误差 (RMSE)

    Args:
        pred: (B, C, H, W)
        target: (B, C, H, W)

    Returns:
        rmse: 标量，越小越好
    """
    mse = F.mse_loss(pred, target)
    rmse = torch.sqrt(mse)
    return rmse.item()


def calculate_ergas(pred, target, scale_factor=32):
    """
    计算ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse)
    用于评估高光谱超分辨率质量

    Args:
        pred: (B, C, H, W)
        target: (B, C, H, W)
        scale_factor: 下采样倍率（默认32，即512/16=32）

    Returns:
        ergas: 标量，越小越好
    """
    pred = pred.view(pred.size(0), pred.size(1), -1)  # (B, C, H*W)
    target = target.view(target.size(0), target.size(1), -1)

    mean_target = target.mean(dim=2, keepdim=True)  # (B, C, 1)
    mse_per_band = ((pred - target) ** 2).mean(dim=2)  # (B, C)

    ergas = 100 * scale_factor * torch.sqrt(
        (mse_per_band / (mean_target.squeeze(2) ** 2 + 1e-8)).mean()
    )

    return ergas.item()


def calculate_cc(pred, target):
    """
    计算相关系数 (Cross Correlation, CC)

    Args:
        pred: (B, C, H, W)
        target: (B, C, H, W)

    Returns:
        cc: 标量，范围[-1,1]，越接近1越好
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()

    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean

    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())

    cc = numerator / (denominator + 1e-8)

    return cc.item()


def evaluate_all_metrics(pred, target, scale_factor=32, data_range=1.0):
    """
    一次性计算所有指标

    Args:
        pred: (B, C, H, W) - 预测的高光谱图像
        target: (B, C, H, W) - 真实的高光谱图像
        scale_factor: HSI下采样倍率（默认32）
        data_range: 数据范围（默认1.0）

    Returns:
        metrics: 字典，包含所有指标
    """
    with torch.no_grad():
        metrics = {
            'PSNR': calculate_psnr(pred, target, data_range),
            'SAM': calculate_sam(pred, target),
            'SSIM': calculate_ssim(pred, target, data_range),
            'RMSE': calculate_rmse(pred, target),
            'ERGAS': calculate_ergas(pred, target, scale_factor),
            'CC': calculate_cc(pred, target)
        }

    return metrics


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 测试
    pred = torch.rand(1, 31, 128, 128)  # 预测的高光谱图像
    target = torch.rand(1, 31, 128, 128)  # 真实的GT

    # 方法1：单独计算
    psnr = calculate_psnr(pred, target)
    sam = calculate_sam(pred, target)
    print(f"PSNR: {psnr:.4f} dB")
    print(f"SAM:  {sam:.4f}°")

    # 方法2：一次性计算所有指标
    metrics = evaluate_all_metrics(pred, target, scale_factor=32)
    print("\n所有指标:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")