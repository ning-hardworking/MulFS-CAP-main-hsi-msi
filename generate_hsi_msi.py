# -*- coding: utf-8 -*-
"""
完整的 HSI/MSI 退化网络训练和数据生成脚本
适配 MulFS-CAP 从 IR-VIS 到 HSI-MSI 的迁移

数据流程：
1. 训练退化网络：X/ + Z/ + Y/ → 学习 hsi_degen 和 msi_degen
2. 生成Pair1数据：X/ → Z_reconst/ + Y_reconst/ (原始配准对)
3. 生成Pair2数据：X_deformed/ → Z_deformed/ + Y_deformed/ (形变配准对)
"""

import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import scipy.io as sio
from pathlib import Path

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ====================== 1. 配置参数 ======================
ROOT_PATH = r"D:\datas\CAVEdata"

# 原始数据（数据集提供）
GT_RAW_DIR = os.path.join(ROOT_PATH, "X")  # 原始GT (512×512×31)
HSI_RAW_DIR = os.path.join(ROOT_PATH, "Z")  # 原始HSI (16×16×31)
MSI_RAW_DIR = os.path.join(ROOT_PATH, "Y")  # 原始MSI (512×512×3)

# 形变GT（generate_deformed_gt.py生成）
GT_DEFORMED_DIR = os.path.join(ROOT_PATH, "X_deformed")  # 形变GT (512×512×31)

# 输出目录（本脚本生成）
Z_RECONST_SAVE = os.path.join(ROOT_PATH, "Z_reconst")  # 重建HSI (Pair1)
Y_RECONST_SAVE = os.path.join(ROOT_PATH, "Y_reconst")  # 重建MSI (Pair1)
HSI_DEFORMED_SAVE = os.path.join(ROOT_PATH, "Z_deformed")  # 形变HSI (Pair2)
MSI_DEFORMED_SAVE = os.path.join(ROOT_PATH, "Y_deformed")  # 形变MSI (Pair2)

# 权重保存路径
WEIGHT_SAVE_PATH = ROOT_PATH

# 数据集参数（CAVE数据集固定参数，不要修改）
GT_SIZE = 512  # GT图像尺寸
HSI_SIZE = 16  # HSI图像尺寸
MSI_BANDS = 3  # MSI通道数
GT_BANDS = 31  # GT/HSI通道数
DOWNSAMPLE_SCALE = GT_SIZE // HSI_SIZE  # 下采样倍率：32

# 训练参数
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-5

# 设备配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("🚀 MulFS-CAP HSI-MSI 退化网络训练和数据生成")
print("=" * 70)
print(f"✅ 训练设备: {DEVICE}")
print(f"✅ 下采样倍率: {DOWNSAMPLE_SCALE}倍 (512 → 16)")
print(f"✅ GT: {GT_SIZE}×{GT_SIZE}×{GT_BANDS}")
print(f"✅ HSI: {HSI_SIZE}×{HSI_SIZE}×{GT_BANDS}")
print(f"✅ MSI: {GT_SIZE}×{GT_SIZE}×{MSI_BANDS}")
print("=" * 70 + "\n")


# ====================== 2. 噪声模块 ======================
class GaussianNoise(nn.Module):
    """高斯噪声"""

    def __init__(self, sigma=0.001):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.sigma
        return x


class PoissonNoise(nn.Module):
    """泊松噪声"""

    def forward(self, x):
        if self.training:
            return torch.poisson(x.clamp(min=1e-8)) / x.clamp(min=1e-8) * x
        return x


# ====================== 3. 光谱注意力模块 ======================
class SpectralAttention(nn.Module):
    """光谱通道注意力"""

    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ====================== 4. 残差块 ======================
class ResidualBlock(nn.Module):
    """带GroupNorm的残差块"""

    def __init__(self, in_channels, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=groups, bias=False)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=groups, bias=False)
        self.norm2 = nn.GroupNorm(groups, in_channels)

        # 初始化
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + residual)


# ====================== 5. HSI 退化网络 ======================
class DeepHSIDegenerator(nn.Module):
    """
    深度HSI退化网络
    GT(31, 512, 512) → HSI(31, 16, 16)

    退化过程：
    1. 特征提取（保持光谱信息）
    2. 空间模糊（模拟光学系统的点扩散函数）
    3. 下采样（32倍，512 → 16）
    4. 噪声注入（泊松噪声 + 高斯噪声）
    """

    def __init__(self, in_bands=31, out_bands=31, scale=32):
        super().__init__()
        self.groups = 1  # 使用全卷积（不分组）以保持光谱相关性

        # 初始卷积
        self.init_conv = nn.Conv2d(in_bands, in_bands, 3, 1, 1, groups=self.groups, bias=False)

        # 残差块（保持光谱特征）
        self.res1 = ResidualBlock(in_bands, self.groups)
        self.res2 = ResidualBlock(in_bands, self.groups)

        # 空间模糊（模拟光学系统的点扩散函数）
        self.blur = nn.Sequential(
            nn.Conv2d(in_bands, in_bands, 5, 1, 2, groups=self.groups, bias=False),
            nn.Conv2d(in_bands, in_bands, 7, 1, 3, groups=self.groups, bias=False)
        )

        # 下采样（32倍：2^5 = 32）
        self.down_sample = nn.Sequential(
            nn.AvgPool2d(2, 2),  # 512 → 256
            nn.AvgPool2d(2, 2),  # 256 → 128
            nn.AvgPool2d(2, 2),  # 128 → 64
            nn.AvgPool2d(2, 2),  # 64 → 32
            nn.AvgPool2d(2, 2)  # 32 → 16
        )

        # 噪声
        self.noise = nn.Sequential(
            PoissonNoise(),
            GaussianNoise(0.001)
        )

        # 初始化
        nn.init.kaiming_normal_(self.init_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.blur[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.blur[1].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        输入: (B, 31, 512, 512)
        输出: (B, 31, 16, 16)
        """
        x = self.init_conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.blur(x)
        x = self.down_sample(x)
        x = self.noise(x)
        return x.clamp(0, 1)


# ====================== 6. MSI 退化网络 ======================
class DeepMSIDegenerator(nn.Module):
    """
    深度MSI退化网络
    GT(31, 512, 512) → MSI(3, 512, 512)

    退化过程：
    1. 特征提取（保持空间分辨率）
    2. 光谱注意力（选择重要的光谱信息）
    3. 光谱降维（31通道 → 3通道，模拟RGB传感器）
    4. 空间平滑（轻微模糊）
    5. 噪声注入（高斯噪声）
    """

    def __init__(self, in_bands=31, out_bands=3):
        super().__init__()

        # 初始卷积（深度可分离）
        self.init_conv = nn.Conv2d(in_bands, in_bands, 3, 1, 1, groups=in_bands, bias=False)

        # 残差块（提取特征）
        self.res1 = ResidualBlock(in_bands, groups=1)

        # 光谱注意力（选择重要的光谱信息）
        self.attention = SpectralAttention(in_bands)

        # 光谱降维（31 → 3，模拟RGB传感器的光谱响应函数）
        self.spectral_conv = nn.Conv2d(in_bands, out_bands, 1, 1, 0, bias=False)

        # 空间平滑（轻微模糊，保持空间分辨率）
        self.spatial_smooth = nn.Conv2d(
            out_bands, out_bands, 3, 1, 2,
            dilation=2, groups=out_bands, bias=False
        )

        # 噪声
        self.noise = GaussianNoise(0.0005)

        # 初始化
        nn.init.kaiming_normal_(self.init_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.spectral_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.spatial_smooth.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        输入: (B, 31, 512, 512)
        输出: (B, 3, 512, 512)
        """
        x = self.init_conv(x)
        x = self.res1(x)
        x = self.attention(x)
        x = self.spectral_conv(x)
        x = self.spatial_smooth(x)
        x = self.noise(x)
        return x.clamp(0, 1)


# ====================== 7. 损失函数 ======================
def total_loss(pred_hsi, real_hsi, pred_msi, real_msi):
    """
    组合损失：L1 + MSE

    Args:
        pred_hsi: 预测的HSI (B, 31, 16, 16)
        real_hsi: 真实的HSI (B, 31, 16, 16)
        pred_msi: 预测的MSI (B, 3, 512, 512)
        real_msi: 真实的MSI (B, 3, 512, 512)

    Returns:
        loss: 标量
    """
    # HSI损失
    hsi_l1 = nn.L1Loss()(pred_hsi, real_hsi)
    hsi_mse = nn.MSELoss()(pred_hsi, real_hsi)

    # MSI损失
    msi_l1 = nn.L1Loss()(pred_msi, real_msi)
    msi_mse = nn.MSELoss()(pred_msi, real_msi)

    # 组合（L1权重更高，更关注细节）
    loss = 0.7 * (hsi_l1 + msi_l1) + 0.3 * (hsi_mse + msi_mse)

    return loss


# ====================== 8. 数据加载函数 ======================
def load_mat_data(file_path):
    """
    加载.mat文件并标准化

    Args:
        file_path: .mat文件路径

    Returns:
        img_np: numpy数组 (C, H, W)，已归一化到[0,1]
    """
    mat_data = sio.loadmat(str(file_path))
    mat_values = [v for k, v in mat_data.items() if not k.startswith('__')]
    img_np = mat_values[0].astype(np.float32)

    # 🔥 自动适配维度：找到通道数=31的维度，移到第0位
    if img_np.ndim == 3:
        # 找到等于31的维度（通道维度）
        channel_axis = None
        for axis in range(3):
            if img_np.shape[axis] == GT_BANDS:
                channel_axis = axis
                break

        # 如果找到了，移到第0位
        if channel_axis is not None:
            img_np = np.moveaxis(img_np, source=channel_axis, destination=0)
        # 否则，假设最后一维是通道
        elif img_np.shape[-1] < img_np.shape[0] and img_np.shape[-1] < img_np.shape[1]:
            img_np = np.transpose(img_np, (2, 0, 1))

    # 标准化到[0,1]
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    return img_np


# ====================== 9. 训练退化网络 ======================
def train_deep_degenerators():
    """
    训练深度退化网络

    Returns:
        hsi_degen: 训练好的HSI退化网络
        msi_degen: 训练好的MSI退化网络
    """
    print("\n" + "=" * 70)
    print("🔥 开始训练深度退化网络")
    print("=" * 70)

    # 创建保存目录
    os.makedirs(WEIGHT_SAVE_PATH, exist_ok=True)

    # 初始化网络
    hsi_degen = DeepHSIDegenerator(GT_BANDS, GT_BANDS, DOWNSAMPLE_SCALE).to(DEVICE)
    msi_degen = DeepMSIDegenerator(GT_BANDS, MSI_BANDS).to(DEVICE)

    # 优化器
    optimizer = optim.Adam(
        list(hsi_degen.parameters()) + list(msi_degen.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 加载数据
    valid_suffix = ['.mat', '.MAT']
    gt_file_paths = sorted([p for p in Path(GT_RAW_DIR).glob("*.*") if p.suffix in valid_suffix])

    print(f"✅ 找到 {len(gt_file_paths)} 组训练数据")
    print(f"✅ 训练参数: Epochs={EPOCHS}, LR={LR}, Batch=1")
    print(f"✅ 优化器: Adam, 学习率调度: CosineAnnealing")
    print("-" * 70)

    # 训练循环
    hsi_degen.train()
    msi_degen.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for idx, gt_path in enumerate(gt_file_paths):
            fname = gt_path.name
            hsi_path = os.path.join(HSI_RAW_DIR, fname)
            msi_path = os.path.join(MSI_RAW_DIR, fname)

            # 检查配对文件
            if not os.path.exists(hsi_path) or not os.path.exists(msi_path):
                print(f"⚠️ 跳过 {fname}：缺少配对文件")
                continue

            try:
                # 加载数据
                gt_np = load_mat_data(gt_path)  # (31, 512, 512)
                hsi_np = load_mat_data(hsi_path)  # (31, 16, 16)
                msi_np = load_mat_data(msi_path)  # (3, 512, 512)

                # 转换为张量
                gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).to(DEVICE)
                hsi_tensor = torch.from_numpy(hsi_np).unsqueeze(0).to(DEVICE)
                msi_tensor = torch.from_numpy(msi_np).unsqueeze(0).to(DEVICE)

                # 第一个样本时打印维度验证
                if idx == 0 and epoch == 0:
                    print(f"✅ 数据维度验证:")
                    print(f"   GT:  {gt_tensor.shape}")
                    print(f"   HSI: {hsi_tensor.shape}")
                    print(f"   MSI: {msi_tensor.shape}")

                    # 前向传播验证
                    with torch.no_grad():
                        pred_hsi_test = hsi_degen(gt_tensor)
                        pred_msi_test = msi_degen(gt_tensor)
                    print(f"   预测HSI: {pred_hsi_test.shape}")
                    print(f"   预测MSI: {pred_msi_test.shape}")
                    print(f"✅ 所有维度匹配！\n")

                # 前向传播
                pred_hsi = hsi_degen(gt_tensor)
                pred_msi = msi_degen(gt_tensor)

                # 计算损失
                loss = total_loss(pred_hsi, hsi_tensor, pred_msi, msi_tensor)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # 清理内存
                del gt_tensor, hsi_tensor, msi_tensor, pred_hsi, pred_msi
                gc.collect()

            except Exception as e:
                print(f"❌ 处理 {fname} 时出错: {str(e)}")
                continue

        # 学习率调度
        scheduler.step()

        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(gt_file_paths)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch + 1:3d}/{EPOCHS}] | Loss: {avg_loss:.8f} | LR: {current_lr:.6f}")

    print("-" * 70)
    print("✅ 训练完成！")

    # 设置为评估模式并冻结参数
    hsi_degen.eval()
    msi_degen.eval()
    for param in hsi_degen.parameters():
        param.requires_grad = False
    for param in msi_degen.parameters():
        param.requires_grad = False

    # 保存权重
    hsi_weight_path = os.path.join(WEIGHT_SAVE_PATH, "deep_hsi_degen_32x.pth")
    msi_weight_path = os.path.join(WEIGHT_SAVE_PATH, "deep_msi_degen_3band.pth")

    torch.save(hsi_degen.state_dict(), hsi_weight_path)
    torch.save(msi_degen.state_dict(), msi_weight_path)

    print(f"💾 HSI退化网络权重已保存: {hsi_weight_path}")
    print(f"💾 MSI退化网络权重已保存: {msi_weight_path}")
    print("=" * 70 + "\n")

    return hsi_degen, msi_degen


# ====================== 10. 批量生成配对数据 ======================
def generate_deformed_pair_data(gt_input_dir, hsi_save_dir, msi_save_dir, desc, hsi_degen, msi_degen):
    """
    使用训练好的退化网络批量生成HSI/MSI配对数据

    Args:
        gt_input_dir: GT图像输入目录
        hsi_save_dir: HSI输出目录
        msi_save_dir: MSI输出目录
        desc: 描述信息
        hsi_degen: HSI退化网络
        msi_degen: MSI退化网络
    """
    print("\n" + "=" * 70)
    print(f"🔥 生成 {desc}")
    print("=" * 70)
    print(f"📂 输入目录: {gt_input_dir}")
    print(f"📂 HSI输出:  {hsi_save_dir}")
    print(f"📂 MSI输出:  {msi_save_dir}")
    print("-" * 70)

    # 创建输出目录
    os.makedirs(hsi_save_dir, exist_ok=True)
    os.makedirs(msi_save_dir, exist_ok=True)

    # 查找GT文件
    valid_suffix = ['.mat', '.MAT']
    gt_file_paths = sorted([p for p in Path(gt_input_dir).glob("*.*") if p.suffix in valid_suffix])

    if len(gt_file_paths) == 0:
        print(f"❌ 错误：未找到任何.mat文件！")
        return

    print(f"✅ 找到 {len(gt_file_paths)} 张GT图像")

    success_count = 0

    with torch.no_grad():
        for idx, gt_path in enumerate(gt_file_paths):
            fname = gt_path.name

            try:
                # 加载GT
                gt_np = load_mat_data(gt_path)  # (31, 512, 512)
                gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).to(DEVICE)

                # 第一个样本时打印维度
                if idx == 0:
                    print(f"✅ GT维度: {gt_tensor.shape} → 标准格式 [1, 31, 512, 512] ✔️")

                # 生成HSI和MSI
                hsi_tensor = hsi_degen(gt_tensor)  # (1, 31, 16, 16)
                msi_tensor = msi_degen(gt_tensor)  # (1, 3, 512, 512)

                # 转换为numpy并保存
                hsi_np = hsi_tensor.squeeze(0).cpu().numpy()
                msi_np = msi_tensor.squeeze(0).cpu().numpy()

                sio.savemat(os.path.join(hsi_save_dir, fname), {'data': hsi_np})
                sio.savemat(os.path.join(msi_save_dir, fname), {'data': msi_np})

                success_count += 1

                # 清理内存
                del gt_tensor, hsi_tensor, msi_tensor, gt_np, hsi_np, msi_np
                gc.collect()

                # 打印进度
                if (idx + 1) % 5 == 0 or (idx + 1) == len(gt_file_paths):
                    print(f"进度: {idx + 1}/{len(gt_file_paths)} 张，成功 {success_count} 张")

            except Exception as e:
                print(f"⚠️ 跳过文件 {fname}: {str(e)}")
                continue

    print("-" * 70)
    print(f"✅ {desc} 生成完成！")
    print(f"✅ 成功生成 {success_count}/{len(gt_file_paths)} 组配对数据")
    print("=" * 70 + "\n")


# ====================== 11. 主函数 ======================
if __name__ == "__main__":
    print("\n" + "🎯 " * 35)
    print("开始执行 MulFS-CAP HSI-MSI 数据生成流程")
    print("🎯 " * 35 + "\n")

    # ========== 方式1: 训练退化网络（第一次运行时使用）==========
    # 如果你还没有训练过退化网络，取消下面这行的注释
    hsi_degen, msi_degen = train_deep_degenerators()

    # ========== 方式2: 加载已训练的权重（推荐）==========
    # 如果你已经训练过，直接加载权重
    print("=" * 70)
    print("📦 加载预训练的退化网络权重")
    print("=" * 70)

    hsi_weight_path = os.path.join(WEIGHT_SAVE_PATH, "deep_hsi_degen_32x.pth")
    msi_weight_path = os.path.join(WEIGHT_SAVE_PATH, "deep_msi_degen_3band.pth")

    # 检查权重文件是否存在
    if not os.path.exists(hsi_weight_path) or not os.path.exists(msi_weight_path):
        print("❌ 错误：未找到预训练权重！")
        print("请先运行训练流程（取消主函数中的训练代码注释）")
        print(f"期望路径: {hsi_weight_path}")
        print(f"期望路径: {msi_weight_path}")
        exit(1)

    # 初始化网络
    hsi_degen = DeepHSIDegenerator(GT_BANDS, GT_BANDS, DOWNSAMPLE_SCALE).to(DEVICE)
    msi_degen = DeepMSIDegenerator(GT_BANDS, MSI_BANDS).to(DEVICE)

    # 加载权重
    hsi_degen.load_state_dict(torch.load(hsi_weight_path))
    msi_degen.load_state_dict(torch.load(msi_weight_path))

    # 设置为评估模式
    hsi_degen.eval()
    msi_degen.eval()
    for param in hsi_degen.parameters():
        param.requires_grad = False
    for param in msi_degen.parameters():
        param.requires_grad = False

    print(f"✅ HSI退化网络权重已加载: {hsi_weight_path}")
    print(f"✅ MSI退化网络权重已加载: {msi_weight_path}")
    print("=" * 70 + "\n")

    # ========== 生成配对数据 ==========
    # 1️⃣ 从原始GT生成重建的HSI和MSI（Pair 1: 原始配准对）
    generate_deformed_pair_data(
        GT_RAW_DIR,  # 输入: X/ (原始GT)
        Z_RECONST_SAVE,  # 输出: Z_reconst/ (重建HSI)
        Y_RECONST_SAVE,  # 输出: Y_reconst/ (重建MSI)
        "Pair 1 配准数据 (Z_reconst + Y_reconst)",
        hsi_degen,
        msi_degen
    )

    # 2️⃣ 从形变GT生成形变的HSI和MSI（Pair 2: 形变配准对）
    generate_deformed_pair_data(
        GT_DEFORMED_DIR,  # 输入: X_deformed/ (形变GT)
        HSI_DEFORMED_SAVE,  # 输出: Z_deformed/ (形变HSI)
        MSI_DEFORMED_SAVE,  # 输出: Y_deformed/ (形变MSI)
        "Pair 2 配准数据 (Z_deformed + Y_deformed)",
        hsi_degen,
        msi_degen
    )

    # ========== 最终总结 ==========
    print("\n" + "🎉 " * 35)
    print("所有数据生成完成！")
    print("🎉 " * 35 + "\n")

    print("=" * 70)
    print("📊 最终数据结构:")
    print("=" * 70)
    print(f"✅ Pair 1 (原始配准对):")
    print(f"   - HSI: {Z_RECONST_SAVE}")
    print(f"   - MSI: {Y_RECONST_SAVE}")
    print(f"   - GT:  {GT_RAW_DIR}")
    print()
    print(f"✅ Pair 2 (形变配准对):")
    print(f"   - HSI: {HSI_DEFORMED_SAVE}")
    print(f"   - MSI: {MSI_DEFORMED_SAVE}")
    print(f"   - GT:  {GT_DEFORMED_DIR}")
    print("=" * 70)
    print()
    print("🚀 下一步：运行 train.py 开始训练 MulFS-CAP！")
    print("=" * 70 + "\n")