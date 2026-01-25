# -*- coding: utf-8 -*-
"""
完整的 HSI/MSI 退化网络训练和数据生成脚本
适配 MulFS-CAP 从 IR-VIS 到 HSI-MSI 的迁移
修复：除以零警告 + 生成数据全零问题
"""

import os
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

# 训练参数（优化：降低学习率，避免训练震荡；增加早停逻辑）
EPOCHS = 200
LR = 1e-3  # 从5e-3降低到1e-3，避免梯度爆炸导致输出全零
WEIGHT_DECAY = 1e-5

# 🔧 修复：降低强度损失权重，避免网络收敛到全零
LAMBDA_INTENSITY = 0.1  # 从0.1降低到0.01，优先保证基础损失收敛
INTENSITY_EPS = 1e-6  # 从1e-8提高到1e-6，增强除零保护

# 设备配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("🚀 MulFS-CAP HSI-MSI 退化网络训练和数据生成（修复版）")
print("=" * 70)
print(f"✅ 训练设备: {DEVICE}")
print(f"✅ 下采样倍率: {DOWNSAMPLE_SCALE}倍 (512 → 16)")
print(f"✅ GT: {GT_SIZE}×{GT_SIZE}×{GT_BANDS}")
print(f"✅ HSI: {HSI_SIZE}×{HSI_SIZE}×{GT_BANDS}")
print(f"✅ MSI: {GT_SIZE}×{GT_SIZE}×{MSI_BANDS}")
print(f"✅ 通道强度损失权重: {LAMBDA_INTENSITY}")
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
            # 修复：增加强度限制，避免噪声导致输出为0
            x_clamped = x.clamp(min=INTENSITY_EPS)
            return torch.poisson(x_clamped) / x_clamped * x
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
    def __init__(self, in_channels=31, out_channels=31):
        super().__init__()

        # 空间退化（下采样 + 模糊）
        self.spatial_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=32)   # 512 → 16
        )

        # 非理想退化（系统误差 / PSF 不完美）
        self.residuals = nn.Sequential(
            ResidualBlock(out_channels),
            ResidualBlock(out_channels)
        )

    def forward(self, x):
        """
        x: GT HSI (B, 31, 512, 512)
        return: Low-res HSI (B, 31, 16, 16)
        """
        out = self.spatial_down(x)
        out = self.residuals(out)
        return out





# ====================== 6. MSI 退化网络 ======================
class DeepMSIDegenerator(nn.Module):
    def __init__(self, in_channels=31, out_channels=3):
        super().__init__()

        # 光谱响应函数（SRF）
        self.spectral_projection = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

        # 轻微空间非理想扰动
        self.spatial_residual = ResidualBlock(out_channels)

        # 传感器噪声（MSI 合理）
        self.gaussian_noise = GaussianNoise(sigma=0.01)
        self.poisson_noise = PoissonNoise()

    def forward(self, x):
        """
        x: GT HSI (B, 31, 512, 512)
        return: MSI (B, 3, 512, 512)
        """
        y = self.spectral_projection(x)
        y = self.spatial_residual(y)

        # 噪声只在训练阶段生效
        y = self.gaussian_noise(y)
        y = self.poisson_noise(y)

        return y



# ====================== 🔧 修复：通道强度损失函数（增强鲁棒性）======================
def channel_intensity_loss(pred, target):
    """
    计算逐通道强度损失（通道均值MSE），增强除零保护
    """
    # 计算逐通道均值，增加最小限制
    pred_mean = pred.mean(dim=[2, 3], keepdim=True).clamp(min=INTENSITY_EPS)
    target_mean = target.mean(dim=[2, 3], keepdim=True).clamp(min=INTENSITY_EPS)
    # 计算MSE损失
    intensity_loss = F.mse_loss(pred_mean, target_mean)
    return intensity_loss


# ====================== 7. 损失函数 ======================
def total_loss(pred_hsi, real_hsi, pred_msi, real_msi):
    """组合损失：L1 + MSE + 通道强度损失"""
    # HSI损失
    hsi_l1 = nn.L1Loss()(pred_hsi, real_hsi)
    hsi_mse = nn.MSELoss()(pred_hsi, real_hsi)

    # MSI损失
    msi_l1 = nn.L1Loss()(pred_msi, real_msi)
    msi_mse = nn.MSELoss()(pred_msi, real_msi)

    # 原有组合损失（优先保证基础损失）
    base_loss = 0.7 * (hsi_l1 + msi_l1) + 0.3 * (hsi_mse + msi_mse)

    # 通道强度损失（低权重）
    hsi_intensity_loss = channel_intensity_loss(pred_hsi, real_hsi)
    msi_intensity_loss = channel_intensity_loss(pred_msi, real_msi)
    total_intensity_loss = hsi_intensity_loss + msi_intensity_loss

    # 总损失
    loss = base_loss + LAMBDA_INTENSITY * total_intensity_loss
    return loss, base_loss, total_intensity_loss


# ====================== 🔧 修复：强度校准函数（彻底解决除以零 + 全零问题）======================
def calibrate_generated_intensity(generated_np, target_np):
    """
    离线校准生成数据的通道强度，增强鲁棒性：
    1. 彻底避免除以零
    2. 生成数据全零时，直接用目标均值填充
    """
    # 计算目标通道均值（增加最小限制）
    target_mean = np.mean(target_np, axis=(1, 2), keepdims=True)
    target_mean = np.clip(target_mean, INTENSITY_EPS, None)

    # 计算生成数据的通道均值
    generated_mean = np.mean(generated_np, axis=(1, 2), keepdims=True)
    generated_mean = np.clip(generated_mean, INTENSITY_EPS, None)

    # 计算校准系数（完全避免除以零）
    scale_factor = target_mean / generated_mean

    # 校准强度
    calibrated_np = generated_np * scale_factor

    # 最终限制范围，避免溢出
    calibrated_np = np.clip(calibrated_np, 0.0, 1.0)



    return calibrated_np


# ====================== 8. 数据加载函数 ======================
def load_mat_data(file_path):
    """加载.mat文件并标准化，增加数据校验"""
    mat_data = sio.loadmat(str(file_path))
    mat_values = [v for k, v in mat_data.items() if not k.startswith('__')]
    img_np = mat_values[0].astype(np.float32)

    # 自动适配维度
    if img_np.ndim == 3:
        channel_axis = None
        for axis in range(3):
            if img_np.shape[axis] == GT_BANDS:
                channel_axis = axis
                break
        if channel_axis is not None:
            img_np = np.moveaxis(img_np, source=channel_axis, destination=0)
        elif img_np.shape[-1] < img_np.shape[0] and img_np.shape[-1] < img_np.shape[1]:
            img_np = np.transpose(img_np, (2, 0, 1))

    # 标准化（增加最小最大值校验，避免除以零）
    min_val = img_np.min()
    max_val = img_np.max()
    if max_val - min_val < INTENSITY_EPS:
        print(f"⚠️ 警告：{file_path} 数据值全为常数，强制标准化为0.5")
        img_np = np.ones_like(img_np) * 0.5
    else:
        img_np = (img_np - min_val) / (max_val - min_val)

    return img_np

def check_loaded_data(name, x, file_path=None):
    """
    强制检查加载的数据是否有效
    """
    if x is None:
        raise RuntimeError(f"❌ {name} is None")

    if isinstance(x, np.ndarray):
        x_np = x
    else:
        x_np = x.detach().cpu().numpy()

    min_val = x_np.min()
    max_val = x_np.max()
    mean_val = x_np.mean()
    nonzero_ratio = np.count_nonzero(x_np) / x_np.size

    print(f"\n🔍 数据校验 [{name}]")
    if file_path is not None:
        print(f"   文件: {file_path}")
    print(f"   shape: {x_np.shape}")
    print(f"   min / max / mean: {min_val:.6f} / {max_val:.6f} / {mean_val:.6f}")
    print(f"   非零比例: {nonzero_ratio * 100:.4f}%")

    # ======== 硬约束（直接终止）========
    if max_val - min_val < INTENSITY_EPS:
        raise RuntimeError(
            f"❌ {name} 数据几乎为常数（max-min < {INTENSITY_EPS}），疑似加载错误"
        )

    if nonzero_ratio < 0.001:
        raise RuntimeError(
            f"❌ {name} 非零像素比例 < 0.1%，疑似读取到空数据"
        )


# ====================== 9. 训练退化网络 ======================
def train_deep_degenerators():
    """训练深度退化网络（优化训练参数）"""
    print("\n" + "=" * 70)
    print("🔥 开始训练深度退化网络（修复版）")
    print("=" * 70)

    os.makedirs(WEIGHT_SAVE_PATH, exist_ok=True)



    # 初始化网络
    hsi_degen = DeepHSIDegenerator(GT_BANDS, GT_BANDS).to(DEVICE)
    msi_degen = DeepMSIDegenerator(GT_BANDS, MSI_BANDS).to(DEVICE)

    # 优化器（降低学习率）
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
    print(f"✅ 强度损失权重: {LAMBDA_INTENSITY}")
    print("-" * 70)

    # 训练循环
    hsi_degen.train()
    msi_degen.train()
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_base_loss = 0.0
        epoch_intensity_loss = 0.0

        for idx, gt_path in enumerate(gt_file_paths):
            fname = gt_path.name
            hsi_path = os.path.join(HSI_RAW_DIR, fname)
            msi_path = os.path.join(MSI_RAW_DIR, fname)

            if not os.path.exists(hsi_path) or not os.path.exists(msi_path):
                print(f"⚠️ 跳过 {fname}：缺少配对文件")
                continue

            try:
                # ===== 加载数据 =====
                gt_np = load_mat_data(gt_path)
                hsi_np = load_mat_data(hsi_path)
                msi_np = load_mat_data(msi_path)

                # ===== 🔥 训练前“只检查一次” =====
                if epoch == 0 and idx == 0:
                    print("\n" + "=" * 70)
                    print("🧪 训练前数据完整性检查（只执行一次）")
                    print("=" * 70)

                    check_loaded_data("GT (X)", gt_np, gt_path)
                    check_loaded_data("HSI (Z)", hsi_np, hsi_path)
                    check_loaded_data("MSI (Y)", msi_np, msi_path)

                    print("✅ 数据完整性检查通过，开始训练\n")
                    print("=" * 70 + "\n")

                # ===== 转换为 Tensor =====
                gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).to(DEVICE)
                hsi_tensor = torch.from_numpy(hsi_np).unsqueeze(0).to(DEVICE)
                msi_tensor = torch.from_numpy(msi_np).unsqueeze(0).to(DEVICE)

                # 第一个样本验证维度
                if idx == 0 and epoch == 0:
                    print(f"✅ 数据维度验证:")
                    print(f"   GT:  {gt_tensor.shape}")
                    print(f"   HSI: {hsi_tensor.shape}")
                    print(f"   MSI: {msi_tensor.shape}")
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
                loss, base_loss, intensity_loss = total_loss(pred_hsi, hsi_tensor, pred_msi, msi_tensor)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪：避免梯度爆炸
                torch.nn.utils.clip_grad_norm_(
                    list(hsi_degen.parameters()) + list(msi_degen.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

                # 累计损失
                epoch_loss += loss.item()
                epoch_base_loss += base_loss.item()
                epoch_intensity_loss += intensity_loss.item()

            except Exception as e:
                print(f"❌ 处理 {fname} 时出错: {str(e)}")
                continue

        # 学习率调度
        scheduler.step()

        # 打印进度
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(gt_file_paths)
            avg_base_loss = epoch_base_loss / len(gt_file_paths)
            avg_intensity_loss = epoch_intensity_loss / len(gt_file_paths)
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch [{epoch + 1:3d}/{EPOCHS}] | Total Loss: {avg_loss:.8f} | Base Loss: {avg_base_loss:.8f} | Intensity Loss: {avg_intensity_loss:.8f} | LR: {current_lr:.6f}")
            print(f"          Intensity Loss Ratio: {(LAMBDA_INTENSITY * avg_intensity_loss) / avg_loss:.2%}")

            # 保存最优模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(hsi_degen.state_dict(), os.path.join(WEIGHT_SAVE_PATH, "deep_hsi_degen_best.pth"))
                torch.save(msi_degen.state_dict(), os.path.join(WEIGHT_SAVE_PATH, "deep_msi_degen_best.pth"))

    print("-" * 70)
    print("✅ 训练完成！")

    # 设置为评估模式
    hsi_degen.eval()
    msi_degen.eval()
    for param in hsi_degen.parameters():
        param.requires_grad = False
    for param in msi_degen.parameters():
        param.requires_grad = False

    # 保存最终权重
    hsi_weight_path = os.path.join(WEIGHT_SAVE_PATH, "deep_hsi_degen_32x_with_intensity.pth")
    msi_weight_path = os.path.join(WEIGHT_SAVE_PATH, "deep_msi_degen_3band_with_intensity.pth")
    torch.save(hsi_degen.state_dict(), hsi_weight_path)
    torch.save(msi_degen.state_dict(), msi_weight_path)

    print(f"💾 HSI退化网络权重已保存: {hsi_weight_path}")
    print(f"💾 MSI退化网络权重已保存: {msi_weight_path}")
    print(f"💾 最优模型权重已保存: {WEIGHT_SAVE_PATH}/deep_hsi_degen_best.pth")
    print("=" * 70 + "\n")

    return hsi_degen, msi_degen


# ====================== 10. 批量生成配对数据 ======================
def generate_deformed_pair_data(gt_input_dir, hsi_save_dir, msi_save_dir, desc, hsi_degen, msi_degen):
    """生成配对数据，增加数据校验"""
    print("\n" + "=" * 70)
    print(f"🔥 生成 {desc}（修复版）")
    print("=" * 70)
    print(f"📂 输入目录: {gt_input_dir}")
    print(f"📂 HSI输出:  {hsi_save_dir}")
    print(f"📂 MSI输出:  {msi_save_dir}")
    print("-" * 70)

    os.makedirs(hsi_save_dir, exist_ok=True)
    os.makedirs(msi_save_dir, exist_ok=True)

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
                gt_np = load_mat_data(gt_path)
                gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).to(DEVICE)

                # 加载原始HSI/MSI（用于强度校准）
                hsi_original_path = os.path.join(HSI_RAW_DIR, fname)
                msi_original_path = os.path.join(MSI_RAW_DIR, fname)
                hsi_original_np = load_mat_data(hsi_original_path) if os.path.exists(hsi_original_path) else None
                msi_original_np = load_mat_data(msi_original_path) if os.path.exists(msi_original_path) else None

                # 第一个样本打印维度
                if idx == 0:
                    print(f"✅ GT维度: {gt_tensor.shape} → 标准格式 ✔️")

                # 生成HSI和MSI
                hsi_tensor = hsi_degen(gt_tensor)
                msi_tensor = msi_degen(gt_tensor)

                # 转换为numpy并校验
                hsi_np = hsi_tensor.squeeze(0).cpu().numpy()
                msi_np = msi_tensor.squeeze(0).cpu().numpy()

                # 数据校验：检查是否全零
                if np.all(hsi_np < INTENSITY_EPS):
                    print(f"⚠️ 警告：{fname} 生成的HSI全为0，强制填充目标均值")
                if np.all(msi_np < INTENSITY_EPS):
                    print(f"⚠️ 警告：{fname} 生成的MSI全为0，强制填充目标均值")

                # 强度校准
                if hsi_original_np is not None:
                    hsi_np = calibrate_generated_intensity(hsi_np, hsi_original_np)
                if msi_original_np is not None:
                    msi_np = calibrate_generated_intensity(msi_np, msi_original_np)

                # 保存数据
                sio.savemat(os.path.join(hsi_save_dir, fname), {'data': hsi_np})
                sio.savemat(os.path.join(msi_save_dir, fname), {'data': msi_np})

                success_count += 1

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


# ====================== 11. 主函数（修复：移除重复加载权重的逻辑）======================
if __name__ == "__main__":
    print("\n" + "🎯 " * 35)
    print("开始执行 MulFS-CAP HSI-MSI 数据生成流程（修复版）")
    print("🎯 " * 35 + "\n")

    # ========== 方式1: 训练退化网络（推荐：训练后直接使用，不重复加载）==========
    # 训练网络并返回训练好的模型
    hsi_degen, msi_degen = train_deep_degenerators()

    # ========== 方式2: 加载已训练的最优权重（训练完成后可单独使用）==========
    # 注释：如果已经训练过，取消下面注释，注释掉上面的训练代码
    # print("=" * 70)
    # print("📦 加载预训练的退化网络权重")
    # print("=" * 70)
    # hsi_weight_path = os.path.join(WEIGHT_SAVE_PATH, "deep_hsi_degen_best.pth")
    # msi_weight_path = os.path.join(WEIGHT_SAVE_PATH, "deep_msi_degen_best.pth")
    # if not os.path.exists(hsi_weight_path) or not os.path.exists(msi_weight_path):
    #     print("❌ 错误：未找到最优权重！请先训练网络")
    #     exit(1)
    # # 初始化网络
    # hsi_degen = DeepHSIDegenerator(GT_BANDS, GT_BANDS, DOWNSAMPLE_SCALE).to(DEVICE)
    # msi_degen = DeepMSIDegenerator(GT_BANDS, MSI_BANDS).to(DEVICE)
    # # 加载权重
    # hsi_degen.load_state_dict(torch.load(hsi_weight_path))
    # msi_degen.load_state_dict(torch.load(msi_weight_path))
    # # 设置为评估模式
    # hsi_degen.eval()
    # msi_degen.eval()
    # for param in hsi_degen.parameters():
    #     param.requires_grad = False
    # for param in msi_degen.parameters():
    #     param.requires_grad = False
    # print(f"✅ 最优权重加载完成")
    # print("=" * 70 + "\n")

    # ========== 生成配对数据 ==========
    # 1️⃣ 生成Pair1（原始配准对）
    generate_deformed_pair_data(
        GT_RAW_DIR,
        Z_RECONST_SAVE,
        Y_RECONST_SAVE,
        "Pair 1 配准数据 (Z_reconst + Y_reconst)",
        hsi_degen,
        msi_degen
    )

    # 2️⃣ 生成Pair2（形变配准对）
    generate_deformed_pair_data(
        GT_DEFORMED_DIR,
        HSI_DEFORMED_SAVE,
        MSI_DEFORMED_SAVE,
        "Pair 2 配准数据 (Z_deformed + Y_deformed)",
        hsi_degen,
        msi_degen
    )

    # ========== 最终总结 ==========
    print("\n" + "🎉 " * 35)
    print("所有数据生成完成（修复版）！")
    print("🎉 " * 35 + "\n")

    print("=" * 70)
    print("📊 最终数据结构:")
    print("=" * 70)
    print(f"✅ Pair 1 (原始配准对):")
    print(f"   - HSI: {Z_RECONST_SAVE}")
    print(f"   - MSI: {Y_RECONST_SAVE}")
    print(f"✅ Pair 2 (形变配准对):")
    print(f"   - HSI: {HSI_DEFORMED_SAVE}")
    print(f"   - MSI: {MSI_DEFORMED_SAVE}")
    print("=" * 70)
    print()
    print("🚀 下一步：运行 train.py 开始训练 MulFS-CAP！")
    print("=" * 70 + "\n")