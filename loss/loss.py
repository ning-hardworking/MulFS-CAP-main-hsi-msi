import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from model import model

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


# ========== 原有损失函数（已适配多通道） ==========

def Loss_intensity(hsi, msi, image_fused):
    """
    ✅ 完全适配31通道的强度损失
    Args:
        hsi: (B, 31, 4, 4) - 低分辨率HSI
        msi: (B, 3, 128, 128) - 高分辨率MSI
        image_fused: (B, 31, 128, 128) - 融合结果（31通道高光谱）

    核心思想：
    1. HSI损失：监督所有31个通道
    2. MSI损失：将31通道的融合结果转换为3通道后与MSI比较
    """
    # ========== 1. HSI损失（31通道完整监督）==========
    # 将融合结果下采样到HSI分辨率
    fused_downsampled = F.interpolate(
        image_fused,
        size=(hsi.size(2), hsi.size(3)),  # 下采样到 4×4
        mode='bilinear',
        align_corners=False
    )
    # ✅ 对所有31个通道进行L1损失计算
    hsi_li = F.l1_loss(fused_downsampled, hsi)

    # ========== 2. MSI损失（31通道 → 3通道转换）==========
    # 方案A：使用学习到的线性组合（推荐，更灵活）
    # 这里简化为对所有31个通道求平均后重复3次
    # 更好的方式是用一个1x1卷积层学习最佳的通道组合

    # ✅ 简化方案：对31个通道按RGB波段分组求平均
    # 假设波段0-10对应蓝色，11-20对应绿色，21-30对应红色
    B, C, H, W = image_fused.shape

    # 方法1：均匀分组（简单但有效）
    band_per_channel = C // 3  # 31 // 3 = 10
    r_band = image_fused[:, :band_per_channel, :, :].mean(dim=1, keepdim=True)  # 前10个波段
    g_band = image_fused[:, band_per_channel:2 * band_per_channel, :, :].mean(dim=1, keepdim=True)  # 中10个
    b_band = image_fused[:, 2 * band_per_channel:, :, :].mean(dim=1, keepdim=True)  # 后11个

    fused_rgb = torch.cat([r_band, g_band, b_band], dim=1)  # (B, 3, H, W)

    # 方法2：使用特定波段（如果知道光谱响应曲线）
    # fused_rgb = image_fused[:, [9, 19, 29], :, :]  # 选择特定波段

    # ✅ 与MSI进行L1损失计算
    msi_li = F.l1_loss(fused_rgb, msi)

    # ========== 3. 总损失 ==========
    return hsi_li + msi_li


class L_Grad(nn.Module):
    """
    ✅ 完全适配31通道的梯度损失
    对所有31个通道分别计算梯度，然后聚合
    """

    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, img1, img2, image_fused=None):
        """
        Args:
            img1: HSI (B, 31, 4, 4) 或 (B, 31, 16, 16)
            img2: MSI (B, 3, 128, 128)
            image_fused: 融合结果 (B, 31, 128, 128) 或 None

        核心思想：
        1. 对HSI的31个通道分别计算梯度
        2. 对MSI的3个通道分别计算梯度
        3. 对融合结果的31个通道分别计算梯度
        4. 通过通道聚合进行比较
        """
        if image_fused is None:
            # ========== 场景1：只比较两张图像（用于配准对）==========
            # 将HSI上采样到MSI的分辨率
            if img1.size(2) != img2.size(2) or img1.size(3) != img2.size(3):
                img1_up = F.interpolate(
                    img1,
                    size=(img2.size(2), img2.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                img1_up = img1

            # ✅ 对所有31个通道计算梯度
            gradient_1_all = self.sobelconv(img1_up)  # (B, 31, H, W)

            # 将31通道梯度转换为3通道（与MSI对应）
            C = gradient_1_all.size(1)
            band_per_channel = C // 3
            gradient_1_r = gradient_1_all[:, :band_per_channel, :, :].mean(dim=1, keepdim=True)
            gradient_1_g = gradient_1_all[:, band_per_channel:2 * band_per_channel, :, :].mean(dim=1, keepdim=True)
            gradient_1_b = gradient_1_all[:, 2 * band_per_channel:, :, :].mean(dim=1, keepdim=True)
            gradient_1 = torch.cat([gradient_1_r, gradient_1_g, gradient_1_b], dim=1)

            gradient_2 = self.sobelconv(img2)  # (B, 3, H, W)
            Loss_gradient = F.l1_loss(gradient_1, gradient_2)
            return Loss_gradient
        else:
            # ========== 场景2：监督融合质量（完整31通道）==========
            # 将HSI上采样到融合图像的分辨率
            if img1.size(2) != image_fused.size(2) or img1.size(3) != image_fused.size(3):
                img1_up = F.interpolate(
                    img1,
                    size=(image_fused.size(2), image_fused.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                img1_up = img1

            # ✅ 对所有31个通道计算梯度
            gradient_hsi_all = self.sobelconv(img1_up)  # (B, 31, H, W)
            gradient_fused_all = self.sobelconv(image_fused)  # (B, 31, H, W)

            # 将31通道梯度转换为3通道（用于与MSI比较）
            C = gradient_hsi_all.size(1)
            band_per_channel = C // 3

            # HSI的31通道梯度 -> 3通道
            gradient_hsi_r = gradient_hsi_all[:, :band_per_channel, :, :].mean(dim=1, keepdim=True)
            gradient_hsi_g = gradient_hsi_all[:, band_per_channel:2 * band_per_channel, :, :].mean(dim=1, keepdim=True)
            gradient_hsi_b = gradient_hsi_all[:, 2 * band_per_channel:, :, :].mean(dim=1, keepdim=True)
            gradient_hsi_3ch = torch.cat([gradient_hsi_r, gradient_hsi_g, gradient_hsi_b], dim=1)

            # 融合结果的31通道梯度 -> 3通道
            gradient_fused_r = gradient_fused_all[:, :band_per_channel, :, :].mean(dim=1, keepdim=True)
            gradient_fused_g = gradient_fused_all[:, band_per_channel:2 * band_per_channel, :, :].mean(dim=1,
                                                                                                       keepdim=True)
            gradient_fused_b = gradient_fused_all[:, 2 * band_per_channel:, :, :].mean(dim=1, keepdim=True)
            gradient_fused_3ch = torch.cat([gradient_fused_r, gradient_fused_g, gradient_fused_b], dim=1)

            # MSI的梯度（3通道）
            gradient_msi = self.sobelconv(img2)  # (B, 3, H, W)

            # ✅ 计算两部分损失
            # 损失1：融合结果的3通道梯度应该保留HSI和MSI的最强梯度
            gradient_joint_3ch = torch.max(gradient_hsi_3ch, gradient_msi)
            loss_3ch = F.l1_loss(gradient_fused_3ch, gradient_joint_3ch)

            # 损失2：融合结果的31通道梯度应该与HSI的31通道梯度一致（光谱保真度）
            loss_31ch = F.l1_loss(gradient_fused_all, gradient_hsi_all)

            # ✅ 综合损失（平衡空间细节和光谱保真度）
            Loss_gradient = loss_3ch + 0.5 * loss_31ch

            return Loss_gradient


class Sobelxy(nn.Module):
    """
    ✅ Sobel算子 - 已支持任意通道数（包括31通道）
    这个类不需要修改
    """
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        """
        ✅ 支持任意通道数
        输入: (B, C, H, W) - C可以是3或31
        输出: (B, C, H, W) - 每个通道的梯度
        """
        B, C, H, W = x.shape

        # ✅ 对每个通道分别计算梯度（支持31通道）
        sobelx_list = []
        sobely_list = []
        for i in range(C):
            channel = x[:, i:i+1, :, :]
            sobelx = F.conv2d(channel, self.weightx, padding=1)
            sobely = F.conv2d(channel, self.weighty, padding=1)
            sobelx_list.append(sobelx)
            sobely_list.append(sobely)

        sobelx = torch.cat(sobelx_list, dim=1)
        sobely = torch.cat(sobely_list, dim=1)

        return torch.abs(sobelx) + torch.abs(sobely)


class SpectralConsistencyLoss(nn.Module):
    """
    ✅ 光谱一致性损失 - 确保31通道的光谱曲线保真度
    """

    def __init__(self):
        super(SpectralConsistencyLoss, self).__init__()

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 31, H, W) - 预测的高光谱图像
            target: (B, 31, H, W) - 真实的高光谱图像
        Returns:
            loss: 光谱一致性损失
        """
        # ✅ 方法1：直接计算所有31个通道的L1损失
        spectral_loss = F.l1_loss(pred, target)

        # ✅ 方法2：计算光谱向量的余弦相似度（更关注光谱形状）
        # 将空间维度展平
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)  # (B, 31, H*W)
        target_flat = target.view(target.size(0), target.size(1), -1)  # (B, 31, H*W)

        # 计算余弦相似度
        pred_norm = F.normalize(pred_flat, p=2, dim=1)  # L2归一化
        target_norm = F.normalize(target_flat, p=2, dim=1)

        # 余弦相似度损失
        cos_sim = (pred_norm * target_norm).sum(dim=1).mean()  # 越大越好
        cos_loss = 1 - cos_sim  # 转换为损失（越小越好）

        # 综合损失
        return spectral_loss + 0.1 * cos_loss


class SpectralAngleLoss(nn.Module):
    """
    ✅ 光谱角距离 (SAM) - 衡量光谱向量之间的角度
    """

    def __init__(self):
        super(SpectralAngleLoss, self).__init__()

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 31, H, W) - 预测的高光谱图像
            target: (B, 31, H, W) - 真实的高光谱图像
        Returns:
            loss: 平均SAM损失（弧度）
        """
        # 将空间维度展平
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)  # (B, 31, H*W)
        target_flat = target.view(target.size(0), target.size(1), -1)  # (B, 31, H*W)

        # 计算内积
        dot_product = torch.sum(pred_flat * target_flat, dim=1)  # (B, H*W)

        # 计算模长
        pred_norm = torch.norm(pred_flat, dim=1) + 1e-8  # (B, H*W)
        target_norm = torch.norm(target_flat, dim=1) + 1e-8  # (B, H*W)

        # 计算cos值
        cos_theta = dot_product / (pred_norm * target_norm)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        # 计算角度（弧度）
        sam = torch.acos(cos_theta)  # (B, H*W)

        return torch.mean(sam)


class CorrelationCoefficient(nn.Module):
    def __init__(self):
        super(CorrelationCoefficient, self).__init__()

    def c_CC(self, A, B):
        A_mean = torch.mean(A, dim=[2, 3], keepdim=True)
        B_mean = torch.mean(B, dim=[2, 3], keepdim=True)
        A_sub_mean = A - A_mean
        B_sub_mean = B - B_mean
        sim = torch.sum(torch.mul(A_sub_mean, B_sub_mean))
        A_sdev = torch.sqrt(torch.sum(torch.pow(A_sub_mean, 2)))
        B_sdev = torch.sqrt(torch.sum(torch.pow(B_sub_mean, 2)))
        out = sim / (A_sdev * B_sdev + 1e-8)  # 避免除零
        return out

    def forward(self, A, B, Fusion=None):
        if Fusion is None:
            CC = self.c_CC(A, B)
        else:
            r_1 = self.c_CC(A, Fusion)
            r_2 = self.c_CC(B, Fusion)
            CC = (r_1 + r_2) / 2
        return CC


class L_correspondence(nn.Module):
    def __init__(self, height=256, weight=256):
        super(L_correspondence, self).__init__()
        self.height = height
        self.weight = weight

    def forward(self, correspondence_matrixs, index_r):
        size = correspondence_matrixs.size()
        device = correspondence_matrixs.device
        small_window_size = int(math.sqrt(size[2]))
        large_window_size = int(math.sqrt(size[3]))
        batch_size = size[1]
        win_num = size[0]
        index = index_r

        base_index = torch.arange(0, self.height * self.weight, device=device).reshape(self.height, self.weight)
        unfold_win = nn.Unfold(kernel_size=(small_window_size, small_window_size), stride=small_window_size)
        base_index = base_index.repeat(1, 1, 1, 1).to(dtype=torch.float32)
        sw_absolute_base_index = unfold_win(base_index)
        lw_absolute_base_index = model.df_window_partition(base_index, large_window_size, small_window_size,
                                                           is_bewindow=False)
        sw_win_ralative_base_index = torch.arange(0, small_window_size * small_window_size, device=device)

        loss_correspondence_matrix = torch.zeros(win_num, batch_size, device=device)
        loss_correspondence_matrix_1 = torch.zeros(win_num, batch_size, device=device)

        for i in range(batch_size):
            for j in range(win_num):
                lw_win_absolute_base_index = lw_absolute_base_index[0, :, j]
                sw_win_absolute_base_index = sw_absolute_base_index[0, :, j]
                indices = (lw_win_absolute_base_index.unsqueeze(dim=1) == index[i, 1, :]).nonzero(as_tuple=True)
                corresponding_lw_absolute_indices = lw_win_absolute_base_index[indices[0]]
                corresponding_allimgae_absolute_indices = index[i, 0, :][indices[1]]
                corresponding_lw_relative_indices = indices[0]
                insw_indices = (sw_win_absolute_base_index.unsqueeze(
                    dim=1) == corresponding_allimgae_absolute_indices).nonzero(as_tuple=True)
                insw_corresponding_sw_absolute_index = sw_win_absolute_base_index[insw_indices[0]]
                insw_corresponding_sw_relative_index = sw_win_ralative_base_index[insw_indices[0]]
                insw_corresponding_lw_absolute_index = corresponding_lw_absolute_indices[insw_indices[1]]
                insw_corresponding_lw_relative_index = corresponding_lw_relative_indices[insw_indices[1]]

                zero_mask = torch.logical_or(insw_corresponding_sw_absolute_index != 0,
                                             insw_corresponding_lw_absolute_index != 0).nonzero()
                nozeropair_insw_corresponding_sw_relative_index = insw_corresponding_sw_relative_index[zero_mask]
                nozeropair_insw_corresponding_lw_relative_index = insw_corresponding_lw_relative_index[zero_mask]

                corresponding_win_index = torch.cat([nozeropair_insw_corresponding_sw_relative_index.permute(1, 0),
                                                     nozeropair_insw_corresponding_lw_relative_index.permute(1, 0)],
                                                    dim=0)
                corresponding_win_matrix = torch.sparse_coo_tensor(corresponding_win_index,
                                                                   torch.ones(corresponding_win_index.size()[1],
                                                                              device=device),
                                                                   (small_window_size * small_window_size,
                                                                    large_window_size * large_window_size))
                assert (torch.sum(torch.abs(corresponding_win_matrix.to_dense())) != 0)

                predict_correspondence_matrix = correspondence_matrixs[j, i, :, :]
                c_num = nozeropair_insw_corresponding_sw_relative_index.size()[0]
                predict_correspondence_matrix_1 = torch.clamp(predict_correspondence_matrix, 1e-6, 1 - 1e-6)
                l_cm = (-1 / c_num) * torch.sum(
                    torch.mul(torch.log(predict_correspondence_matrix_1), corresponding_win_matrix.to_dense()))

                loss_correspondence_matrix[j, i] = l_cm

                l_c = F.l1_loss(predict_correspondence_matrix, corresponding_win_matrix.to_dense())
                loss_correspondence_matrix_1[j, i] = l_c

        loss_correspondence_matrix = torch.mean(loss_correspondence_matrix)
        loss_correspondence_matrix_1 = torch.mean(loss_correspondence_matrix_1)

        return loss_correspondence_matrix, loss_correspondence_matrix_1


# ========== 新增：HSI-MSI专用损失函数 ==========

class SpectralAngleLoss(nn.Module):
    """
    光谱角距离 (Spectral Angle Mapper, SAM)
    衡量两个光谱向量之间的角度，值越小越好
    范围: [0, π/2]
    """

    def __init__(self):
        super(SpectralAngleLoss, self).__init__()

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - 预测的高光谱图像
            target: (B, C, H, W) - 真实的高光谱图像
        Returns:
            loss: 标量，平均SAM损失
        """
        # 将空间维度展平: (B, C, H, W) -> (B, C, H*W)
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)

        # 计算内积
        dot_product = torch.sum(pred_flat * target_flat, dim=1)  # (B, H*W)

        # 计算模长
        pred_norm = torch.norm(pred_flat, dim=1) + 1e-8  # (B, H*W)
        target_norm = torch.norm(target_flat, dim=1) + 1e-8  # (B, H*W)

        # 计算cos值
        cos_theta = dot_product / (pred_norm * target_norm)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 防止数值误差

        # 计算角度
        sam = torch.acos(cos_theta)  # (B, H*W)

        # 返回平均SAM
        return torch.mean(sam)


class SpectralConsistencyLoss(nn.Module):
    """
    光谱一致性损失
    确保融合结果的光谱曲线与GT保持一致
    """

    def __init__(self):
        super(SpectralConsistencyLoss, self).__init__()

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - 预测图像
            target: (B, C, H, W) - 目标图像
        Returns:
            loss: 光谱一致性损失
        """
        # 计算每个像素的光谱曲线差异
        # 使用L1损失
        spectral_diff = torch.abs(pred - target)

        # 对空间维度求平均，保留光谱维度
        spectral_loss = torch.mean(spectral_diff, dim=[2, 3])  # (B, C)

        # 对所有通道和批次求平均
        return torch.mean(spectral_loss)


class SpatialFidelityLoss(nn.Module):
    """
    空间保真度损失
    确保融合结果保持MSI的高空间分辨率细节
    """

    def __init__(self):
        super(SpatialFidelityLoss, self).__init__()
        self.sobel = Sobelxy()

    def forward(self, fused, msi, gt):
        """
        Args:
            fused: (B, 31, H, W) - 融合结果
            msi: (B, 3, H, W) - 高分辨率MSI
            gt: (B, 31, H, W) - GT
        Returns:
            loss: 空间保真度损失
        """
        # 1. 梯度保持：融合结果的前3个通道应保持MSI的空间细节
        fused_rgb = fused[:, :3, :, :]
        grad_fused = self.sobel(fused_rgb)
        grad_msi = self.sobel(msi)
        grad_loss = F.l1_loss(grad_fused, grad_msi)

        # 2. 结构相似性：与GT的梯度一致
        grad_gt = self.sobel(gt[:, :3, :, :])
        struct_loss = F.l1_loss(grad_fused, grad_gt)

        return grad_loss + struct_loss


class LowPassFilterLoss(nn.Module):
    """
    低通滤波损失
    确保融合结果下采样后与HSI光谱一致
    """

    def __init__(self, scale_factor=32):  # 512/16=32
        super(LowPassFilterLoss, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, fused, hsi):
        """
        Args:
            fused: (B, 31, 512, 512) - 融合结果
            hsi: (B, 31, 16, 16) - 低分辨率HSI
        Returns:
            loss: 低通损失
        """
        # 将融合结果下采样到HSI的分辨率
        fused_downsampled = F.interpolate(
            fused,
            size=(hsi.size(2), hsi.size(3)),
            mode='bilinear',
            align_corners=False
        )

        # 计算光谱差异
        loss = F.l1_loss(fused_downsampled, hsi)

        return loss


class CombinedHSILoss(nn.Module):
    """
    组合损失函数 - 整合所有HSI-MSI专用损失
    """

    def __init__(self,
                 weight_sam=1.0,
                 weight_spectral=1.0,
                 weight_spatial=1.0,
                 weight_lowpass=1.0):
        super(CombinedHSILoss, self).__init__()

        self.sam_loss = SpectralAngleLoss()
        self.spectral_loss = SpectralConsistencyLoss()
        self.spatial_loss = SpatialFidelityLoss()
        self.lowpass_loss = LowPassFilterLoss()

        self.weight_sam = weight_sam
        self.weight_spectral = weight_spectral
        self.weight_spatial = weight_spatial
        self.weight_lowpass = weight_lowpass

    def forward(self, fused, hsi, msi, gt):
        """
        Args:
            fused: (B, 31, 512, 512) - 融合结果
            hsi: (B, 31, 16, 16) - 低分辨率HSI
            msi: (B, 3, 512, 512) - 高分辨率MSI
            gt: (B, 31, 512, 512) - GT
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典（用于日志）
        """
        # 计算各项损失
        sam = self.sam_loss(fused, gt)
        spectral = self.spectral_loss(fused, gt)
        spatial = self.spatial_loss(fused, msi, gt)
        lowpass = self.lowpass_loss(fused, hsi)

        # 加权求和
        total = (self.weight_sam * sam +
                 self.weight_spectral * spectral +
                 self.weight_spatial * spatial +
                 self.weight_lowpass * lowpass)

        # 返回总损失和详细信息
        loss_dict = {
            'sam': sam.item(),
            'spectral': spectral.item(),
            'spatial': spatial.item(),
            'lowpass': lowpass.item(),
            'total_hsi': total.item()
        }

        return total, loss_dict


# ========== 便捷函数 ==========

def get_hsi_loss_fn(device='cuda'):
    """
    获取配置好的HSI损失函数
    """
    return CombinedHSILoss(
        weight_sam=0.5,
        weight_spectral=1.0,
        weight_spatial=0.8,
        weight_lowpass=1.2
    ).to(device)