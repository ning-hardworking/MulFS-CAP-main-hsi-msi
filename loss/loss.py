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
    强度损失
    注意：现在hsi是31通道，msi是3通道，需要特殊处理
    """
    # 对于HSI-MSI融合，融合结果应该接近GT（31通道）
    # 这里假设image_fused是31通道的融合结果
    hsi_li = F.l1_loss(image_fused, hsi)
    # MSI只有3通道，可以只比较前3个通道
    msi_li = F.l1_loss(image_fused[:, :3, :, :], msi)
    li = hsi_li + msi_li
    return li


class L_Grad(nn.Module):
    """
    梯度损失 - 已适配多通道
    """

    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, img1, img2, image_fused=None):
        """
        对于HSI-MSI：
        - img1: GT (31, H, W) 或 HSI
        - img2: GT (31, H, W) 或 MSI
        - image_fused: 融合结果 (31, H, W)
        """
        if image_fused is None:
            # 计算所有通道的平均梯度
            gradient_1 = self.sobelconv(img1)
            gradient_2 = self.sobelconv(img2)
            Loss_gradient = F.l1_loss(gradient_1, gradient_2)
            return Loss_gradient
        else:
            # 计算融合图像与输入图像的梯度损失
            gradient_1 = self.sobelconv(img1)
            gradient_2 = self.sobelconv(img2)
            gradient_fused = self.sobelconv(image_fused)
            gradient_joint = torch.max(gradient_1, gradient_2)
            Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
            return Loss_gradient


class Sobelxy(nn.Module):
    """
    Sobel算子 - 支持多通道
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
        输入: (B, C, H, W)
        输出: (B, C, H, W) - 每个通道的梯度
        """
        B, C, H, W = x.shape

        # 对每个通道分别计算梯度
        sobelx_list = []
        sobely_list = []
        for i in range(C):
            channel = x[:, i:i + 1, :, :]
            sobelx = F.conv2d(channel, self.weightx, padding=1)
            sobely = F.conv2d(channel, self.weighty, padding=1)
            sobelx_list.append(sobelx)
            sobely_list.append(sobely)

        sobelx = torch.cat(sobelx_list, dim=1)
        sobely = torch.cat(sobely_list, dim=1)

        return torch.abs(sobelx) + torch.abs(sobely)


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