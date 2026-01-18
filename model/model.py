import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import kornia
from kornia.filters.kernels import get_gaussian_kernel2d


class FeatureExtractor(nn.Module):
    """
    特征提取器 - 适配多通道输入
    原始：8通道输入
    现在：需要处理MSI(3通道)和HSI(31通道)的共同特征空间(64通道)
    """

    def __init__(self, in_channels=64):
        super(FeatureExtractor, self).__init__()
        self.E = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.E(x)
        return out


class Decoder(nn.Module):
    """
    解码器 - 输出31通道高光谱图像
    原始：输出1通道灰度图
    现在：输出31通道高光谱图像
    """

    def __init__(self, out_channels=31):
        super(Decoder, self).__init__()
        self.D_0 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
        )
        self.D = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=(3, 3), stride=1),
        )

    def forward(self, x):
        out_f = self.D_0(x)
        out = self.D(out_f)
        return out, out_f


class Enhance(nn.Module):
    """
    增强模块 - 保持64通道
    """

    def __init__(self):
        super(Enhance, self).__init__()
        self.E = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.E(x)
        return out


class base(nn.Module):
    """
    基础特征提取 - 适配不同输入通道
    MSI: 3通道 -> 64通道
    HSI: 31通道 -> 64通道
    """

    def __init__(self, in_channels=3, is_hsi=False):
        super(base, self).__init__()
        if is_hsi:
            in_channels = 31  # HSI输入
        else:
            in_channels = 3  # MSI输入

        self.B = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.B(x)
        return out


class FusionMoudle(nn.Module):
    """
    融合模块 - 输出31通道
    """

    def __init__(self, out_channels=31):
        super(FusionMoudle, self).__init__()
        self.D = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=(3, 3), stride=1),
        )

    def forward(self, hsi, msi):
        """
        hsi: (B, 64, H, W) - HSI特征
        msi: (B, 64, H, W) - MSI特征
        """
        x = hsi + msi
        out = self.D(x)
        return out


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        device = x.device
        x = x + self.pe[:, :x.size(1)].to(device)
        return self.dropout(x)


class AffineTransform(nn.Module):
    """
    仿射变换 - 支持多通道
    """

    def __init__(self, degrees=0, translate=0.1, return_warp=False):
        super(AffineTransform, self).__init__()
        self.trs = kornia.augmentation.RandomAffine(degrees, (translate, translate), return_transform=True, p=1)
        self.return_warp = return_warp

    def forward(self, input):
        device = input.device
        batch_size, _, height, weight = input.shape

        # 仿射变换
        warped, affine_param = self.trs(input)

        T = torch.FloatTensor([[2. / weight, 0, -1],
                               [0, 2. / height, -1],
                               [0, 0, 1]]).repeat(batch_size, 1, 1).to(device)
        theta = torch.inverse(torch.bmm(torch.bmm(T, affine_param), torch.inverse(T)))

        base = kornia.utils.create_meshgrid(height, weight, device=device).to(input.dtype)
        grid = F.affine_grid(theta[:, :2, :], size=input.size(), align_corners=False)

        disp = grid - base

        if self.return_warp:
            warped_grid_sample = F.grid_sample(input, grid, align_corners=False, mode='bilinear')
            return warped_grid_sample, disp
        else:
            return disp


class ElasticTransform(nn.Module):
    """
    弹性变换 - 支持多通道
    """

    def __init__(self, kernel_size=63, sigma=32, align_corners=False, mode="bilinear", return_warp=False):
        super(ElasticTransform, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.align_corners = align_corners
        self.mode = mode
        self.return_warp = return_warp

    def forward(self, input):
        batch_size, _, height, weight = input.shape
        device = input.device
        noise = torch.rand(batch_size, 2, height, weight, device=device) * 2 - 1

        if self.return_warp:
            warped, disp = self.elastic_transform2d(input, noise)
            return warped, disp
        else:
            disp = self.elastic_transform2d(input, noise)
            return disp

    def elastic_transform2d(self, image: torch.Tensor, noise: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

        if not isinstance(noise, torch.Tensor):
            raise TypeError(f"Input noise is not torch.Tensor. Got {type(noise)}")

        if not len(image.shape) == 4:
            raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

        if not len(noise.shape) == 4 or noise.shape[1] != 2:
            raise ValueError(f"Invalid noise shape, we expect Bx2xHxW. Got: {noise.shape}")

        device = image.device

        kernel_x: torch.Tensor = get_gaussian_kernel2d((self.kernel_size, self.kernel_size), (self.sigma, self.sigma))[
            None]
        kernel_y: torch.Tensor = get_gaussian_kernel2d((self.kernel_size, self.kernel_size), (self.sigma, self.sigma))[
            None]

        disp_x: torch.Tensor = noise[:, :1].to(device)
        disp_y: torch.Tensor = noise[:, 1:].to(device)

        disp_x = kornia.filters.filter2d(disp_x, kernel=kernel_y, border_type="constant")
        disp_y = kornia.filters.filter2d(disp_y, kernel=kernel_x, border_type="constant")

        disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)

        if self.return_warp:
            b, c, h, w = image.shape
            base = kornia.utils.create_meshgrid(h, w, device=image.device).to(image.dtype)
            grid = (base + disp).clamp(-1, 1)
            warped = F.grid_sample(image, grid, align_corners=self.align_corners, mode=self.mode)
            return warped, disp
        else:
            return disp


class ImageTransform(nn.Module):
    """
    图像变换 - 支持多通道HSI和MSI
    """

    def __init__(self, ET_kernel_size=101, ET_kernel_sigma=16, AT_translate=0.01):
        super(ImageTransform, self).__init__()
        self.affine = AffineTransform(translate=AT_translate)
        self.elastic = ElasticTransform(kernel_size=ET_kernel_size, sigma=ET_kernel_sigma)

    def generate_grid(self, input):
        device = input.device
        batch_size, _, height, weight = input.size()

        # 仿射变换
        affine_disp = self.affine(input)
        # 弹性变换
        elastic_disp = self.elastic(input)
        # 生成网格
        base = kornia.utils.create_meshgrid(height, weight).to(dtype=input.dtype).repeat(batch_size, 1, 1, 1).to(device)
        disp = affine_disp + elastic_disp
        grid = base + disp
        return grid

    def make_transform_matrix(self, grid):
        device = grid.device
        batch_size, height, weight, _ = grid.size()
        grid_s = torch.zeros_like(grid)
        grid_s[:, :, :, 0] = ((grid[:, :, :, 0] / 2) + 0.5) * (weight - 1)
        grid_s[:, :, :, 1] = ((grid[:, :, :, 1] / 2) + 0.5) * (height - 1)
        grid_s = torch.round(grid_s).to(dtype=torch.int64)
        base_s = kornia.utils.create_meshgrid(height, weight, normalized_coordinates=False).to(dtype=grid.dtype).repeat(
            batch_size, 1, 1, 1).to(device)
        x_index = base_s.reshape([batch_size, height * weight, 2]).to(dtype=torch.int64)
        y_index = grid_s.reshape([batch_size, height * weight, 2]).to(dtype=torch.int64)
        x_index_o = x_index[:, :, 0] + x_index[:, :, 1] * height
        y_index_o = y_index[:, :, 0] + y_index[:, :, 1] * height

        mask_x_min = (y_index[:, :, 0] > -1).to(torch.int64).unsqueeze(dim=-1).repeat(1, 1, 2)
        mask_y_min = (y_index[:, :, 1] > -1).to(torch.int64).unsqueeze(dim=-1).repeat(1, 1, 2)
        mask_x_max = (y_index[:, :, 0] < weight).to(torch.int64).unsqueeze(dim=-1).repeat(1, 1, 2)
        mask_y_max = (y_index[:, :, 1] < height).to(torch.int64).unsqueeze(dim=-1).repeat(1, 1, 2)
        mask = torch.mul(torch.mul(mask_x_min, mask_y_min), torch.mul(mask_x_max, mask_y_max))
        x_index_o = torch.mul(x_index_o, mask[:, :, 0])
        y_index_o = torch.mul(y_index_o, mask[:, :, 0])
        filler = mask[:, :, 0].to(dtype=torch.float32)

        index = torch.cat([x_index_o.unsqueeze(dim=1), y_index_o.unsqueeze(dim=1)], dim=1)
        index_r = torch.cat([y_index_o.unsqueeze(dim=1), x_index_o.unsqueeze(dim=1)], dim=1)

        return index, index_r, filler

    def forward(self, image_1, image_2):
        """
        image_1: HSI (B, 31, H, W)
        image_2: MSI (B, 3, H, W)
        注意：需要确保两者空间分辨率一致（通过上采样HSI或下采样MSI）
        """
        # ⚠️ 关键修改：需要先将HSI上采样到MSI的分辨率
        if image_1.size(2) != image_2.size(2) or image_1.size(3) != image_2.size(3):
            image_1 = F.interpolate(image_1, size=(image_2.size(2), image_2.size(3)),
                                    mode='bilinear', align_corners=False)

        # 生成变换网格（使用MSI的分辨率）
        grid = self.generate_grid(image_2)

        # 生成变换矩阵
        index, index_r, filler = self.make_transform_matrix(grid)

        # 应用变换
        image_1_warp = F.grid_sample(image_1, grid, align_corners=False, mode='bilinear')
        image_2_warp = F.grid_sample(image_2, grid, align_corners=False, mode='bilinear')

        return image_1_warp, image_2_warp, index, index_r, filler


def window_partition(x, window_size, stride):
    """
    窗口分割 - 支持多通道
    """
    batch_size, channel, height, weight = x.size()
    unfold_win = nn.Unfold(kernel_size=(window_size, window_size), stride=stride)
    x_windows = unfold_win(x)
    x_out_windows = x_windows.reshape(batch_size, channel, window_size, window_size, x_windows.size()[2]).permute(4, 0,
                                                                                                                  1, 2,
                                                                                                                  3)
    return x_out_windows


class resume(nn.Module):
    def __init__(self, height, weight, window_size, stride, channel):
        super(resume, self).__init__()
        self.channel = channel
        self.window_size = window_size
        self.flod_win = nn.Fold(output_size=(height, weight), kernel_size=(window_size, window_size), stride=stride)

    def forward(self, x_windows):
        size = x_windows.size()
        x_out = x_windows.permute(1, 2, 3, 4, 0)
        x_out = x_out.reshape(size[1], self.channel * self.window_size * self.window_size, size[0])
        r_out = self.flod_win(x_out)
        return r_out


def feature_reorganization(similaritys, x):
    """
    特征重组 - 支持多通道
    """
    device = similaritys.device
    windows_num, batch_size, sw_size_pow2, lw_size_pow2 = similaritys.size()
    sw_size = int(math.sqrt(sw_size_pow2))
    lw_size = int(math.sqrt(lw_size_pow2))
    _, channel, height, weight = x.size()

    fold = nn.Fold(output_size=(sw_size, sw_size), kernel_size=(1, 1), stride=1)
    unflod_win = nn.Unfold(kernel_size=(1, 1), stride=1)
    resume_sw = resume(height, weight, sw_size, sw_size, channel)

    x_windows = df_window_partition(x, lw_size, sw_size)

    sample_windows = torch.zeros(windows_num, batch_size, channel, sw_size, sw_size, device=device)

    for i in range(windows_num):
        for j in range(batch_size):
            x_window = x_windows[i, j, :, :, :].unsqueeze(dim=0)
            x_patchs = unflod_win(x_window).permute(0, 2, 1)
            similarity = similaritys[i, j]
            sample_patch = torch.bmm(similarity.unsqueeze(dim=0), x_patchs).permute(0, 2, 1)
            sample_window = fold(sample_patch)
            sample_windows[i, j, :, :, :] = sample_window.squeeze(dim=0)

    sample = resume_sw(sample_windows)

    return sample


def df_window_partition(x, large_window_size, small_window_size, is_bewindow=True):
    """
    可变形窗口分割 - 支持多通道
    """
    batch_size, channel, height, weight = x.size()
    padding_num = int((large_window_size - small_window_size) / 2)
    center_unfold = nn.Unfold(kernel_size=(large_window_size, large_window_size), stride=small_window_size)
    x_center_w = center_unfold(F.pad(x, pad=[padding_num, padding_num, padding_num, padding_num]))

    corner_unfold = nn.Unfold(kernel_size=(large_window_size, large_window_size),
                              stride=(height - large_window_size, weight - large_window_size))
    x_corner_w = corner_unfold(x)
    top_bottom_unfold = nn.Unfold(kernel_size=(large_window_size, large_window_size),
                                  stride=(height - large_window_size, small_window_size))

    x_top_bottom_w = top_bottom_unfold(F.pad(x, pad=[padding_num, padding_num, 0, 0]))

    left_right_unfold = nn.Unfold(kernel_size=(large_window_size, large_window_size),
                                  stride=(small_window_size, weight - large_window_size))
    x_left_right_w = left_right_unfold(F.pad(x, pad=[0, 0, padding_num, padding_num]))

    weight_block_num = int(weight / small_window_size)
    height_block_num = int(height / small_window_size)

    m = torch.ones((1, 1, height_block_num, weight_block_num))
    m_unfold = nn.Unfold(kernel_size=(2, 2), stride=1)
    m_fold = nn.Fold(output_size=(height_block_num, weight_block_num), kernel_size=(2, 2), stride=1)
    mask = m_fold(m_unfold(m))

    mask[:, :, 0, :] = 3
    mask[:, :, height_block_num - 1, :] = 3
    mask[:, :, height_block_num - 1, weight_block_num - 1] = 1
    mask[:, :, 0, weight_block_num - 1] = 1
    mask[:, :, height_block_num - 1, 0] = 1
    mask[:, :, 0, 0] = 1

    windows = torch.zeros_like(x_center_w)

    lr_index = 2
    tb_index = 1
    corner_index = 0
    for i in range(height_block_num):
        for j in range(weight_block_num):
            index = i * weight_block_num + j
            c = mask[0, 0, i, j]
            if c == 4:
                windows[:, :, index] = x_center_w[:, :, index]
            elif c == 2:
                windows[:, :, index] = x_left_right_w[:, :, lr_index]
                lr_index += 1
            elif c == 3:
                if tb_index == weight_block_num - 1:
                    tb_index += 2
                windows[:, :, index] = x_top_bottom_w[:, :, tb_index]
                tb_index += 1
            elif c == 1:
                windows[:, :, index] = x_corner_w[:, :, corner_index]
                corner_index += 1

    if is_bewindow:
        out_windows = windows.reshape(batch_size, channel, large_window_size, large_window_size,
                                      windows.size()[2]).permute(4, 0, 1, 2, 3)
    else:
        out_windows = windows
    return out_windows


class MHCSAB(nn.Module):
    """
    多头跨尺度注意力块 - 适配64通道特征
    """

    def __init__(self, channels=64):
        super(MHCSAB, self).__init__()
        self.channels = channels

        self.LargeScaleEncoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(channels // 2),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=channels // 2, out_channels=channels // 2, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(channels // 2),
            nn.PReLU(),
        )
        self.SmallScaleEncoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(channels // 2),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=channels // 2, out_channels=channels // 2, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(channels // 2),
            nn.PReLU(),
        )
        self.mapping_l2s = nn.Sequential(
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 4)
        )
        self.mapping_s2l = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 64)
        )
        self.Decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(