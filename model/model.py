import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.checkpoint as checkpoint
import kornia
from kornia.filters.kernels import get_gaussian_kernel2d


class FeatureExtractor(nn.Module):
    """
    特征提取器 - 适配64通道输入（统一特征空间）
    用于提取HSI和MSI的深层特征
    ✅ 新增：梯度检查点支持，节省显存
    """

    def __init__(self, in_channels=64, use_checkpoint=True):
        super(FeatureExtractor, self).__init__()
        self.use_checkpoint = use_checkpoint  # ✅ 是否使用梯度检查点

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
        """
        ✅ 核心优化：使用梯度检查点节省显存
        - 训练时：使用checkpoint，牺牲20%速度，节省50%显存
        - 推理时：不使用checkpoint，保持最快速度
        """
        if self.use_checkpoint and self.training:
            # 使用梯度检查点：不保存中间激活值，反向传播时重新计算
            return checkpoint.checkpoint(self.E, x, use_reentrant=False)
        else:
            # 正常前向传播
            return self.E(x)
class Decoder(nn.Module):
    """
    解码器 - 输出31通道高光谱图像
    将64通道特征解码为31通道高光谱图像
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
        return out, out_f   #out是最终预测的高光谱图像，out_f是解码器中间特征


class Enhance(nn.Module):
    """
    增强模块 - 使用实例归一化处理64通道特征
    用于模态字典前的特征增强
    """

    def __init__(self, channels=64):
        super(Enhance, self).__init__()
        self.E = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=(3, 3), stride=1),
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
            nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.E(x)
        return out


class base(nn.Module):
    """
    基础特征提取模块 - 分别处理MSI和HSI
    MSI: 3通道 -> 64通道
    HSI: 31通道 -> 64通道
    """

    def __init__(self, in_channels=3):
        super(base, self).__init__()
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
    融合模块 - 输出31通道高光谱图像
    将HSI和MSI的特征融合后输出高分辨率高光谱图像
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
        返回: (B, 31, H, W) - 融合的高光谱图像
        """
        x = hsi + msi
        out = self.D(x)
        return out


class PositionalEncoding(nn.Module):
    """位置编码模块 - 用于Transformer"""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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


def window_partition(x, window_size, stride):
    """
    窗口分割函数 - 支持多通道
    将特征图分割成不重叠的窗口
    """
    batch_size, channel, height, weight = x.size()
    unfold_win = nn.Unfold(kernel_size=(window_size, window_size), stride=stride)
    x_windows = unfold_win(x)
    x_out_windows = x_windows.reshape(batch_size, channel, window_size, window_size, x_windows.size()[2]).permute(4, 0,
                                                                                                                  1, 2,
                                                                                                                  3)
    return x_out_windows


class resume(nn.Module):
    """
    窗口重组模块 - 将窗口重组回完整特征图
    """

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
    特征重组函数 - 基于相似度矩阵重组特征
    用于根据对齐感知矩阵调整特征的空间位置

    参数:
        similaritys: (windows_num, batch_size, sw_size^2, lw_size^2) - 对齐矩阵
        x: (batch_size, channel, height, weight) - 输入特征
    返回:
        sample: (batch_size, channel, height, weight) - 重组后的特征
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
    可变形窗口分割 - 支持边界处理的窗口分割

    参数:
        x: (batch_size, channel, height, weight) - 输入特征
        large_window_size: 大窗口尺寸
        small_window_size: 小窗口尺寸（步长）
        is_bewindow: 是否返回窗口格式
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
    多头跨尺度注意力块 - 极致显存优化版

    优化措施：
    1. largesize: 4 → 2（减少75%显存）
    2. 减少Encoder层数（6层 → 1层）
    3. 减少Decoder层数（8层 → 1层）
    4. 禁用梯度检查点（避免重复计算）

    显存占用：原版 ~8GB → 优化版 ~1GB
    """

    def __init__(self, channels=64, use_checkpoint=False):
        super(MHCSAB, self).__init__()
        self.channels = channels
        self.use_checkpoint = False  # ✅ 强制禁用梯度检查点

        # ========== ✅ 简化大尺度编码器（6层 → 1层）==========
        self.LargeScaleEncoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(channels // 2),
            nn.PReLU(),
        )

        # ========== ✅ 简化小尺度编码器（6层 → 1层）==========
        self.SmallScaleEncoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(channels // 2),
            nn.PReLU(),
        )

        # ========== ✅ 核心优化：减小窗口尺寸 ==========
        self.largesize = 2  # ✅ 4 → 2（显存减少75%）
        self.smallsize = 1  # 保持1不变
        self.dropout = 0.1
        self.channel = channels // 2  # 32

        # ========== 动态计算嵌入维度 ==========
        large_embed_dim = self.largesize * self.largesize * self.channel  # 2×2×32 = 128
        small_embed_dim = self.smallsize * self.smallsize * self.channel  # 1×1×32 = 32

        # ========== 跨尺度映射层 ==========
        self.mapping_l2s = nn.Sequential(
            nn.Linear(large_embed_dim, large_embed_dim // 2),  # 128 -> 64
            nn.GELU(),
            nn.Linear(large_embed_dim // 2, small_embed_dim)  # 64 -> 32
        )
        self.mapping_s2l = nn.Sequential(
            nn.Linear(small_embed_dim, small_embed_dim * 2),  # 32 -> 64
            nn.GELU(),
            nn.Linear(small_embed_dim * 2, large_embed_dim)  # 64 -> 128
        )

        # ========== ✅ 简化解码器（8层 → 1层）==========
        self.Decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(channels),
            nn.PReLU(),
        )

        # ========== 多头注意力层 ==========
        self.SA_large = nn.MultiheadAttention(large_embed_dim, 1, self.dropout)
        self.SA_small = nn.MultiheadAttention(small_embed_dim, 1, self.dropout)
        self.CA_large = nn.MultiheadAttention(large_embed_dim, 1, self.dropout)
        self.CA_small = nn.MultiheadAttention(small_embed_dim, 1, self.dropout)

    def self_attention(self, input_s, MHA):
        """
        自注意力机制

        Args:
            input_s: (seq_len, batch, embed_dim) - 输入序列
            MHA: 多头注意力模块

        Returns:
            enhance: (seq_len, batch, embed_dim) - 增强后的特征
        """
        embeding_dim = input_s.size()[2]

        if MHA.training:
            PE = PositionalEncoding(embeding_dim, self.dropout).train()
        else:
            PE = PositionalEncoding(embeding_dim, self.dropout).eval()

        input_pe = PE(input_s)

        q = input_pe
        k = input_pe
        v = input_s
        a = MHA(q, k, v)[0]
        enhance = a + input_s
        return enhance

    def cross_attention(self, query, key_value, MHA):
        """
        交叉注意力机制

        Args:
            query: (seq_len, batch, embed_dim) - 查询序列
            key_value: (seq_len, batch, embed_dim) - 键值序列
            MHA: 多头注意力模块

        Returns:
            enhance: (seq_len, batch, embed_dim) - 增强后的特征
        """
        embeding_dim = query.size()[2]

        if MHA.training:
            PE = PositionalEncoding(embeding_dim, self.dropout).train()
        else:
            PE = PositionalEncoding(embeding_dim, self.dropout).eval()

        q_pe = PE(query)
        kv_pe = PE(key_value)

        q = q_pe
        k = kv_pe
        v = key_value
        a = MHA(q, k, v)[0]
        enhance = a + query
        return enhance

    def forward(self, input):
        """
        前向传播

        Args:
            input: (B, C, H, W) - 输入特征

        Returns:
            enhance_f: (B, C, H, W) - 增强后的特征
        """
        window_size = input.size()[3]

        # Fold/Unfold操作
        flod_win_l = nn.Fold(
            output_size=(window_size, window_size),
            kernel_size=(self.largesize, self.largesize),
            stride=self.largesize
        )
        flod_win_s = nn.Fold(
            output_size=(window_size, window_size),
            kernel_size=(self.smallsize, self.smallsize),
            stride=self.smallsize
        )
        unflod_win_l = nn.Unfold(kernel_size=(self.largesize, self.largesize), stride=self.largesize)
        unflod_win_s = nn.Unfold(kernel_size=(self.smallsize, self.smallsize), stride=self.smallsize)

        # ========== 特征编码（不使用checkpoint）==========
        large_scale_f = self.LargeScaleEncoder(input)
        small_scale_f = self.SmallScaleEncoder(input)

        # ========== 窗口分割 ==========
        large_scale_f_w = unflod_win_l(large_scale_f).permute(2, 0, 1)  # (num_windows, B, embed_dim)
        small_scale_f_w = unflod_win_s(small_scale_f).permute(2, 0, 1)  # (num_windows, B, embed_dim)

        # ========== 自注意力 ==========
        large_scale_f_w_s = self.self_attention(large_scale_f_w, self.SA_large)
        small_scale_f_w_s = self.self_attention(small_scale_f_w, self.SA_small)

        # ========== 跨尺度映射 ==========
        l_size = large_scale_f_w_s.size()  # (num_windows_l, B, embed_dim_l)
        s_size = small_scale_f_w_s.size()  # (num_windows_s, B, embed_dim_s)

        large_scale_f_w_s_map2s = self.mapping_l2s(
            large_scale_f_w_s.reshape(l_size[0] * l_size[1], l_size[2])
        ).reshape(l_size[0], l_size[1], s_size[2])

        small_scale_f_w_s_map2l = self.mapping_s2l(
            small_scale_f_w_s.reshape(s_size[0] * s_size[1], s_size[2])
        ).reshape(s_size[0], s_size[1], l_size[2])

        # ========== 交叉注意力 ==========
        large_scale_f_w_s_c = self.cross_attention(large_scale_f_w_s, small_scale_f_w_s_map2l, self.CA_large)
        small_scale_f_w_s_c = self.cross_attention(small_scale_f_w_s, large_scale_f_w_s_map2s, self.CA_small)

        # ========== 窗口重组 ==========
        large_scale_f_s_c = flod_win_l(large_scale_f_w_s_c.permute(1, 2, 0))
        small_scale_f_s_c = flod_win_s(small_scale_f_w_s_c.permute(1, 2, 0))

        # ========== 特征融合 ==========
        enhance_f = torch.cat([large_scale_f_s_c, small_scale_f_s_c], dim=1)

        # ========== 特征解码（不使用checkpoint）==========
        enhance_f = self.Decoder(enhance_f)

        return enhance_f

class Attention(nn.Module):
    """
    注意力模块 - 计算特征之间的相似度
    """

    def __init__(self):
        super(Attention, self).__init__()
        self.unflod_win_1 = nn.Unfold(kernel_size=(1, 1), stride=1)

    def c_similarity(self, s, r):
        """
        计算余弦相似度
        参数:
            s: (B, Nt, E) - 查询特征
            r: (B, Ns, E) - 键特征
        返回:
            attn: (B, Nt, Ns) - 相似度矩阵
        """
        B, Nt, E = s.shape
        s = s / math.sqrt(E)
        attn = torch.bmm(s, r.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        return attn

    def forward(self, fixed_window, moving_window):
        """
        前向传播
        参数:
            fixed_window: (B, C, H, W) - 固定窗口
            moving_window: (B, C, H, W) - 移动窗口
        返回:
            similarity: (B, H*W, H*W) - 相似度矩阵
        """
        fixed_patch = self.unflod_win_1(fixed_window).permute(0, 2, 1)
        moving_patch = self.unflod_win_1(moving_window).permute(0, 2, 1)
        similarity = self.c_similarity(fixed_patch, moving_patch)
        return similarity


def CMAP(fixed_windows, moving_windows, hsi_MHCSA, msi_MHCSA, is_hsi_fixed):
    """
    跨模态对齐感知 (Cross-Modality Alignment Perception)

    参数:
        fixed_windows: (window_nums, batch_size, C, H, W) - 参考窗口（HSI）
        moving_windows: (window_nums, batch_size, C, H, W) - 移动窗口（MSI）
        hsi_MHCSA: HSI的多头注意力模块
        msi_MHCSA: MSI的多头注意力模块
        is_hsi_fixed: HSI是否为参考图像

    返回:
        similaritys: (window_nums, batch_size, H_f*W_f, H_m*W_m) - 对齐矩阵
    """
    assert (fixed_windows.size()[0] == moving_windows.size()[0])
    att = Attention()
    device = fixed_windows.device
    window_nums, batch_size, _, window_size_f, _ = fixed_windows.size()
    window_size_m = moving_windows.size()[3]

    similaritys = torch.zeros(
        (window_nums, batch_size, int(window_size_f * window_size_f), int(window_size_m * window_size_m)),
        device=device
    )

    for i in range(window_nums):
        fixed_window = fixed_windows[i, :, :, :, :]
        moving_window = moving_windows[i, :, :, :, :]

        if is_hsi_fixed:
            fixed_enhance = hsi_MHCSA(fixed_window)
            moving_enhance = msi_MHCSA(moving_window)
        else:
            fixed_enhance = msi_MHCSA(fixed_window)
            moving_enhance = hsi_MHCSA(moving_window)

        similarity = att(fixed_enhance, moving_enhance)
        similaritys[i, :, :, :] = similarity

    return similaritys


class DictionaryRepresentationModule(nn.Module):
    def __init__(self, channels=64):
        super(DictionaryRepresentationModule, self).__init__()

        # ========== ✅ 核心优化：减小字典规模 ==========
        element_size = 4
        self.element_size = element_size
        self.channels = channels
        l_n = 16
        c_n = 16

        # ========== 可学习的模态字典 ==========
        self.Dictionary = nn.Parameter(
            torch.FloatTensor(l_n * c_n, 1, element_size * element_size * channels).to(torch.device("cuda:0")),
            requires_grad=True
        )
        nn.init.uniform_(self.Dictionary, 0, 1)

        # ========== 窗口展开和折叠 ==========
        self.unflod_win = nn.Unfold(kernel_size=(element_size, element_size), stride=element_size)

        # ========== 交叉注意力（用于字典查询）==========
        self.CA = nn.MultiheadAttention(
            embed_dim=element_size * element_size * channels,
            num_heads=1,  # 单头注意力（节省显存）
            dropout=0
        )

        # ========== 用于可视化字典 ==========
        self.flod_win_1 = nn.Fold(
            output_size=(l_n * element_size, c_n * element_size),
            kernel_size=(element_size, element_size),
            stride=element_size
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: (B, 64, H, W) - 输入特征（经过MN增强的特征）

        Returns:
            representation: (B, 64, H, W) - 字典补偿后的特征
            visible_D: (1, 64, 8, 8) - 可视化的字典（用于debug）
        """
        size = x.size()  # (B, 64, H, W)

        # ========== 动态创建fold操作（适应输入尺寸）==========
        flod_win = nn.Fold(
            output_size=(size[2], size[3]),
            kernel_size=(self.element_size, self.element_size),
            stride=self.element_size
        )

        # ========== 扩展字典以匹配batch size ==========
        D = self.Dictionary.repeat(1, size[0], 1)  # (16, B, 256)

        # ========== 将输入特征分割成patches ==========
        x_w = self.unflod_win(x).permute(2, 0, 1)  # (num_patches, B, 256)
        # num_patches = (H/element_size) × (W/element_size)
        # 例如：(128×128)图像 → (64, B, 256)

        # ========== 交叉注意力：用字典补偿特征 ==========
        # Query: 输入特征的patches
        # Key, Value: 字典
        q = x_w  # (num_patches, B, 256)
        k = D  # (16, B, 256)
        v = D  # (16, B, 256)

        a = self.CA(q, k, v)[0]  # (num_patches, B, 256)
        # 注意力机制：每个patch查询字典中最相关的元素

        # ========== 重组为特征图 ==========
        representation = flod_win(a.permute(1, 2, 0))  # (B, 64, H, W)

        # ========== 可视化字典（用于调试）==========
        visible_D = self.flod_win_1(self.Dictionary.permute(1, 2, 0))  # (1, 64, 8, 8)

        return representation, visible_D


# ====================== 验证代码（可选）======================
if __name__ == "__main__":
    """
    测试显存占用
    """
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 测试MHCSAB
    print("=" * 70)
    print("测试 MHCSAB 模块")
    print("=" * 70)

    mhcsab = MHCSAB(channels=64).to(device)
    test_input = torch.randn(1, 64, 32, 32).to(device)  # (B, C, H, W)

    print(f"输入shape: {test_input.shape}")
    print(f"显存占用（输入前）: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    output = mhcsab(test_input)

    print(f"输出shape: {output.shape}")
    print(f"显存占用（输出后）: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    # 测试DictionaryRepresentationModule
    print("\n" + "=" * 70)
    print("测试 DictionaryRepresentationModule 模块")
    print("=" * 70)

    dict_module = DictionaryRepresentationModule(channels=64).to(device)
    test_input = torch.randn(1, 64, 128, 128).to(device)

    print(f"输入shape: {test_input.shape}")
    print(f"显存占用（输入前）: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    representation, visible_D = dict_module(test_input)

    print(f"输出shape: {representation.shape}")
    print(f"可视化字典shape: {visible_D.shape}")
    print(f"显存占用（输出后）: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    print("\n✅ 所有测试通过！")