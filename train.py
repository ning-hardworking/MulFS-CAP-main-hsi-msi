import os
import time
from pathlib import Path

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import scipy.io as sio
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

from utils.utils import save_img
from tqdm import tqdm

import args
from loss import loss as Loss
from model import model
from utils import utils

# 全局常量定义
model_name = "MulFS-CAP-HSI-MSI"
device_id = "0"


def adjust_learning_rate(optimizer, epoch_count):
    lr = args.args.LR + 0.5 * (args.args.LR_target - args.args.LR) * (
            1 + math.cos((epoch_count - args.args.Warm_epoch) / (args.args.Epoch - args.args.Warm_epoch) * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def warmup_learning_rate(optimizer, epoch_count):
    lr = epoch_count * ((args.args.LR_target - args.args.LR) / args.args.Warm_epoch) + args.args.LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class TrainDataset(data.Dataset):
    def __init__(self, hsi_dir, msi_dir, gt_dir,
                 hsi_deformed_dir, msi_deformed_dir, gt_deformed_dir,
                 transform=None):
        super(TrainDataset, self).__init__()

        # Pair 1: 原始配准数据
        self.hsi_paths = self.find_mat_files(hsi_dir)
        self.msi_paths = self.find_mat_files(msi_dir)
        self.gt_paths = self.find_mat_files(gt_dir)

        # Pair 2: 形变配准数据
        self.hsi_d_paths = self.find_mat_files(hsi_deformed_dir)
        self.msi_d_paths = self.find_mat_files(msi_deformed_dir)
        self.gt_d_paths = self.find_mat_files(gt_deformed_dir)

        assert len(self.hsi_paths) == len(self.hsi_d_paths), \
            f"配对数据数量不一致: Pair1={len(self.hsi_paths)}, Pair2={len(self.hsi_d_paths)}"

        self.transform = transform
        print(f"✅ 数据集加载成功:")
        print(f"   - Pair 1 (原始配准): {len(self.hsi_paths)} 对样本")
        print(f"   - Pair 2 (形变配准): {len(self.hsi_d_paths)} 对样本")

    def find_mat_files(self, dir_path):
        """查找所有.mat文件"""
        mat_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
        mat_files.sort()
        return mat_files

    def read_mat_image(self, path, key=None):
        """
        读取.mat文件中的图像数据（增强版：自动识别通道维度）

        参数:
            path: .mat文件路径
            key: .mat文件中的变量名（如果为None，则自动查找）

        返回:
            torch.Tensor: shape为(C, H, W)的张量
        """
        try:
            mat_data = sio.loadmat(path)

            # 自动查找数据键（排除MATLAB元数据）
            if key is None:
                valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if len(valid_keys) == 0:
                    raise ValueError(f"未找到有效数据键: {path}")
                key = valid_keys[0]

            img = mat_data[key]  # numpy数组: 可能是 (H,W,C) 或 (C,H,W) 或其他

            # ========== 🔥 关键修复：智能识别通道维度 🔥 ==========
            if img.ndim == 2:
                # 2D图像 -> 添加通道维度
                img = torch.from_numpy(img).float().unsqueeze(0)  # (H, W) -> (1, H, W)

            elif img.ndim == 3:
                # 3D图像 -> 需要识别哪个维度是通道
                shape = img.shape

                # 🔥 策略1: 找到最小的维度作为通道（通常通道数最小）
                min_dim_idx = np.argmin(shape)

                # 🔥 策略2: 验证是否符合预期的通道数 (3 或 31)
                expected_channels = [3, 31]
                channel_dim_idx = None

                for i, s in enumerate(shape):
                    if s in expected_channels:
                        channel_dim_idx = i
                        break

                # 优先使用策略2，如果找不到则使用策略1
                if channel_dim_idx is not None:
                    target_dim = channel_dim_idx
                else:
                    target_dim = min_dim_idx

                # 将通道维度移到第0位
                img = np.moveaxis(img, source=target_dim, destination=0)
                img = torch.from_numpy(img).float()  # (C, H, W)

            else:
                raise ValueError(f"不支持的图像维度: {img.ndim}D，路径: {path}")

            # ========== 归一化到[0, 1] ==========
            if img.max() > 1.0:
                img = img / img.max()

            # ========== 应用transform（如果需要resize）==========
            if self.transform is not None:
                img = self.transform(img)

            # ========== 🔥 最终验证：打印第一个样本的维度 🔥 ==========
            if not hasattr(self, '_first_load_done'):
                print(f"\n✅ 数据加载验证 (文件: {os.path.basename(path)}):")
                print(f"   原始shape: {mat_data[key].shape}")
                print(f"   处理后:    {img.shape}")
                print(f"   预期格式:  (C, H, W) 其中 C∈{3, 31}, H,W∈{16, 512}\n")
                self._first_load_done = True

            return img

        except Exception as e:
            print(f"❌ 读取文件失败: {path}")
            print(f"   错误信息: {str(e)}")
            raise

    def __getitem__(self, index):
        # Pair 1: 原始配准对
        hsi_1 = self.read_mat_image(self.hsi_paths[index])  # (31, 16, 16)
        msi_1 = self.read_mat_image(self.msi_paths[index])  # (3, 512, 512)
        gt_1 = self.read_mat_image(self.gt_paths[index])  # (31, 512, 512)

        # Pair 2: 形变配准对
        hsi_2 = self.read_mat_image(self.hsi_d_paths[index])  # (31, 16, 16)
        msi_2 = self.read_mat_image(self.msi_d_paths[index])  # (3, 512, 512)
        gt_2 = self.read_mat_image(self.gt_d_paths[index])  # (31, 512, 512)

        return hsi_1, msi_1, gt_1, hsi_2, msi_2, gt_2

    def __len__(self):
        return len(self.hsi_paths)


# 核心：所有执行逻辑必须包裹到if __name__ == '__main__'中
if __name__ == '__main__':
    # ========== 初始化环境 ==========
    os.environ['CUDA_LAUNCH_BLOCKING'] = device_id
    device = torch.device("cuda:" + device_id if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # ========== 初始化保存目录 ==========
    now = int(time.time())
    timeArr = time.localtime(now)
    nowTime = time.strftime("%Y%m%d_%H-%M-%S", timeArr)
    save_model_dir = args.args.train_save_model_dir + "/" + nowTime + "_" + model_name + "_model"
    save_img_dir = args.args.train_save_img_dir + "/" + nowTime + "_" + model_name + "_img"
    utils.check_dir(save_model_dir)
    utils.check_dir(save_img_dir)

    # ========== 数据加载器初始化 ==========
    tf = None

    # ✅ 修改数据集初始化
    dataset = TrainDataset(
        args.args.hsi_train_dir,  # Z_reconst/
        args.args.msi_train_dir,  # Y_reconst/
        args.args.gt_train_dir,  # X/
        args.args.hsi_deformed_train_dir,  # Z_deformed/
        args.args.msi_deformed_train_dir,  # Y_deformed/
        args.args.gt_deformed_train_dir,  # X_deformed/
        tf
    )

    data_iter = data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=args.args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        multiprocessing_context=torch.multiprocessing.get_context('spawn')
    )

    iter_num = int(dataset.__len__() / args.args.batch_size)
    save_image_iter = max(1, int(iter_num / args.args.save_image_num))

    # ========== 模型初始化 ==========
    print("🔧 正在初始化模型...")
    Lgrad = Loss.L_Grad().to(device)
    CC = Loss.CorrelationCoefficient().to(device)
    Lcorrespondence = Loss.L_correspondence()

    # ✅ 关键修复1: 创建两个不同的base模块
    with torch.no_grad():
        base_msi = model.base(in_channels=3)  # MSI: 3通道 -> 64通道
        base_hsi = model.base(in_channels=31)  # HSI: 31通道 -> 64通道
        hsi_MFE = model.FeatureExtractor()
        msi_MFE = model.FeatureExtractor()
        fusion_decoder = model.Decoder()
        PAFE = model.FeatureExtractor()
        decoder = model.Decoder()
        MN_hsi = model.Enhance()
        MN_msi = model.Enhance()
        HSIDP = model.DictionaryRepresentationModule()
        MSIDP = model.DictionaryRepresentationModule()
        ImageDeformation = model.ImageTransform()
        MHCSA_hsi = model.MHCSAB()
        MHCSA_msi = model.MHCSAB()
        fusion_module = model.FusionMoudle()

    # 模型训练模式+设备迁移
    print("📦 正在加载模型到GPU...")
    base_msi.train().to(device)
    base_hsi.train().to(device)
    hsi_MFE.train().to(device)
    msi_MFE.train().to(device)
    fusion_decoder.train().to(device)
    PAFE.train().to(device)
    decoder.train().to(device)
    HSIDP.train().to(device)
    MSIDP.train().to(device)
    MN_hsi.train().to(device)
    MN_msi.train().to(device)
    MHCSA_hsi.train().to(device)
    MHCSA_msi.train().to(device)
    fusion_module.train().to(device)

    # ========== 优化器初始化 ==========
    print("⚙️ 正在配置优化器...")
    optimizer_FE = torch.optim.Adam([
        {'params': base_msi.parameters()},  # ✅ 修复2: 包含两个base
        {'params': base_hsi.parameters()},
        {'params': hsi_MFE.parameters()},
        {'params': msi_MFE.parameters()},
        {'params': fusion_decoder.parameters()},
        {'params': PAFE.parameters()},
        {'params': decoder.parameters()},
        {'params': MN_hsi.parameters()},
        {'params': MN_msi.parameters()}
    ], lr=0.0002)

    optimizer_HSIDP = torch.optim.Adam(HSIDP.parameters(), lr=0.0008)
    optimizer_MSIDP = torch.optim.Adam(MSIDP.parameters(), lr=0.0008)
    optimizer_MHCSAhsi = torch.optim.Adam(MHCSA_hsi.parameters(), lr=args.args.LR)
    optimizer_MHCSAmsi = torch.optim.Adam(MHCSA_msi.parameters(), lr=args.args.LR)
    optimizer_FusionModule = torch.optim.Adam(fusion_module.parameters(), lr=0.0002)

    # ✅ 优化3: 混合精度训练
    scaler = GradScaler()
    print("✅ 已启用混合精度训练（AMP），显存占用将减少约50%")


    # ========== 训练函数定义 ==========
    def train(epoch):
        """
        训练函数
        处理两对配准数据：
        - Pair 1: (hsi_1, msi_1, gt_1) - 原始配准对（来自Z_reconst, Y_reconst, X）
        - Pair 2: (hsi_2, msi_2, gt_2) - 形变配准对（来自Z_deformed, Y_deformed, X_deformed）

        核心思路：
        1. 分别提取两对配准数据的特征
        2. 用Pair1的HSI特征 + Pair2的MSI特征 构造未配准对
        3. 通过跨模态对齐感知学习对齐关系
        4. 生成最终的融合结果
        """
        epoch_loss_HSIDP = []
        epoch_loss_MSIDP = []
        epoch_loss_same = []
        epoch_loss_fusion_total = []

        for step, x in enumerate(data_iter):
            # ========== 数据加载（6个张量）==========
            hsi_1, msi_1, gt_1, hsi_2, msi_2, gt_2 = [
                item.to(device, non_blocking=True) for item in x
            ]

            # 打印维度（仅第一个batch）
            if step == 0 and epoch == 0:
                print(f"\n✅ 数据维度验证:")
                print(f"   Pair 1: HSI={hsi_1.shape}, MSI={msi_1.shape}, GT={gt_1.shape}")
                print(f"   Pair 2: HSI={hsi_2.shape}, MSI={msi_2.shape}, GT={gt_2.shape}")

            # ========== 上采样HSI到MSI的分辨率 ==========
            hsi_1_up = F.interpolate(
                hsi_1,
                size=(msi_1.size(2), msi_1.size(3)),
                mode='bilinear',
                align_corners=False
            )
            hsi_2_up = F.interpolate(
                hsi_2,
                size=(msi_2.size(2), msi_2.size(3)),
                mode='bilinear',
                align_corners=False
            )

            # ========== 混合精度训练 ==========
            with autocast():
                # ====================================================================
                # 阶段1: 基础特征提取（64通道统一特征空间）
                # ====================================================================
                hsi_1_base = base_hsi(hsi_1_up)  # (B, 31, 512, 512) -> (B, 64, 512, 512)
                msi_1_base = base_msi(msi_1)  # (B, 3, 512, 512)  -> (B, 64, 512, 512)
                hsi_2_base = base_hsi(hsi_2_up)  # (B, 31, 512, 512) -> (B, 64, 512, 512)
                msi_2_base = base_msi(msi_2)  # (B, 3, 512, 512)  -> (B, 64, 512, 512)

                # 释放不需要的上采样结果
                del hsi_1_up, hsi_2_up
                torch.cuda.empty_cache()

                # ====================================================================
                # 阶段2: 深层特征提取（用于融合重建）
                # ====================================================================
                # Pair 1 的深层特征
                hsi_1_fe = hsi_MFE(hsi_1_base)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                msi_1_fe = msi_MFE(msi_1_base)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                simple_fusion_f_1 = hsi_1_fe + msi_1_fe
                fusion_image_1, fusion_f_1 = fusion_decoder(simple_fusion_f_1)  # -> (B, 31, 512, 512)

                # Pair 2 的深层特征
                hsi_2_fe = hsi_MFE(hsi_2_base)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                msi_2_fe = msi_MFE(msi_2_base)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                simple_fusion_f_2 = hsi_2_fe + msi_2_fe
                fusion_image_2, fusion_f_2 = fusion_decoder(simple_fusion_f_2)  # -> (B, 31, 512, 512)

                del simple_fusion_f_1, simple_fusion_f_2
                torch.cuda.empty_cache()

                # ====================================================================
                # 阶段3: PAFE特征提取（用于对齐感知）
                # ====================================================================
                # Pair 1 的PAFE特征
                hsi_1_f = PAFE(hsi_1_base)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                msi_1_f = PAFE(msi_1_base)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                simple_fusion_pf_1 = hsi_1_f + msi_1_f
                fusion_pimage_1, fusion_pf_1 = decoder(simple_fusion_pf_1)  # -> (B, 31, 512, 512)

                # Pair 2 的PAFE特征
                hsi_2_f = PAFE(hsi_2_base)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                msi_2_f = PAFE(msi_2_base)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                simple_fusion_pf_2 = hsi_2_f + msi_2_f
                fusion_pimage_2, fusion_pf_2 = decoder(simple_fusion_pf_2)  # -> (B, 31, 512, 512)

                del simple_fusion_pf_1, simple_fusion_pf_2
                del hsi_1_base, hsi_2_base, msi_1_base, msi_2_base
                torch.cuda.empty_cache()

                # ====================================================================
                # 阶段4: 模态归一化（Modality Normalization）
                # ====================================================================
                hsi_1_e_f = MN_hsi(hsi_1_f)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                msi_1_e_f = MN_msi(msi_1_f)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                hsi_2_e_f = MN_hsi(hsi_2_f)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                msi_2_e_f = MN_msi(msi_2_f)  # (B, 64, 512, 512) -> (B, 64, 512, 512)

                # ====================================================================
                # 阶段5: 字典表示模块（Dictionary Representation Module）
                # 用可学习的模态字典补偿单模态特征缺失的信息
                # ====================================================================
                HSIDP_hsi_1_f, _ = HSIDP(hsi_1_e_f)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                MSIDP_msi_1_f, _ = MSIDP(msi_1_e_f)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                HSIDP_hsi_2_f, _ = HSIDP(hsi_2_e_f)  # (B, 64, 512, 512) -> (B, 64, 512, 512)
                MSIDP_msi_2_f, _ = MSIDP(msi_2_e_f)  # (B, 64, 512, 512) -> (B, 64, 512, 512)

                del hsi_1_e_f, msi_1_e_f, hsi_2_e_f, msi_2_e_f
                torch.cuda.empty_cache()

                # ====================================================================
                # 阶段6: 跨模态对齐感知（Cross-Modality Alignment Perception）
                # 核心：构造未配准对（Pair1的HSI + Pair2的MSI）
                # ====================================================================
                # 🔥 关键设计：用Pair1的HSI作为参考（fixed），Pair2的MSI作为移动（moving）
                # 这样可以学习如何将未配准的MSI对齐到HSI
                fixed_DP = HSIDP_hsi_1_f  # 参考图像特征（来自Pair1）
                moving_DP = MSIDP_msi_2_f  # 移动图像特征（来自Pair2）

                # 窗口分割
                moving_DP_lw = model.df_window_partition(
                    moving_DP,
                    args.args.large_w_size,  # 52
                    args.args.small_w_size  # 32
                )  # -> (num_windows, B, 64, 52, 52)

                fixed_DP_sw = model.window_partition(
                    fixed_DP,
                    args.args.small_w_size,  # 32
                    args.args.small_w_size  # 32
                )  # -> (num_windows, B, 64, 32, 32)

                # 计算对齐感知矩阵
                correspondence_matrixs = model.CMAP(
                    fixed_DP_sw,  # 参考窗口
                    moving_DP_lw,  # 移动窗口
                    MHCSA_hsi,  # HSI的多头跨尺度注意力
                    MHCSA_msi,  # MSI的多头跨尺度注意力
                    True  # HSI作为参考
                )  # -> (num_windows, B, 32*32, 52*52)

                del fixed_DP_sw, moving_DP_lw
                torch.cuda.empty_cache()

                # ====================================================================
                # 阶段7: 特征重组和最终融合
                # 根据对齐矩阵重组MSI特征，使其与HSI对齐
                # ====================================================================
                msi_2_f_sample = model.feature_reorganization(
                    correspondence_matrixs,  # 对齐矩阵
                    msi_2_fe  # Pair2的MSI特征
                )  # -> (B, 64, 512, 512) - 对齐后的MSI特征

                # 最终融合：Pair1的HSI + 对齐后的Pair2的MSI
                fusion_image_sample = fusion_module(
                    hsi_1_fe,  # Pair1的HSI特征
                    msi_2_f_sample  # 对齐后的Pair2的MSI特征
                )  # -> (B, 31, 512, 512)

                # ====================================================================
                # 阶段8: 损失计算
                # ====================================================================

                # 8.1 基础融合损失（监督两对配准数据的融合质量）
                loss_fusion_1 = (
                        Lgrad(gt_1, gt_1, fusion_image_1) +
                        Loss.Loss_intensity(gt_1, gt_1, fusion_image_1) +
                        Lgrad(gt_1, gt_1, fusion_pimage_1) +
                        Loss.Loss_intensity(gt_1, gt_1, fusion_pimage_1)
                )

                loss_fusion_2 = (
                        Lgrad(gt_2, gt_2, fusion_image_2) +
                        Loss.Loss_intensity(gt_2, gt_2, fusion_image_2) +
                        Lgrad(gt_2, gt_2, fusion_pimage_2) +
                        Loss.Loss_intensity(gt_2, gt_2, fusion_pimage_2)
                )

                loss_0 = loss_fusion_1 + loss_fusion_2

                # 8.2 字典一致性损失（确保字典补偿后的特征与融合特征一致）
                loss_HSIDP = (
                        - CC(HSIDP_hsi_1_f, fusion_pf_1.detach())
                        - CC(HSIDP_hsi_2_f, fusion_pf_2.detach())
                )

                loss_MSIDP = (
                        - CC(MSIDP_msi_1_f, fusion_pf_1.detach())
                        - CC(MSIDP_msi_2_f, fusion_pf_2.detach())
                )

                # 8.3 模态一致性损失（确保HSI和MSI的字典补偿结果一致）
                loss_same = (
                        F.mse_loss(HSIDP_hsi_1_f, MSIDP_msi_1_f) +
                        F.mse_loss(HSIDP_hsi_2_f, MSIDP_msi_2_f)
                )

                loss_1 = 2 * (loss_HSIDP + loss_MSIDP + 0.5 * loss_same)

                # 8.4 对齐融合损失（监督最终的对齐融合结果）
                # 注意：用gt_1监督，因为用的是hsi_1 + aligned(msi_2)
                loss_2 = (
                        Lgrad(gt_1, gt_1, fusion_image_sample) +
                        Loss.Loss_intensity(gt_1, gt_1, fusion_image_sample)
                )

                # 8.5 对齐监督损失（暂时禁用，需要保存index_r才能启用）
                # 如果你在generate_deformed_gt.py中保存了变换矩阵，可以启用这部分
                # loss_correspondence_matrix, loss_correspondence_matrix_1 = Lcorrespondence(
                #     correspondence_matrixs, index_r
                # )
                # loss_3 = 4 * (loss_correspondence_matrix + loss_correspondence_matrix_1)

                # 总损失
                loss = loss_0 + loss_1 + loss_2  # + loss_3 (需要index_r时启用)

            # ========== 反向传播（混合精度）==========
            optimizer_HSIDP.zero_grad()
            optimizer_MSIDP.zero_grad()
            optimizer_MHCSAhsi.zero_grad()
            optimizer_MHCSAmsi.zero_grad()
            optimizer_FusionModule.zero_grad()
            optimizer_FE.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer_FE)
            scaler.step(optimizer_HSIDP)
            scaler.step(optimizer_MSIDP)
            scaler.step(optimizer_MHCSAhsi)
            scaler.step(optimizer_MHCSAmsi)
            scaler.step(optimizer_FusionModule)
            scaler.update()

            # ========== 显存清理 ==========
            del hsi_1_f, msi_1_f, hsi_2_f, msi_2_f
            del hsi_1_fe, msi_1_fe, hsi_2_fe, msi_2_fe
            del HSIDP_hsi_1_f, MSIDP_msi_1_f, HSIDP_hsi_2_f, MSIDP_msi_2_f
            del fusion_f_1, fusion_pf_1, fusion_f_2, fusion_pf_2
            del correspondence_matrixs, msi_2_f_sample
            del fixed_DP, moving_DP
            torch.cuda.empty_cache()

            # ========== 记录损失 ==========
            epoch_loss_HSIDP.append(loss_HSIDP.item())
            epoch_loss_MSIDP.append(loss_MSIDP.item())
            epoch_loss_same.append(loss_same.item())
            epoch_loss_fusion_total.append(loss.item())

            # ========== 保存图像（可视化训练进度）==========
            if step % save_image_iter == 0:
                epoch_step_name = str(epoch) + "epoch" + str(step) + "step"
                if epoch % 2 == 0:
                    output_name = save_img_dir + "/" + epoch_step_name + ".jpg"

                    # 上采样HSI用于可视化（取前3通道模拟RGB）
                    hsi_1_vis = F.interpolate(
                        hsi_1,
                        size=(msi_1.size(2), msi_1.size(3)),
                        mode='bilinear',
                        align_corners=False
                    )

                    # 拼接图像：HSI_1 | MSI_2 | Fusion_1 | Fusion_sample | Fusion_2
                    out = torch.cat([
                        hsi_1_vis[:, :3, :, :],  # Pair1的HSI（RGB通道）
                        msi_2[:, :3, :, :],  # Pair2的MSI
                        fusion_image_1[:, :3, :, :],  # Pair1的融合结果（RGB通道）
                        fusion_image_sample[:, :3, :, :],  # 对齐融合结果（RGB通道）
                        fusion_image_2[:, :3, :, :]  # Pair2的融合结果（RGB通道）
                    ], dim=3)

                    save_img(out, output_name)
                    del hsi_1_vis

            # ========== 保存模型 ==========
            if ((epoch + 1) == args.args.Epoch and (step + 1) % iter_num == 0) or \
                    (epoch % args.args.save_model_num == 0 and (step + 1) % iter_num == 0):
                ckpts = {
                    "bfe_msi": base_msi.state_dict(),
                    "bfe_hsi": base_hsi.state_dict(),
                    "msi_mfe": msi_MFE.state_dict(),
                    "hsi_mfe": hsi_MFE.state_dict(),
                    "pafe": PAFE.state_dict(),
                    "fusion_decoder": fusion_decoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "mn_msi": MN_msi.state_dict(),
                    "mn_hsi": MN_hsi.state_dict(),
                    "msi_dgfp": MSIDP.state_dict(),
                    "hsi_dgfp": HSIDP.state_dict(),
                    "mhcsab_msi": MHCSA_msi.state_dict(),
                    "mhcsab_hsi": MHCSA_hsi.state_dict(),
                    "fusion_block": fusion_module.state_dict(),
                }
                save_dir = '{:s}/epoch{:d}_iter{:d}.pth'.format(save_model_dir, epoch, step + 1)
                torch.save(ckpts, save_dir)
                print(f"💾 模型已保存: {save_dir}")

            # ========== 最终清理 ==========
            del hsi_1, msi_1, gt_1, hsi_2, msi_2, gt_2
            del fusion_image_1, fusion_pimage_1, fusion_image_2, fusion_pimage_2, fusion_image_sample
            torch.cuda.empty_cache()

        # ========== 打印epoch统计信息 ==========
        epoch_loss_HSIDP_mean = np.mean(epoch_loss_HSIDP)
        epoch_loss_MSIDP_mean = np.mean(epoch_loss_MSIDP)
        epoch_loss_same_mean = np.mean(epoch_loss_same)
        epoch_loss_fusion_mean = np.mean(epoch_loss_fusion_total)

        print()
        print(f"📊 Epoch {epoch} 统计:")
        print(f"   - 总损失(Total Loss):     {epoch_loss_fusion_mean:.6f}")
        print(f"   - HSI字典损失(HSIDP):      {epoch_loss_HSIDP_mean:.6f}")
        print(f"   - MSI字典损失(MSIDP):      {epoch_loss_MSIDP_mean:.6f}")
        print(f"   - 模态一致性损失(Same):    {epoch_loss_same_mean:.6f}")



    """
        输入: 6个张量
├── Pair 1: hsi_1, msi_1, gt_1  (来自 Z_reconst, Y_reconst, X)
└── Pair 2: hsi_2, msi_2, gt_2  (来自 Z_deformed, Y_deformed, X_deformed)

核心设计:
├── 用Pair1和Pair2分别训练基础融合能力
└── 构造未配准对: hsi_1 + msi_2 (跨Pair构造)
    └── 通过CMAP学习对齐关系
        └── 生成最终对齐融合结果
        """



    # ========== 启动训练循环 ==========
    print("\n🚀 开始训练...")
    print(f"📌 训练参数:")
    print(f"   - Epochs: {args.args.Epoch}")
    print(f"   - Batch Size: {args.args.batch_size}")
    print(f"   - 训练样本数: {len(dataset)}")
    print(f"   - 每epoch迭代数: {iter_num}")
    print(f"   - 图像尺寸: {args.args.img_size}×{args.args.img_size}")
    print(f"   - 混合精度: 已启用\n")

    for epoch in tqdm(range(args.args.Epoch), desc="训练进度"):
        if epoch < args.args.Warm_epoch:
            warmup_learning_rate(optimizer_MHCSAhsi, epoch)
            warmup_learning_rate(optimizer_MHCSAmsi, epoch)
        else:
            adjust_learning_rate(optimizer_MHCSAhsi, epoch)
            adjust_learning_rate(optimizer_MHCSAmsi, epoch)

        train(epoch)

    print("\n🎉 训练完成！")