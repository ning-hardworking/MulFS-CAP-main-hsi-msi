import os
import time
from pathlib import Path

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import scipy.io as sio
from torch.cuda.amp import autocast, GradScaler
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
                 transform=None, target_size=512):  # ✅ 默认512
        super(TrainDataset, self).__init__()

        self.target_size = target_size

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
        print(f"✅ 数据集加载成功 (目标尺寸: {target_size}×{target_size}):")
        print(f"   - Pair 1 (原始配准): {len(self.hsi_paths)} 对样本")
        print(f"   - Pair 2 (形变配准): {len(self.hsi_d_paths)} 对样本")

        # ✅ 新增：显示关键参数
        print(f"\n📐 尺寸参数:")
        print(f"   - MSI/GT目标尺寸: {target_size}×{target_size}")
        print(f"   - HSI目标尺寸: {target_size // 32}×{target_size // 32}")
        print(f"   - 下采样比例: 32")

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
        """读取.mat文件（已优化版本）"""
        try:
            mat_data = sio.loadmat(path)

            if key is None:
                valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if len(valid_keys) == 0:
                    raise ValueError(f"未找到有效数据键: {path}")
                key = valid_keys[0]

            img = mat_data[key]

            # 智能识别通道维度
            if img.ndim == 2:
                img = torch.from_numpy(img).float().unsqueeze(0)
            elif img.ndim == 3:
                shape = img.shape
                expected_channels = [3, 31]
                channel_dim_idx = None

                for i, s in enumerate(shape):
                    if s in expected_channels:
                        channel_dim_idx = i
                        break

                if channel_dim_idx is not None:
                    target_dim = channel_dim_idx
                else:
                    target_dim = np.argmin(shape)

                img = np.moveaxis(img, source=target_dim, destination=0)
                img = torch.from_numpy(img).float()
            else:
                raise ValueError(f"不支持的图像维度: {img.ndim}D，路径: {path}")

            # 归一化
            if img.max() > 1.0:
                img = img / img.max()

            if self.transform is not None:
                img = self.transform(img)

            return img

        except Exception as e:
            print(f"❌ 读取文件失败: {path}")
            print(f"   错误信息: {str(e)}")
            raise

    def __getitem__(self, index):
        # Pair 1: 原始配准对
        hsi_1 = self.read_mat_image(self.hsi_paths[index])
        msi_1 = self.read_mat_image(self.msi_paths[index])
        gt_1 = self.read_mat_image(self.gt_paths[index])

        # Pair 2: 形变配准对
        hsi_2 = self.read_mat_image(self.hsi_d_paths[index])
        msi_2 = self.read_mat_image(self.msi_d_paths[index])
        gt_2 = self.read_mat_image(self.gt_d_paths[index])

        # ========== 🔥 Resize到目标尺寸 🔥 ==========
        # ✅ 修正：保持固定的下采样比例 32
        original_ratio = 32  # 原始数据: 512/16 = 32
        hsi_target_size = self.target_size // original_ratio  # 512 // 32 = 16

        # ✅ 验证尺寸合理性
        if hsi_target_size < 8:
            raise ValueError(
                f"❌ HSI目标尺寸({hsi_target_size}×{hsi_target_size})太小！"
                f"建议 img_size >= 256"
            )

        # Resize MSI和GT（保持原始512×512，或resize到target_size）
        if msi_1.size(-1) != self.target_size:
            msi_1 = F.interpolate(msi_1.unsqueeze(0), size=(self.target_size, self.target_size),
                                  mode='bilinear', align_corners=False).squeeze(0)
            gt_1 = F.interpolate(gt_1.unsqueeze(0), size=(self.target_size, self.target_size),
                                 mode='bilinear', align_corners=False).squeeze(0)

            msi_2 = F.interpolate(msi_2.unsqueeze(0), size=(self.target_size, self.target_size),
                                  mode='bilinear', align_corners=False).squeeze(0)
            gt_2 = F.interpolate(gt_2.unsqueeze(0), size=(self.target_size, self.target_size),
                                 mode='bilinear', align_corners=False).squeeze(0)

        # Resize HSI（保持原始16×16，或resize到hsi_target_size）
        if hsi_1.size(-1) != hsi_target_size:
            hsi_1 = F.interpolate(hsi_1.unsqueeze(0), size=(hsi_target_size, hsi_target_size),
                                  mode='bilinear', align_corners=False).squeeze(0)
            hsi_2 = F.interpolate(hsi_2.unsqueeze(0), size=(hsi_target_size, hsi_target_size),
                                  mode='bilinear', align_corners=False).squeeze(0)

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

    # ✅ 修改数据集初始化
    tf = None

    dataset = TrainDataset(
        args.args.hsi_train_dir,
        args.args.msi_train_dir,
        args.args.gt_train_dir,
        args.args.hsi_deformed_train_dir,
        args.args.msi_deformed_train_dir,
        args.args.gt_deformed_train_dir,
        tf,
        target_size=args.args.img_size  # ✅ 传入目标尺寸
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

    # ========== 损失函数初始化 ==========
    print("🔧 正在初始化损失函数...")
    Lgrad = Loss.L_Grad().to(device)
    CC = Loss.CorrelationCoefficient().to(device)
    Lcorrespondence = Loss.L_correspondence()

    # ✅ 新增：31通道专用损失函数
    SpectralLoss = Loss.SpectralConsistencyLoss().to(device)
    SAMLoss = Loss.SpectralAngleLoss().to(device)
    print("✅ 已加载31通道专用损失函数")

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
        # ========== ✅ 完整的维度和参数验证 ==========
        if epoch == 0:
            print(f"\n{'=' * 70}")
            print(f"🔍 Epoch 0 - 完整参数验证:")
            print(f"{'=' * 70}")

        epoch_loss_HSIDP = []
        epoch_loss_MSIDP = []
        epoch_loss_same = []
        epoch_loss_fusion_total = []

        for step, x in enumerate(data_iter):
            # ========== 数据加载（6个张量）==========
            hsi_1, msi_1, gt_1, hsi_2, msi_2, gt_2 = [
                item.to(device, non_blocking=True) for item in x
            ]

            # ✅ 第一个batch进行完整验证
            if step == 0 and epoch == 0:
                print(f"\n📊 1. 数据维度验证:")
                print(f"   Pair 1:")
                print(f"     - HSI:  {hsi_1.shape}  (期望: [1, 31, 16, 16])")
                print(f"     - MSI:  {msi_1.shape}  (期望: [1, 3, 512, 512])")
                print(f"     - GT:   {gt_1.shape}   (期望: [1, 31, 512, 512])")
                print(f"   Pair 2:")
                print(f"     - HSI:  {hsi_2.shape}  (期望: [1, 31, 16, 16])")
                print(f"     - MSI:  {msi_2.shape}  (期望: [1, 3, 512, 512])")
                print(f"     - GT:   {gt_2.shape}   (期望: [1, 31, 512, 512])")

                # ✅ 验证维度匹配
                assert hsi_1.size(1) == 31, f"❌ HSI通道数错误！期望31，实际{hsi_1.size(1)}"
                assert msi_1.size(1) == 3, f"❌ MSI通道数错误！期望3，实际{msi_1.size(1)}"
                assert gt_1.size(1) == 31, f"❌ GT通道数错误！期望31，实际{gt_1.size(1)}"

                assert hsi_1.size(2) == 16 and hsi_1.size(3) == 16, \
                    f"❌ HSI尺寸错误！期望16×16，实际{hsi_1.size(2)}×{hsi_1.size(3)}"
                assert msi_1.size(2) == 512 and msi_1.size(3) == 512, \
                    f"❌ MSI尺寸错误！期望512×512，实际{msi_1.size(2)}×{msi_1.size(3)}"
                assert gt_1.size(2) == 512 and gt_1.size(3) == 512, \
                    f"❌ GT尺寸错误！期望512×512，实际{gt_1.size(2)}×{gt_1.size(3)}"

                print(f"   ✅ 所有维度验证通过！")

                # ✅ 验证数值范围
                print(f"\n📊 2. 数值范围验证:")
                print(f"   - HSI:  min={hsi_1.min():.4f}, max={hsi_1.max():.4f}")
                print(f"   - MSI:  min={msi_1.min():.4f}, max={msi_1.max():.4f}")
                print(f"   - GT:   min={gt_1.min():.4f}, max={gt_1.max():.4f}")

                assert 0 <= hsi_1.min() and hsi_1.max() <= 1, "❌ HSI数值范围应在[0,1]"
                assert 0 <= msi_1.min() and msi_1.max() <= 1, "❌ MSI数值范围应在[0,1]"
                assert 0 <= gt_1.min() and gt_1.max() <= 1, "❌ GT数值范围应在[0,1]"
                print(f"   ✅ 所有数值范围正常！")

                # ✅ 验证窗口参数
                print(f"\n📊 3. 窗口参数验证:")
                hsi_h, hsi_w = hsi_1.size(2), hsi_1.size(3)
                small_w = int(args.args.small_w_size)
                large_w = int(args.args.large_w_size)

                print(f"   - HSI尺寸: {hsi_h}×{hsi_w}")
                print(f"   - small_w_size: {small_w}")
                print(f"   - large_w_size: {large_w}")
                print(f"   - 窗口数量: {hsi_h // small_w}×{hsi_w // small_w} = {(hsi_h // small_w) ** 2}个")

                assert small_w <= hsi_h, \
                    f"❌ small_w_size({small_w}) > HSI高度({hsi_h})！"
                assert large_w > small_w, \
                    f"❌ large_w_size({large_w}) 应该 > small_w_size({small_w})！"
                assert hsi_h % small_w == 0, \
                    f"❌ HSI尺寸({hsi_h})应该被small_w_size({small_w})整除！"

                print(f"   ✅ 窗口参数验证通过！")

                print(f"\n{'=' * 70}\n")
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
        # ✅ 显存监控
        if epoch == 0:
            print(f"\n💾 训练前显存状态:")
            print(f"   已分配: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            print(f"   已缓存: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB\n")

        epoch_loss_HSIDP = []
        epoch_loss_MSIDP = []
        epoch_loss_same = []
        epoch_loss_fusion_total = []

        for step, x in enumerate(data_iter):
            # ========== 数据加载（6个张量）==========
            hsi_1, msi_1, gt_1, hsi_2, msi_2, gt_2 = [
                item.to(device, non_blocking=True) for item in x
            ]

            # ✅ 打印维度（仅第一个batch）
            if step == 0 and epoch == 0:
                print(f"\n✅ 数据维度验证:")
                print(f"   Pair 1:")
                print(f"     - HSI:  {hsi_1.shape}  (期望: [B, 31, 4, 4])")
                print(f"     - MSI:  {msi_1.shape}  (期望: [B, 3, 128, 128])")
                print(f"     - GT:   {gt_1.shape}   (期望: [B, 31, 128, 128])")
                print(f"   Pair 2:")
                print(f"     - HSI:  {hsi_2.shape}  (期望: [B, 31, 4, 4])")
                print(f"     - MSI:  {msi_2.shape}  (期望: [B, 3, 128, 128])")
                print(f"     - GT:   {gt_2.shape}   (期望: [B, 31, 128, 128])")

                # ✅ 验证GT确实是31通道
                assert gt_1.size(1) == 31, f"❌ GT通道数错误！期望31，实际{gt_1.size(1)}"
                assert gt_2.size(1) == 31, f"❌ GT通道数错误！期望31，实际{gt_2.size(1)}"
                print(f"✅ 所有维度匹配！\n")

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
                hsi_1_base = base_hsi(hsi_1_up)  # (B, 31, 128, 128) -> (B, 64, 128, 128)
                msi_1_base = base_msi(msi_1)  # (B, 3, 128, 128)  -> (B, 64, 128, 128)
                hsi_2_base = base_hsi(hsi_2_up)  # (B, 31, 128, 128) -> (B, 64, 128, 128)
                msi_2_base = base_msi(msi_2)  # (B, 3, 128, 128)  -> (B, 64, 128, 128)

                # 释放不需要的上采样结果
                del hsi_1_up, hsi_2_up
                torch.cuda.empty_cache()

                # ====================================================================
                # 阶段2: 深层特征提取（用于融合重建）
                # ====================================================================
                # Pair 1 的深层特征
                hsi_1_fe = hsi_MFE(hsi_1_base)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                msi_1_fe = msi_MFE(msi_1_base)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                simple_fusion_f_1 = hsi_1_fe + msi_1_fe
                fusion_image_1, fusion_f_1 = fusion_decoder(simple_fusion_f_1)  # -> (B, 31, 128, 128)

                # Pair 2 的深层特征
                hsi_2_fe = hsi_MFE(hsi_2_base)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                msi_2_fe = msi_MFE(msi_2_base)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                simple_fusion_f_2 = hsi_2_fe + msi_2_fe
                fusion_image_2, fusion_f_2 = fusion_decoder(simple_fusion_f_2)  # -> (B, 31, 128, 128)

                del simple_fusion_f_1, simple_fusion_f_2
                torch.cuda.empty_cache()

                # ====================================================================
                # 阶段3: PAFE特征提取（用于对齐感知）
                # ====================================================================
                # Pair 1 的PAFE特征
                hsi_1_f = PAFE(hsi_1_base)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                msi_1_f = PAFE(msi_1_base)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                simple_fusion_pf_1 = hsi_1_f + msi_1_f
                fusion_pimage_1, fusion_pf_1 = decoder(simple_fusion_pf_1)  # -> (B, 31, 128, 128)

                # Pair 2 的PAFE特征
                hsi_2_f = PAFE(hsi_2_base)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                msi_2_f = PAFE(msi_2_base)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                simple_fusion_pf_2 = hsi_2_f + msi_2_f
                fusion_pimage_2, fusion_pf_2 = decoder(simple_fusion_pf_2)  # -> (B, 31, 128, 128)

                del simple_fusion_pf_1, simple_fusion_pf_2
                del hsi_1_base, hsi_2_base, msi_1_base, msi_2_base
                torch.cuda.empty_cache()

                # ====================================================================
                # 阶段4: 模态归一化（Modality Normalization）
                # ====================================================================
                hsi_1_e_f = MN_hsi(hsi_1_f)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                msi_1_e_f = MN_msi(msi_1_f)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                hsi_2_e_f = MN_hsi(hsi_2_f)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                msi_2_e_f = MN_msi(msi_2_f)  # (B, 64, 128, 128) -> (B, 64, 128, 128)

                # ====================================================================
                # 阶段5: 字典表示模块（Dictionary Representation Module）
                # 用可学习的模态字典补偿单模态特征缺失的信息
                # ====================================================================
                HSIDP_hsi_1_f, _ = HSIDP(hsi_1_e_f)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                MSIDP_msi_1_f, _ = MSIDP(msi_1_e_f)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                HSIDP_hsi_2_f, _ = HSIDP(hsi_2_e_f)  # (B, 64, 128, 128) -> (B, 64, 128, 128)
                MSIDP_msi_2_f, _ = MSIDP(msi_2_e_f)  # (B, 64, 128, 128) -> (B, 64, 128, 128)

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
                    args.args.large_w_size,  # 12
                    args.args.small_w_size  # 8
                )  # -> (num_windows, B, 64, 12, 12)

                fixed_DP_sw = model.window_partition(
                    fixed_DP,
                    args.args.small_w_size,  # 8
                    args.args.small_w_size  # 8
                )  # -> (num_windows, B, 64, 8, 8)

                # 计算对齐感知矩阵
                correspondence_matrixs = model.CMAP(
                    fixed_DP_sw,  # 参考窗口
                    moving_DP_lw,  # 移动窗口
                    MHCSA_hsi,  # HSI的多头跨尺度注意力
                    MHCSA_msi,  # MSI的多头跨尺度注意力
                    True  # HSI作为参考
                )  # -> (num_windows, B, 8*8, 12*12)

                del fixed_DP_sw, moving_DP_lw
                torch.cuda.empty_cache()

                # ====================================================================
                # 阶段7: 特征重组和最终融合
                # 根据对齐矩阵重组MSI特征，使其与HSI对齐
                # ====================================================================
                msi_2_f_sample = model.feature_reorganization(
                    correspondence_matrixs,  # 对齐矩阵
                    msi_2_fe  # Pair2的MSI特征
                )  # -> (B, 64, 128, 128) - 对齐后的MSI特征

                # 最终融合：Pair1的HSI + 对齐后的Pair2的MSI
                fusion_image_sample = fusion_module(
                    hsi_1_fe,  # Pair1的HSI特征
                    msi_2_f_sample  # 对齐后的Pair2的MSI特征
                )  # -> (B, 31, 128, 128)

                # ====================================================================
                # 阶段8: 损失计算
                # ====================================================================

                # 8.1 基础融合损失（监督两对配准数据的融合质量）
                # ✅ 修正：传入正确的 (hsi, msi, fusion) 三元组
                loss_fusion_1 = (
                        Lgrad(hsi_1, msi_1, fusion_image_1) +  # 31通道梯度损失
                        Loss.Loss_intensity(hsi_1, msi_1, fusion_image_1) +  # 31通道强度损失
                        Lgrad(hsi_1, msi_1, fusion_pimage_1) +
                        Loss.Loss_intensity(hsi_1, msi_1, fusion_pimage_1) +
                        0.5 * SpectralLoss(fusion_image_1, gt_1)  # ✅ 新增：光谱一致性损失
                )

                loss_fusion_2 = (
                        Lgrad(hsi_2, msi_2, fusion_image_2) +
                        Loss.Loss_intensity(hsi_2, msi_2, fusion_image_2) +
                        Lgrad(hsi_2, msi_2, fusion_pimage_2) +
                        Loss.Loss_intensity(hsi_2, msi_2, fusion_pimage_2) +
                        0.5 * SpectralLoss(fusion_image_2, gt_2)  # ✅ 新增
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

                # 8.4 对齐融合损失
                # ✅ 修正：用 hsi_1 和 msi_2 监督（因为是 hsi_1 + aligned(msi_2) 的融合）
                loss_2 = (
                        Lgrad(hsi_1, msi_2, fusion_image_sample) +
                        Loss.Loss_intensity(hsi_1, msi_2, fusion_image_sample)
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



            # ========== 记录损失 ==========
            epoch_loss_HSIDP.append(loss_HSIDP.item())
            epoch_loss_MSIDP.append(loss_MSIDP.item())
            epoch_loss_same.append(loss_same.item())
            epoch_loss_fusion_total.append(loss.item())

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



            # ✅ 每10步打印显存（可选）
            if step % 10 == 0:
                print(f"Step {step}/{iter_num} - 显存: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

        # ========== 打印epoch统计信息 ==========
        epoch_loss_HSIDP_mean = np.mean(epoch_loss_HSIDP)
        epoch_loss_MSIDP_mean = np.mean(epoch_loss_MSIDP)
        epoch_loss_same_mean = np.mean(epoch_loss_same)
        epoch_loss_fusion_mean = np.mean(epoch_loss_fusion_total)

        # ✅ 新增：计算评估指标（使用最后一个batch的结果）
        with torch.no_grad():
            # 计算PSNR
            mse = F.mse_loss(fusion_image_sample, gt_1)
            psnr = 10 * torch.log10(1.0 / mse)

            # 计算SAM（光谱角距离）
            # 将空间维度展平: (B, 31, H, W) -> (B, 31, H*W)
            pred_flat = fusion_image_sample.view(fusion_image_sample.size(0), fusion_image_sample.size(1), -1)
            target_flat = gt_1.view(gt_1.size(0), gt_1.size(1), -1)

            # 计算内积和模长
            dot_product = torch.sum(pred_flat * target_flat, dim=1)  # (B, H*W)
            pred_norm = torch.norm(pred_flat, dim=1) + 1e-8
            target_norm = torch.norm(target_flat, dim=1) + 1e-8

            # 计算cos值并转换为角度
            cos_theta = dot_product / (pred_norm * target_norm)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            sam = torch.acos(cos_theta).mean() * 180 / np.pi  # 转换为度数

        print()
        print(f"📊 Epoch {epoch} 统计:")
        print(f"   - 总损失(Total Loss):     {epoch_loss_fusion_mean:.6f}")
        print(f"   - HSI字典损失(HSIDP):      {epoch_loss_HSIDP_mean:.6f}")
        print(f"   - MSI字典损失(MSIDP):      {epoch_loss_MSIDP_mean:.6f}")
        print(f"   - 模态一致性损失(Same):    {epoch_loss_same_mean:.6f}")
        with torch.no_grad():
            spectral_loss_val = SpectralLoss(fusion_image_sample, gt_1)
            sam_loss_val = SAMLoss(fusion_image_sample, gt_1)
            print(f"   - 光谱一致性损失:          {spectral_loss_val.item():.6f}")
            print(f"   - 光谱角损失(SAM):         {sam_loss_val.item():.6f}")

        print(f"   📈 评估指标:")
        print(f"      - PSNR: {psnr.item():.4f} dB")
        print(f"      - SAM:  {sam.item():.4f}°")



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