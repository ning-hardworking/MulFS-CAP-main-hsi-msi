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
    def __init__(self, hsi_dir, msi_dir, gt_dir, transform=None):
        super(TrainDataset, self).__init__()
        self.hsi_dir = hsi_dir
        self.msi_dir = msi_dir
        self.gt_dir = gt_dir
        self.hsi_paths = self.find_mat_files(self.hsi_dir)
        self.msi_paths = self.find_mat_files(self.msi_dir)
        self.gt_paths = self.find_mat_files(self.gt_dir)

        assert len(self.hsi_paths) == len(self.msi_paths) == len(self.gt_paths), \
            f"HSI、MSI和GT文件数量不相等: {len(self.hsi_paths)}, {len(self.msi_paths)}, {len(self.gt_paths)}"

        self.transform = transform
        print(f"✅ 数据集加载成功: {len(self.hsi_paths)} 对样本")

    def find_mat_files(self, dir_path):
        """查找所有.mat文件"""
        mat_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
        mat_files.sort()  # 确保顺序一致
        return mat_files

    def read_mat_image(self, path, key=None):
        """
        读取.mat文件中的图像数据

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
                key = valid_keys[0]  # 使用第一个有效键

            img = mat_data[key]  # shape: (H, W, C)

            # 转换为torch张量并调整维度顺序: (H, W, C) -> (C, H, W)
            img = torch.from_numpy(img).float()
            if img.ndim == 2:  # 如果是2D，添加通道维度
                img = img.unsqueeze(0)
            elif img.ndim == 3:
                img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

            # 归一化到[0, 1]（如果数据范围不是[0,1]）
            if img.max() > 1.0:
                img = img / img.max()

            # 应用transform（如果需要resize）
            if self.transform is not None:
                img = self.transform(img)

            return img

        except Exception as e:
            print(f"❌ 读取文件失败: {path}")
            print(f"   错误信息: {str(e)}")
            raise

    def __getitem__(self, index):
        hsi_path = self.hsi_paths[index]
        msi_path = self.msi_paths[index]
        gt_path = self.gt_paths[index]

        # 读取数据
        # HSI: (31, 16, 16) - 31通道，低分辨率
        # MSI: (3, 512, 512) - 3通道，高分辨率
        # GT:  (31, 512, 512) - 31通道，高分辨率
        hsi_img = self.read_mat_image(hsi_path)  # (31, 16, 16)
        msi_img = self.read_mat_image(msi_path)  # (3, 512, 512)
        gt_img = self.read_mat_image(gt_path)  # (31, 512, 512)

        return hsi_img, msi_img, gt_img

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

    dataset = TrainDataset(args.args.hsi_train_dir, args.args.msi_train_dir, args.args.gt_train_dir, tf)

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
        epoch_loss_HSIDP = []
        epoch_loss_MSIDP = []
        epoch_loss_same = []
        epoch_loss_correspondence_matrix = []
        epoch_loss_correspondence_predict = []

        for step, x in enumerate(data_iter):
            # 数据加载
            hsi = x[0].to(device, non_blocking=True)
            msi = x[1].to(device, non_blocking=True)
            gt = x[2].to(device, non_blocking=True)

            # ========== ✅ 关键修复：ImageDeformation在autocast外执行 ==========
            # 先上采样HSI（在autocast外，使用FP32）
            hsi_upsampled = F.interpolate(
                hsi,
                size=(msi.size(2), msi.size(3)),
                mode='bilinear',
                align_corners=False
            )

            # ✅ ImageDeformation不使用混合精度（避免torch.inverse错误）
            with torch.no_grad():
                hsi_d, msi_d, _, index_r, _ = ImageDeformation(hsi_upsampled, msi)

            # ========== 其他部分使用混合精度 ==========
            with autocast():
                # ========== 阶段2: 基础特征提取 ==========
                hsi_1 = base_hsi(hsi_upsampled)
                hsi_d_1 = base_hsi(hsi_d)
                msi_1 = base_msi(msi)
                msi_d_1 = base_msi(msi_d)

                # ✅ 释放不再需要的张量
                del hsi_upsampled
                torch.cuda.empty_cache()

                # ========== 阶段3: 深层特征提取（第一路径） ==========
                hsi_fe = hsi_MFE(hsi_1)
                msi_fe = msi_MFE(msi_1)
                simple_fusion_f_1 = hsi_fe + msi_fe
                fusion_image_1, fusion_f_1 = fusion_decoder(simple_fusion_f_1)

                del simple_fusion_f_1
                torch.cuda.empty_cache()

                hsi_d_fe = hsi_MFE(hsi_d_1)
                msi_d_fe = msi_MFE(msi_d_1)
                simple_fusion_d_f_1 = hsi_d_fe + msi_d_fe
                fusion_d_image_1, fusion_d_f_1 = fusion_decoder(simple_fusion_d_f_1)

                del simple_fusion_d_f_1
                torch.cuda.empty_cache()

                # ========== 阶段4: PAFE特征提取 ==========
                hsi_f = PAFE(hsi_1)
                msi_f = PAFE(msi_1)
                simple_fusion_f = hsi_f + msi_f
                fusion_image, fusion_f = decoder(simple_fusion_f)

                del simple_fusion_f, hsi_1
                torch.cuda.empty_cache()

                hsi_d_f = PAFE(hsi_d_1)
                msi_d_f = PAFE(msi_d_1)
                simple_fusion_d_f = hsi_d_f + msi_d_f
                fusion_d_image, fusion_d_f = decoder(simple_fusion_d_f)

                del simple_fusion_d_f, hsi_d_1, msi_1, msi_d_1
                torch.cuda.empty_cache()

                # ========== 阶段5: 模态归一化和字典补偿 ==========
                hsi_e_f = MN_hsi(hsi_f)
                msi_e_f = MN_msi(msi_f)
                hsi_d_e_f = MN_hsi(hsi_d_f)
                msi_d_e_f = MN_msi(msi_d_f)

                HSIDP_hsi_f, _ = HSIDP(hsi_e_f)
                MSIDP_msi_f, _ = MSIDP(msi_e_f)
                HSIDP_hsi_d_f, _ = HSIDP(hsi_d_e_f)
                MSIDP_msi_d_f, _ = MSIDP(msi_d_e_f)

                del hsi_e_f, msi_e_f, hsi_d_e_f, msi_d_e_f
                torch.cuda.empty_cache()

                # ========== 阶段6: 跨模态对齐感知 ==========
                fixed_DP = HSIDP_hsi_f
                moving_DP = MSIDP_msi_d_f

                moving_DP_lw = model.df_window_partition(moving_DP, args.args.large_w_size, args.args.small_w_size)
                fixed_DP_sw = model.window_partition(fixed_DP, args.args.small_w_size, args.args.small_w_size)

                correspondence_matrixs = model.CMAP(fixed_DP_sw, moving_DP_lw, MHCSA_hsi, MHCSA_msi, True)

                del fixed_DP_sw, moving_DP_lw
                torch.cuda.empty_cache()

                # ========== 阶段7: 特征重组和最终融合 ==========
                msi_d_f_sample = model.feature_reorganization(correspondence_matrixs, msi_d_fe)
                fusion_image_sample = fusion_module(hsi_fe, msi_d_f_sample)

                # ========== 阶段8: 损失计算 ==========
                loss_fusion = Lgrad(gt, gt, fusion_image) + Loss.Loss_intensity(gt, gt, fusion_image) + \
                              Lgrad(gt, gt, fusion_d_image) + Loss.Loss_intensity(gt, gt, fusion_d_image)

                loss_fusion_1 = Lgrad(gt, gt, fusion_image_1) + Loss.Loss_intensity(gt, gt, fusion_image_1) + \
                                Lgrad(gt, gt, fusion_d_image_1) + Loss.Loss_intensity(gt, gt, fusion_d_image_1)

                loss_0 = loss_fusion

                loss_HSIDP = - CC(HSIDP_hsi_f, fusion_f.detach()) - CC(HSIDP_hsi_d_f, fusion_d_f.detach())
                loss_MSIDP = - CC(MSIDP_msi_f, fusion_f.detach()) - CC(MSIDP_msi_d_f, fusion_d_f.detach())
                loss_same = F.mse_loss(HSIDP_hsi_f, MSIDP_msi_f) + F.mse_loss(HSIDP_hsi_d_f, MSIDP_msi_d_f)

                loss_1 = 2 * (loss_HSIDP + loss_MSIDP + loss_same)
                loss_2 = Lgrad(gt, gt, fusion_image_sample) + Loss.Loss_intensity(gt, gt, fusion_image_sample)

                loss_correspondence_matrix, loss_correspondence_matrix_1 = Lcorrespondence(
                    correspondence_matrixs, index_r)
                loss_3 = 4 * (loss_correspondence_matrix + loss_correspondence_matrix_1)

                loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_fusion_1

            # ========== 反向传播（混合精度） ==========
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

            # ✅ 显存清理
            del hsi_f, msi_f, hsi_d_f, msi_d_f
            del hsi_fe, msi_fe, hsi_d_fe, msi_d_fe
            del HSIDP_hsi_f, MSIDP_msi_f, HSIDP_hsi_d_f, MSIDP_msi_d_f
            del fusion_f, fusion_d_f, fusion_f_1, fusion_d_f_1
            del correspondence_matrixs, msi_d_f_sample
            del fixed_DP, moving_DP
            torch.cuda.empty_cache()

            # 记录损失
            epoch_loss_HSIDP.append(loss_HSIDP.item())
            epoch_loss_MSIDP.append(loss_MSIDP.item())
            epoch_loss_same.append(loss_same.item())
            epoch_loss_correspondence_matrix.append(loss_correspondence_matrix.item())
            epoch_loss_correspondence_predict.append(loss_correspondence_matrix_1.item())

            # 保存图像
            if step % save_image_iter == 0:
                epoch_step_name = str(epoch) + "epoch" + str(step) + "step"
                if epoch % 2 == 0:
                    output_name = save_img_dir + "/" + epoch_step_name + ".jpg"
                    # 上采样HSI用于可视化
                    hsi_vis = F.interpolate(hsi, size=(msi.size(2), msi.size(3)), mode='bilinear', align_corners=False)
                    out = torch.cat([
                        hsi_vis[:, :3, :, :],
                        msi_d[:, :3, :, :],
                        fusion_image_1[:, :3, :, :],
                        fusion_image_sample[:, :3, :, :],
                        fusion_d_image_1[:, :3, :, :]
                    ], dim=3)
                    save_img(out, output_name)
                    del hsi_vis

            # 保存模型
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

            # ✅ 最终清理
            del hsi, msi, gt, hsi_d, msi_d
            del fusion_image, fusion_d_image, fusion_image_1, fusion_d_image_1, fusion_image_sample
            torch.cuda.empty_cache()

        # 打印epoch损失
        epoch_loss_correspondence_matrix_mean = np.mean(epoch_loss_correspondence_matrix)
        epoch_loss_correspondence_predict_mean = np.mean(epoch_loss_correspondence_predict)
        epoch_loss_HSIDP_mean = np.mean(epoch_loss_HSIDP)
        epoch_loss_MSIDP_mean = np.mean(epoch_loss_MSIDP)
        epoch_loss_same_mean = np.mean(epoch_loss_same)

        print()
        print(f"📊 -epoch {epoch}")
        print(
            f"   -loss_cm {epoch_loss_correspondence_matrix_mean:.6f} -loss_cp {epoch_loss_correspondence_predict_mean:.6f}")
        print(f"   -loss_HSIDP {epoch_loss_HSIDP_mean:.6f} -loss_MSIDP {epoch_loss_MSIDP_mean:.6f}")
        print(f"   -loss_same {epoch_loss_same_mean:.6f}")


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