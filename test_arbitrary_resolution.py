"""
HSI-MSI融合测试脚本（任意尺寸版）
- 保持原始图像分辨率，只调整为16的倍数
- 计算PSNR、SAM等评估指标
- 保存融合结果为.mat文件
"""

import os
from pathlib import Path
import gc

import torch
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import scipy.io as sio

import model.model as model
import utils.utils as utils
import args

# ========== 设备配置 ==========
device_id = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = device_id
device = torch.device("cuda:" + device_id if torch.cuda.is_available() else "cpu")

print(f"🚀 使用设备: {device}")


# ========== 评估指标函数（与固定尺寸版完全相同）==========
def calculate_psnr(pred, target, data_range=1.0):
    """计算PSNR"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(data_range ** 2 / mse)
    return psnr.item()


def calculate_sam(pred, target):
    """计算SAM（光谱角距离）"""
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)

    dot_product = torch.sum(pred_flat * target_flat, dim=1)
    pred_norm = torch.norm(pred_flat, dim=1) + 1e-8
    target_norm = torch.norm(target_flat, dim=1) + 1e-8

    cos_theta = dot_product / (pred_norm * target_norm)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    sam = torch.acos(cos_theta).mean() * 180 / np.pi

    return sam.item()


def calculate_ssim(pred, target, data_range=1.0):
    """计算SSIM"""
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
    """计算RMSE"""
    mse = F.mse_loss(pred, target)
    rmse = torch.sqrt(mse)
    return rmse.item()


# ========== 数据集类（任意尺寸版）==========
class HSI_MSI_TestDataset(data.Dataset):
    def __init__(self, hsi_dir, msi_dir, gt_dir):
        """
        Args:
            hsi_dir: 低分辨率HSI目录
            msi_dir: 高分辨率MSI目录
            gt_dir: 高分辨率GT目录
        """
        super(HSI_MSI_TestDataset, self).__init__()

        self.hsi_paths = self.find_mat_files(hsi_dir)
        self.msi_paths = self.find_mat_files(msi_dir)
        self.gt_paths = self.find_mat_files(gt_dir)

        assert len(self.hsi_paths) == len(self.msi_paths) == len(self.gt_paths), \
            f"数据数量不一致: HSI={len(self.hsi_paths)}, MSI={len(self.msi_paths)}, GT={len(self.gt_paths)}"

        print(f"✅ 加载了 {len(self.hsi_paths)} 对测试样本 (保持原始分辨率)")

    def find_mat_files(self, dir_path):
        """查找所有.mat文件"""
        mat_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
        mat_files.sort()
        return mat_files

    def read_mat_image(self, path):
        """读取.mat文件"""
        try:
            mat_data = sio.loadmat(path)
            valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if len(valid_keys) == 0:
                raise ValueError(f"未找到有效数据键: {path}")
            key = valid_keys[0]
            img = mat_data[key]

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

            if img.max() > 1.0:
                img = img / img.max()

            return img

        except Exception as e:
            print(f"❌ 读取文件失败: {path}")
            print(f"   错误信息: {str(e)}")
            raise

    def __getitem__(self, index):
        hsi = self.read_mat_image(self.hsi_paths[index])  # (31, H_low, W_low)
        msi = self.read_mat_image(self.msi_paths[index])  # (3, H_high, W_high)
        gt = self.read_mat_image(self.gt_paths[index])  # (31, H_high, W_high)

        # ✅ 记录原始尺寸
        _, h_orig, w_orig = msi.shape

        # ✅ 调整为16的倍数（与IR-VIS逻辑一致）
        h_new = (h_orig // 16) * 16
        w_new = (w_orig // 16) * 16

        if h_orig % 16 != 0 or w_orig % 16 != 0:
            # 调整MSI和GT
            msi = F.interpolate(
                msi.unsqueeze(0), (h_new, w_new),
                mode='bilinear', align_corners=False
            ).squeeze(0)

            gt = F.interpolate(
                gt.unsqueeze(0), (h_new, w_new),
                mode='bilinear', align_corners=False
            ).squeeze(0)

            # 调整HSI（保持32倍下采样比例）
            hsi_h_new = h_new // 32
            hsi_w_new = w_new // 32
            hsi = F.interpolate(
                hsi.unsqueeze(0), (hsi_h_new, hsi_w_new),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        file_name = os.path.basename(self.hsi_paths[index])

        return hsi, msi, gt, (h_orig, w_orig), file_name

    def __len__(self):
        return len(self.hsi_paths)


# ========== 路径配置 ==========
hsi_test_dir = r"D:/datas/CAVEdata/Z_reconst"
msi_test_dir = r"D:/datas/CAVEdata/Y_reconst"
gt_test_dir = r"D:/datas/CAVEdata/X"

save_dir = "./test_results_arbitrary"
save_fusion_dir = os.path.join(save_dir, "fusion")
save_metrics_dir = os.path.join(save_dir, "metrics")

utils.check_dir(save_dir)
utils.check_dir(save_fusion_dir)
utils.check_dir(save_metrics_dir)

# ========== 数据加载器 ==========
test_dataset = HSI_MSI_TestDataset(hsi_test_dir, msi_test_dir, gt_test_dir)

test_data_iter = data.DataLoader(
    dataset=test_dataset,
    shuffle=False,
    batch_size=1,
    num_workers=0
)

# ========== 加载模型（与固定尺寸版完全相同）==========
print("\n🔧 正在初始化模型...")
with torch.no_grad():
    base_msi = model.base(in_channels=3)
    base_hsi = model.base(in_channels=31)
    hsi_MFE = model.FeatureExtractor()
    msi_MFE = model.FeatureExtractor()
    fusion_decoder = model.Decoder()
    PAFE = model.FeatureExtractor()
    decoder = model.Decoder()
    MN_hsi = model.Enhance()
    MN_msi = model.Enhance()
    HSIDP = model.DictionaryRepresentationModule()
    MSIDP = model.DictionaryRepresentationModule()
    MHCSA_hsi = model.MHCSAB()
    MHCSA_msi = model.MHCSAB()
    fusion_module = model.FusionMoudle()

pretrain_dir = r"./checkpoints/train_models/20250120_12-00-00_MulFS-CAP-HSI-MSI_model"
checkpoint_name = "epoch99_iter100.pth"

checkpoint_path = os.path.join(pretrain_dir, checkpoint_name)

if not os.path.exists(checkpoint_path):
    print(f"❌ 模型文件不存在: {checkpoint_path}")
    exit(1)

print(f"📦 加载模型: {checkpoint_path}")
checkpoints = torch.load(checkpoint_path, map_location=device)

utils.load_state_dir(base_msi, checkpoints['bfe_msi'], device)
utils.load_state_dir(base_hsi, checkpoints['bfe_hsi'], device)
utils.load_state_dir(msi_MFE, checkpoints['msi_mfe'], device)
utils.load_state_dir(hsi_MFE, checkpoints['hsi_mfe'], device)
utils.load_state_dir(PAFE, checkpoints['pafe'], device)
utils.load_state_dir(fusion_decoder, checkpoints['fusion_decoder'], device)
utils.load_state_dir(decoder, checkpoints['decoder'], device)
utils.load_state_dir(MSIDP, checkpoints['msi_dgfp'], device)
utils.load_state_dir(HSIDP, checkpoints['hsi_dgfp'], device)
utils.load_state_dir(MN_msi, checkpoints['mn_msi'], device)
utils.load_state_dir(MN_hsi, checkpoints['mn_hsi'], device)
utils.load_state_dir(MHCSA_msi, checkpoints['mhcsab_msi'], device)
utils.load_state_dir(MHCSA_hsi, checkpoints['mhcsab_hsi'], device)
utils.load_state_dir(fusion_module, checkpoints['fusion_block'], device)

base_msi.eval()
base_hsi.eval()
msi_MFE.eval()
hsi_MFE.eval()
PAFE.eval()
fusion_decoder.eval()
decoder.eval()
MSIDP.eval()
HSIDP.eval()
MN_msi.eval()
MN_hsi.eval()
MHCSA_msi.eval()
MHCSA_hsi.eval()
fusion_module.eval()

print("✅ 模型加载完成！\n")

# ========== 测试循环 ==========
print("🚀 开始测试...")

all_metrics = {
    'PSNR': [],
    'SAM': [],
    'SSIM': [],
    'RMSE': []
}

results_per_image = []

for x in tqdm(test_data_iter, desc="测试进度"):
    hsi, msi, gt, original_size, file_name = x
    h_orig, w_orig = original_size[0].item(), original_size[1].item()
    file_name = file_name[0]

    hsi = hsi.to(device)
    msi = msi.to(device)
    gt = gt.to(device)

    with torch.no_grad():
        # ========== 上采样HSI ==========
        hsi_up = F.interpolate(
            hsi,
            size=(msi.size(2), msi.size(3)),
            mode='bilinear',
            align_corners=False
        )

        # ========== 前向推理（与固定尺寸版完全相同）==========
        hsi_base = base_hsi(hsi_up)
        msi_base = base_msi(msi)

        hsi_fe = hsi_MFE(hsi_base)
        msi_fe = msi_MFE(msi_base)

        hsi_f = PAFE(hsi_base)
        msi_f = PAFE(msi_base)

        hsi_e_f = MN_hsi(hsi_f)
        msi_e_f = MN_msi(msi_f)

        HSIDP_hsi_f, _ = HSIDP(hsi_e_f)
        MSIDP_msi_f, _ = MSIDP(msi_e_f)

        fixed_DP = HSIDP_hsi_f
        moving_DP = MSIDP_msi_f

        moving_DP_lw = model.df_window_partition(
            moving_DP, args.args.large_w_size, args.args.small_w_size
        )
        fixed_DP_sw = model.window_partition(
            fixed_DP, args.args.small_w_size, args.args.small_w_size
        )

        correspondence_matrixs = model.CMAP(
            fixed_DP_sw, moving_DP_lw, MHCSA_hsi, MHCSA_msi, True
        )

        msi_f_sample = model.feature_reorganization(correspondence_matrixs, msi_fe)
        fusion_image = fusion_module(hsi_fe, msi_f_sample)

        # ✅ 恢复原始尺寸（如果被调整过）
        if fusion_image.size(2) != h_orig or fusion_image.size(3) != w_orig:
            fusion_image = F.interpolate(
                fusion_image, (h_orig, w_orig),
                mode='bilinear', align_corners=False
            )
            gt = F.interpolate(
                gt, (h_orig, w_orig),
                mode='bilinear', align_corners=False
            )

        # ========== 计算评估指标 ==========
        psnr = calculate_psnr(fusion_image, gt, data_range=1.0)
        sam = calculate_sam(fusion_image, gt)
        ssim = calculate_ssim(fusion_image, gt, data_range=1.0)
        rmse = calculate_rmse(fusion_image, gt)

        all_metrics['PSNR'].append(psnr)
        all_metrics['SAM'].append(sam)
        all_metrics['SSIM'].append(ssim)
        all_metrics['RMSE'].append(rmse)

        results_per_image.append({
            'filename': file_name,
            'PSNR': psnr,
            'SAM': sam,
            'SSIM': ssim,
            'RMSE': rmse,
            'original_size': f"{h_orig}x{w_orig}"
        })

        # ========== 保存融合结果 ==========
        output_path = os.path.join(save_fusion_dir, file_name)
        sio.savemat(output_path, {'data': fusion_image.squeeze(0).cpu().numpy()})

        # ========== 清理显存 ==========
        del hsi_up, hsi_base, msi_base, hsi_fe, msi_fe
        del hsi_f, msi_f, hsi_e_f, msi_e_f
        del HSIDP_hsi_f, MSIDP_msi_f, fixed_DP, moving_DP
        del correspondence_matrixs, msi_f_sample, fusion_image
        torch.cuda.empty_cache()

# ========== 保存详细结果到CSV ==========
import csv

csv_path = os.path.join(save_metrics_dir, 'results_per_image.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'PSNR', 'SAM', 'SSIM', 'RMSE', 'original_size'])
    writer.writeheader()
    writer.writerows(results_per_image)

print(f"\n✅ 详细结果已保存至: {csv_path}")

# ========== 打印统计结果 ==========
print("\n" + "=" * 70)
print("📊 测试结果统计")
print("=" * 70)

for metric_name, values in all_metrics.items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    print(f"{metric_name:8s}: {mean_val:8.4f} ± {std_val:6.4f}  (min: {min_val:7.4f}, max: {max_val:7.4f})")

print("=" * 70)
print(f"✅ 融合结果已保存至: {save_fusion_dir}")
print(f"✅ 评估指标已保存至: {csv_path}")
print("=" * 70)