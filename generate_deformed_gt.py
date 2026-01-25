import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from pathlib import Path
from PIL import Image

# ====================== 1. 配置参数（✅ 新增 X_rigid_only 路径，其余不变） ======================
ROOT_PATH = r"D:\datas\CAVEdata"
GT_RAW_DIR = os.path.join(ROOT_PATH, "X")  # 原始GT/MSI文件夹
GT_DEFORMED_SAVE_DIR = os.path.join(ROOT_PATH, "X_deformed")  # 刚性+非刚性形变 保存目录
GT_RIGID_ONLY_SAVE_DIR = os.path.join(ROOT_PATH, "X_rigid_only")  # ✅ 新增：仅刚性形变 保存目录

# 形变参数：
RIGID_PARAMS = {"degrees": 3, "translate": 0.03, "scale": (0.95, 1.05)}
ELASTIC_PARAMS = {"kernel_size": 41, "sigma": 3}

# 设备配置：自动GPU/CPU适配
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")


# ====================== 2. 刚性形变+非刚性形变 原论文原版（无需修改，核心不变） ======================
class AffineTransform(torch.nn.Module):
    """刚性形变：旋转、平移、缩放（原论文）"""

    def __init__(self, degrees=3, translate=0.03, scale=(0.95, 1.05), return_warp=True):
        super().__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.return_warp = return_warp

    def forward(self, x):
        batch_size, C, H, W = x.shape
        theta = torch.zeros((batch_size, 2, 3), device=x.device)
        for i in range(batch_size):
            angle = np.random.uniform(-self.degrees, self.degrees) * np.pi / 180.0
            sx = np.random.uniform(self.scale[0], self.scale[1])
            sy = np.random.uniform(self.scale[0], self.scale[1])
            tx = np.random.uniform(-self.translate, self.translate) * W
            ty = np.random.uniform(-self.translate, self.translate) * H

            theta[i, 0, 0] = sx * np.cos(angle)
            theta[i, 0, 1] = -sy * np.sin(angle)
            theta[i, 0, 2] = tx
            theta[i, 1, 0] = sx * np.sin(angle)
            theta[i, 1, 1] = sy * np.cos(angle)
            theta[i, 1, 2] = ty
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        warped = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
        if self.return_warp:
            return warped, grid
        return warped


class ElasticTransform(torch.nn.Module):
    """
    HSI-MSI 安全版 Elastic Transform
    - 位移幅度：≈ 8 像素
    - 位移单位：归一化坐标
    - padding_mode：zeros
    - 含能量保护（mean-preserving）
    """

    def __init__(self, kernel_size=41, sigma=3, return_warp=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma  # 🔥 这里 sigma 现在代表“最大像素位移 ≈ 8”
        self.return_warp = return_warp

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        device = x.device

        # ===================== 1️⃣ 生成随机位移场（像素单位） =====================
        dx = torch.randn((B, 1, H, W), device=device)
        dy = torch.randn((B, 1, H, W), device=device)

        # ===================== 2️⃣ 高斯平滑（生成连续形变） =====================
        pad = self.kernel_size // 2
        dx = F.pad(dx, [pad] * 4, mode='reflect')
        dy = F.pad(dy, [pad] * 4, mode='reflect')

        coords = torch.arange(self.kernel_size, device=device) - pad
        g = torch.exp(-(coords ** 2) / (2 * (self.kernel_size / 6) ** 2))
        g = g / g.sum()
        kernel = g[:, None] * g[None, :]
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)

        dx = F.conv2d(dx, kernel, padding=0)
        dy = F.conv2d(dy, kernel, padding=0)

        # ===================== 3️⃣ 控制位移幅度：≈ 8 像素 =====================
        dx = dx / (dx.abs().max() + 1e-6) * self.sigma
        dy = dy / (dy.abs().max() + 1e-6) * self.sigma

        # ===================== 4️⃣ 像素位移 → 归一化坐标位移 =====================
        dx_norm = dx / (W - 1)
        dy_norm = dy / (H - 1)

        # ===================== 5️⃣ 构建 grid（归一化坐标） =====================
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )

        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)

        deform_grid = base_grid + torch.cat(
            [dx_norm, dy_norm], dim=1
        ).permute(0, 2, 3, 1)

        # ===================== 6️⃣ 采样（zeros padding，防止边界能量污染） =====================
        warped = F.grid_sample(
            x,
            deform_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        # ===================== 7️⃣ 能量保护（对 HSI 极其重要） =====================
        mean_before = x.mean(dim=[2, 3], keepdim=True)
        mean_after = warped.mean(dim=[2, 3], keepdim=True)
        warped = warped * (mean_before / (mean_after + 1e-6))

        if self.return_warp:
            return warped, deform_grid
        return warped

# ====================== 3. 核心函数：✅ 同时生成【仅刚性】+【刚性+非刚性】双版本形变图像 ======================
def generate_deformed_gt_images():
    # ✅ 新增：同时创建两个保存目录，不存在则自动新建
    os.makedirs(GT_DEFORMED_SAVE_DIR, exist_ok=True)
    os.makedirs(GT_RIGID_ONLY_SAVE_DIR, exist_ok=True)

    # 只筛选 有效图像文件 ：.mat/.MAT 你的文件都是这个格式，精准过滤
    valid_suffix = ['.mat', '.MAT']
    gt_file_paths = [p for p in Path(GT_RAW_DIR).glob("*.*") if p.suffix in valid_suffix]

    print(f"✅ 共找到 {len(gt_file_paths)} 张有效.mat图像，开始生成双版本形变图像...")
    print(f"📌 版本1：仅刚性形变 → 保存至 {GT_RIGID_ONLY_SAVE_DIR}")
    print(f"📌 版本2：刚性+非刚性形变 → 保存至 {GT_DEFORMED_SAVE_DIR}")
    if len(gt_file_paths) == 0:
        print("❌ 错误：未找到任何.mat文件！请检查文件夹路径是否正确")
        return

    # 初始化形变模块
    rigid_transform = AffineTransform(**RIGID_PARAMS).to(device)
    elastic_transform = ElasticTransform(**ELASTIC_PARAMS).to(device)

    # 逐张处理：单张读取→形变→双版本保存→释放内存
    success_count = 0
    for idx, file_path in enumerate(gt_file_paths):
        file_name = file_path.name
        # ✅ 拼接两个版本的保存路径，文件名完全一致
        save_path_deformed = os.path.join(GT_DEFORMED_SAVE_DIR, file_name)
        save_path_rigid = os.path.join(GT_RIGID_ONLY_SAVE_DIR, file_name)
        img_np = None

        try:
            # ================ 万能读取.mat文件，彻底无key判断，根治之前的报错 ================
            mat_data = sio.loadmat(str(file_path))
            mat_values = [v for k, v in mat_data.items() if not k.startswith('__')]
            img_np = mat_values[0]

            # ========== 数据合法性校验 ==========
            if img_np is None or img_np.ndim < 2:
                print(f"⚠️ 跳过 {file_name} ：数据为空或维度异常")
                continue

            # ========== 数据格式处理 + 归一化（CAVE数据集必备） ==========
            img_np = img_np.astype(np.float32)
            # 自动适配CAVE的维度：(H,W,C) ↔ (C,H,W) 转成PyTorch标准格式
            if img_np.ndim == 3:
                if img_np.shape[-1] < img_np.shape[0] and img_np.shape[-1] < img_np.shape[1]:
                    img_np = np.transpose(img_np, (2, 0, 1))
            # 归一化到 [0, 1] 区间，模型训练必须
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

            # ========== 格式转换：numpy → PyTorch张量 ==========
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)  # (1, C, H, W)

            # ========== ✅ 核心形变逻辑（一次计算，得到两个结果，效率拉满） ==========
            rigid_warped, _ = rigid_transform(img_tensor)  # 结果1：仅刚性形变
            deformed_img, _ = elastic_transform(rigid_warped)  # 结果2：刚性+非刚性形变

            # ========== ✅ 第一步：保存【仅刚性形变】的图像 到 X_rigid_only ==========
            rigid_img_np = rigid_warped.squeeze(0).cpu().numpy()
            sio.savemat(save_path_rigid, {'data': rigid_img_np})

            # ========== ✅ 第二步：保存【刚性+非刚性形变】的图像 到 X_deformed ==========
            deformed_img_np = deformed_img.squeeze(0).cpu().numpy()
            sio.savemat(save_path_deformed, {'data': deformed_img_np})

            success_count += 1
            # 打印进度
            if (idx + 1) % 5 == 0:
                print(f"✅ 进度: {idx + 1}/{len(gt_file_paths)} 张，成功生成 {success_count} 张")

        except Exception as e:
            print(f"❌ 处理 {file_name} 失败：{str(e)}")
            continue



    # ✅ 打印双版本生成结果
    print(f"\n🎉 双版本形变图像全部生成完成！共成功生成 {success_count} 张")
    print(f"✅ 仅刚性形变图像 → {GT_RIGID_ONLY_SAVE_DIR}")
    print(f"✅ 刚性+非刚性形变图像 → {GT_DEFORMED_SAVE_DIR}")
    print(f"✅ 所有文件名与原始文件完全一致，可直接用于训练，无任何错位！")


# ====================== 运行函数 ======================
if __name__ == "__main__":
    generate_deformed_gt_images()