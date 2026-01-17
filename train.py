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
from utils.utils import save_img
from tqdm import tqdm

import args
from loss import loss as Loss
from model import model
from utils import utils

# 全局常量定义（可保留在外部）
model_name = "MulFS-CAP"
device_id = "0"


# 定义函数/类（必须放在if __name__外，供子进程导入）
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
    def __init__(self, hsi_dir, msi_dir,gt_dir,transform):  # ✅ 你已经定义了hsi/msi/gt三输入！只是没用到
        super(TrainDataset, self).__init__()
        self.hsi_dir = hsi_dir
        self.msi_dir = msi_dir
        self.gt_dir =gt_dir
        self.hsi_path, self.hsi_paths = self.find_file(self.hsi_dir)
        self.msi_path, self.msi_paths = self.find_file(self.msi_dir)
        self.gt_path, self.gt_paths = self.find_file(self.gt_dir)
        assert (len(self.hsi_path) == len(self.msi_path)==len(self.gt_path)), "可见光和红外文件数量不相等" # 提示语还是IR-VIS的
        self.transform = transform

    def find_file(self, dir):
        """路径拼接修复，子进程兼容"""
        path = os.listdir(dir)
        if os.path.isdir(os.path.join(dir, path[0])):
            paths = []
            for dir_name in os.listdir(dir):
                subdir_full = os.path.join(dir, dir_name)
                for file_name in os.listdir(subdir_full):
                    full_path = os.path.join(subdir_full, file_name)
                    paths.append(full_path)
        else:
            paths = list(Path(dir).glob('*'))
        # 过滤：只保留png/jpg/jpeg 灰度图文件！【IR-VIS专属】
        paths = [p for p in paths if os.path.getsize(p) > 0 and str(p).lower().endswith(('.png', '.jpg', '.jpeg'))]
        return path, paths

    def read_image(self, path):
        # 【IR-VIS核心专属】读取单通道灰度图：PIL读图→转灰度图→transform预处理
        img = Image.open(str(path)).convert('L')  # convert('L')=单通道灰度图，固定1通道！
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        hsi_path = self.hsi_paths[index]
        msi_path = self.msi_paths[index]
        gt_path = self.gt_paths[index]
        hsi_img = self.read_image(hsi_path)  # 实际读的是IR灰度图
        msi_img = self.read_image(msi_path)  # 实际读的是VIS灰度图
        gt_img = self.read_image(gt_path)
        return hsi_img, msi_img,gt_img

    def __len__(self):
        return len(self.vis_path)  # ❌ BUG 1：self.vis_path不存在！应该是len(self.hsi_paths)


# 核心：所有执行逻辑必须包裹到if __name__ == '__main__'中
if __name__ == '__main__':
    # ========== 初始化环境 ==========
    os.environ['CUDA_LAUNCH_BLOCKING'] = device_id
    device = torch.device("cuda:" + device_id if torch.cuda.is_available() else "cpu")

    # 关键修复：Windows仅支持spawn启动方式，移除forkserver
    # 强制设置为spawn（Windows默认，但显式设置避免冲突）
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 若已设置过，忽略报错

    # ========== 初始化保存目录 ==========
    now = int(time.time())
    timeArr = time.localtime(now)
    nowTime = time.strftime("%Y%m%d_%H-%M-%S", timeArr)
    save_model_dir = args.args.train_save_model_dir + "/" + nowTime + "_" + model_name + "_model"
    save_img_dir = args.args.train_save_img_dir + "/" + nowTime + "_" + model_name + "_img"
    utils.check_dir(save_model_dir)
    utils.check_dir(save_img_dir)

    # ========== 数据加载器初始化 ==========
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize([args.args.img_size, args.args.img_size]),
        torchvision.transforms.ToTensor()  # (0, 255) -> (0, 1)
    ])

    dataset = TrainDataset(args.args.hsi_train_dir, args.args.msi_train_dir, tf)

    # 关键：设置num_workers=4，开启多进程加速（Windows兼容）
    data_iter = data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=args.args.batch_size,
        num_workers=4,  # Windows下建议不超过CPU核心数（如4核设为4，8核设为8）
        pin_memory=True,  # 开启pin_memory加速GPU数据传输
        drop_last=True,  # 丢弃最后一个不完整批次，避免训练报错
        multiprocessing_context=torch.multiprocessing.get_context('spawn')  # 显式指定spawn上下文
    )

    iter_num = int(dataset.__len__() / args.args.batch_size)
    save_image_iter = int(iter_num / args.args.save_image_num)

    # ========== 模型初始化 ==========
    Lgrad = Loss.L_Grad().to(device)
    CC = Loss.CorrelationCoefficient().to(device)
    Lcorrespondence = Loss.L_correspondence()

    with torch.no_grad():
        base = model.base()
        vis_MFE = model.FeatureExtractor()
        ir_MFE = model.FeatureExtractor()
        fusion_decoder = model.Decoder()
        PAFE = model.FeatureExtractor()
        decoder = model.Decoder()
        MN_vis = model.Enhance()
        MN_ir = model.Enhance()
        VISDP = model.DictionaryRepresentationModule()
        IRDP = model.DictionaryRepresentationModule()
        ImageDeformation = model.ImageTransform()
        MHCSA_vis = model.MHCSAB()
        MHCSA_ir = model.MHCSAB()
        fusion_module = model.FusionMoudle()

    # 模型训练模式+设备迁移
    base.train().to(device)
    vis_MFE.train().to(device)
    ir_MFE.train().to(device)
    fusion_decoder.train().to(device)
    PAFE.train().to(device)
    decoder.train().to(device)
    VISDP.train().to(device)
    IRDP.train().to(device)
    MN_vis.train().to(device)
    MN_ir.train().to(device)
    MHCSA_vis.train().to(device)
    MHCSA_ir.train().to(device)
    fusion_module.train().to(device)

    # ========== 优化器初始化 ==========
    optimizer_FE = torch.optim.Adam([{'params': base.parameters()},
                                     {'params': vis_MFE.parameters()}, {'params': ir_MFE.parameters()},
                                     {'params': fusion_decoder.parameters()},
                                     {'params': PAFE.parameters()}, {'params': decoder.parameters()},
                                     {'params': MN_vis.parameters()}, {'params': MN_ir.parameters()}],
                                    lr=0.0002)
    optimizer_VISDP = torch.optim.Adam(VISDP.parameters(), lr=0.0008)
    optimizer_IRDP = torch.optim.Adam(IRDP.parameters(), lr=0.0008)
    optimizer_MHCSAvis = torch.optim.Adam(MHCSA_vis.parameters(), lr=args.args.LR)
    optimizer_MHCSAir = torch.optim.Adam(MHCSA_ir.parameters(), lr=args.args.LR)
    optimizer_FusionModule = torch.optim.Adam(fusion_module.parameters(), lr=0.0002)


    # ========== 训练函数定义（放在main内，避免子进程重复加载） ==========
    def train(epoch):
        epoch_loss_VISDP = []
        epoch_loss_IRDP = []
        epoch_loss_same = []
        epoch_loss_correspondence_matrix = []
        epoch_loss_correspondence_predict = []

        for step, x in enumerate(data_iter):
            vis = x[0].to(device, non_blocking=True)  # non_blocking加速GPU传输
            ir = x[1].to(device, non_blocking=True)

            with torch.no_grad():
                vis_d, ir_d, _, index_r, _ = ImageDeformation(vis, ir)

            vis_1 = base(vis)
            vis_d_1 = base(vis_d)
            ir_1 = base(ir)
            ir_d_1 = base(ir_d)

            vis_fe = vis_MFE(vis_1)
            ir_fe = ir_MFE(ir_1)
            simple_fusion_f_1 = vis_fe + ir_fe
            fusion_image_1, fusion_f_1 = fusion_decoder(simple_fusion_f_1)
            vis_d_fe = vis_MFE(vis_d_1)
            ir_d_fe = ir_MFE(ir_d_1)
            simple_fusion_d_f_1 = vis_d_fe + ir_d_fe
            fusion_d_image_1, fusion_d_f_1 = fusion_decoder(simple_fusion_d_f_1)

            vis_f = PAFE(vis_1)
            ir_f = PAFE(ir_1)
            simple_fusion_f = vis_f + ir_f
            fusion_image, fusion_f = decoder(simple_fusion_f)
            vis_d_f = PAFE(vis_d_1)
            ir_d_f = PAFE(ir_d_1)
            simple_fusion_d_f = vis_d_f + ir_d_f
            fusion_d_image, fusion_d_f = decoder(simple_fusion_d_f)

            vis_e_f = MN_vis(vis_f)
            ir_e_f = MN_ir(ir_f)
            vis_d_e_f = MN_vis(vis_d_f)
            ir_d_e_f = MN_ir(ir_d_f)

            VISDP_vis_f, _ = VISDP(vis_e_f)
            IRDP_ir_f, _ = IRDP(ir_e_f)
            VISDP_vis_d_f, _ = VISDP(vis_d_e_f)
            IRDP_ir_d_f, _ = IRDP(ir_d_e_f)

            fixed_DP = VISDP_vis_f
            moving_DP = IRDP_ir_d_f

            moving_DP_lw = model.df_window_partition(moving_DP, args.args.large_w_size, args.args.small_w_size)
            fixed_DP_sw = model.window_partition(fixed_DP, args.args.small_w_size, args.args.small_w_size)

            correspondence_matrixs = model.CMAP(fixed_DP_sw, moving_DP_lw, MHCSA_vis, MHCSA_ir,
                                                True)

            ir_d_f_sample = model.feature_reorganization(correspondence_matrixs, ir_d_fe)
            fusion_image_sample = fusion_module(vis_fe, ir_d_f_sample)

            # 计算损失
            loss_fusion = Lgrad(vis, ir, fusion_image) + Loss.Loss_intensity(vis, ir, fusion_image) + \
                          Lgrad(vis_d, ir_d, fusion_d_image) + Loss.Loss_intensity(vis_d, ir_d, fusion_d_image)
            loss_fusion_1 = Lgrad(vis, ir, fusion_image_1) + Loss.Loss_intensity(vis, ir, fusion_image_1) + \
                            Lgrad(vis_d, ir_d, fusion_d_image_1) + Loss.Loss_intensity(vis_d, ir_d, fusion_d_image_1)
            loss_0 = loss_fusion
            loss_VISDP = - CC(VISDP_vis_f, fusion_f.detach()) - CC(VISDP_vis_d_f, fusion_d_f.detach())
            loss_IRDP = - CC(IRDP_ir_f, fusion_f.detach()) - CC(IRDP_ir_d_f, fusion_d_f.detach())
            loss_same = F.mse_loss(VISDP_vis_f, IRDP_ir_f) + F.mse_loss(VISDP_vis_d_f, IRDP_ir_d_f)
            loss_1 = 2 * (loss_VISDP + loss_IRDP + loss_same)
            loss_2 = Lgrad(vis, ir, fusion_image_sample) + Loss.Loss_intensity(vis, ir, fusion_image_sample)
            loss_correspondence_matrix, loss_correspondence_matrix_1 = Lcorrespondence(
                correspondence_matrixs, index_r)
            loss_3 = 4 * (loss_correspondence_matrix + loss_correspondence_matrix_1)
            loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_fusion_1

            # 反向传播
            optimizer_VISDP.zero_grad()
            optimizer_IRDP.zero_grad()
            optimizer_MHCSAvis.zero_grad()
            optimizer_MHCSAir.zero_grad()
            optimizer_FusionModule.zero_grad()
            optimizer_FE.zero_grad()
            loss.backward()
            optimizer_FE.step()
            optimizer_VISDP.step()
            optimizer_IRDP.step()
            optimizer_MHCSAvis.step()
            optimizer_MHCSAir.step()
            optimizer_FusionModule.step()

            # 记录损失
            epoch_loss_VISDP.append(loss_VISDP.item())
            epoch_loss_IRDP.append(loss_IRDP.item())
            epoch_loss_same.append(loss_same.item())
            epoch_loss_correspondence_matrix.append(loss_correspondence_matrix.item())
            epoch_loss_correspondence_predict.append(loss_correspondence_matrix_1.item())

            # 保存图像
            if step % save_image_iter == 0:
                epoch_step_name = str(epoch) + "epoch" + str(step) + "step"
                if epoch % 2 == 0:
                    output_name = save_img_dir + "/" + epoch_step_name + ".jpg"
                    out = torch.cat([vis, ir_d, fusion_image_1, fusion_image_sample, fusion_d_image_1], dim=2)
                    save_img(out, output_name)

            # 保存模型
            if ((epoch + 1) == args.args.Epoch and (step + 1) % iter_num == 0) or (
                    epoch % args.args.save_model_num == 0 and (step + 1) % iter_num == 0):
                ckpts = {
                    "bfe": base.state_dict(),
                    "ir_mfe": ir_MFE.state_dict(),
                    "vis_mfe": vis_MFE.state_dict(),
                    "pafe": PAFE.state_dict(),
                    "mn_ir": MN_ir.state_dict(),
                    "mn_vis": MN_vis.state_dict(),
                    "ir_dgfp": IRDP.state_dict(),
                    "vis_dgfp": VISDP.state_dict(),
                    "mhcsab_ir": MHCSA_ir.state_dict(),
                    "mhcsab_vis": MHCSA_vis.state_dict(),
                    "fusion_block": fusion_module.state_dict(),
                }
                save_dir = '{:s}/epoch{:d}_iter{:d}.pth'.format(save_model_dir, epoch, step + 1)
                torch.save(ckpts, save_dir)

        # 打印epoch损失
        epoch_loss_correspondence_matrix_mean = np.mean(epoch_loss_correspondence_matrix)
        epoch_loss_correspondence_predict_mean = np.mean(epoch_loss_correspondence_predict)
        epoch_loss_VISDP_mean = np.mean(epoch_loss_VISDP)
        epoch_loss_IRDP_mean = np.mean(epoch_loss_IRDP)
        epoch_loss_same_mean = np.mean(epoch_loss_same)

        print()
        print(" -epoch " + str(epoch))
        print(" -loss_cm " + str(epoch_loss_correspondence_matrix_mean) + " -loss_cp " + str(
            epoch_loss_correspondence_predict_mean))
        print(" -loss_VISDP " + str(epoch_loss_VISDP_mean) + " -loss_IRDP " + str(
            epoch_loss_IRDP_mean))
        print(" -loss_same " + str(epoch_loss_same_mean))


    # ========== 启动训练循环 ==========
    for epoch in tqdm(range(args.args.Epoch)):
        if epoch < args.args.Warm_epoch:
            warmup_learning_rate(optimizer_MHCSAvis, epoch)
            warmup_learning_rate(optimizer_MHCSAir, epoch)
        else:
            adjust_learning_rate(optimizer_MHCSAvis, epoch)
            adjust_learning_rate(optimizer_MHCSAir, epoch)

        train(epoch)