import argparse

parser = argparse.ArgumentParser(description='**MulFS-CAP-HSI-MSI**')

# ========== Pair 1: 原始配准数据 ==========
parser.add_argument('--hsi_train_dir', default=r'D:/datas/CAVEdata/Z_reconst', type=str)
parser.add_argument('--msi_train_dir', default=r'D:/datas/CAVEdata/Y_reconst', type=str)
parser.add_argument('--gt_train_dir', default=r'D:/datas/CAVEdata/X', type=str)

# ========== Pair 2: 形变配准数据 ==========
parser.add_argument('--hsi_deformed_train_dir', default=r'D:/datas/CAVEdata/Z_deformed', type=str)
parser.add_argument('--msi_deformed_train_dir', default=r'D:/datas/CAVEdata/Y_deformed', type=str)
parser.add_argument('--gt_deformed_train_dir', default=r'D:/datas/CAVEdata/X_deformed', type=str)

# ========== 测试数据 ==========
parser.add_argument('--hsi_test_dir', default=r'D:/datas/CAVEdata/Z_reconst', type=str)
parser.add_argument('--msi_test_dir', default=r'D:/datas/CAVEdata/Y_reconst', type=str)

# ========== 保存路径 ==========
parser.add_argument('--train_save_img_dir', default='./checkpoints/images', type=str)
parser.add_argument('--train_save_model_dir', default='./checkpoints/train_models', type=str)
parser.add_argument('--pretrain_model_dir', default='./pretrain')
parser.add_argument('--save_image_num', dest='save_image_num', default=10, type=int)  # ✅ 减少保存频率
parser.add_argument('--save_model_num', dest='save_model_num', default=20, type=int)  # ✅ 减少保存频率

# ========== 🔥 核心显存优化参数 🔥 ==========
parser.add_argument('--img_size', type=int, default=512)  # ✅ 512 → 128（显存减少93%）
parser.add_argument('--small_w_size', type=float, default=8)   # ✅ 32 → 8
parser.add_argument('--large_w_size', type=float, default=12)  # ✅ 52 → 12
parser.add_argument('--batch_size', dest='batch_size', default=1, type=int)

# ========== 训练参数 ==========
parser.add_argument('--LR', type=float, default=0.0002)
parser.add_argument('--LR_target', type=float, default=0.001)
parser.add_argument('--Epoch', type=float, default=100)
parser.add_argument('--Warm_epoch', type=float, default=160)
parser.add_argument('--dropout', type=float, default=0.1)

args = parser.parse_args()