import scipy.io as sio
import numpy as np
from pathlib import Path

print("=" * 70)
print("🔍 步骤1：检查原始GT图像 (X文件夹)")
print("=" * 70)

x_dir = r"D:/datas/CAVEdata/Z"
x_files = sorted(Path(x_dir).glob("*.mat"))

print(f"文件夹: {x_dir}")
print(f"文件数量: {len(x_files)}\n")

if len(x_files) == 0:
    print("❌ 错误：X文件夹为空！")
else:
    for i, file_path in enumerate(x_files[:32]):
        print(f"📄 文件 {i + 1}: {file_path.name}")

        mat_data = sio.loadmat(str(file_path))
        valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]

        if len(valid_keys) > 0:
            data = mat_data[valid_keys[0]]
            print(f"   Keys: {valid_keys}")
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")
            print(f"   Min: {data.min():.6f}")
            print(f"   Max: {data.max():.6f}")
            print(f"   Mean: {data.mean():.6f}")

            # ✅ 修正判断逻辑
            if data.max() < 0.01:  # 正常图像max应该接近1.0
                print(f"   ❌ 异常：Max值太小，数据可能损坏！")
            elif data.max() > 1.1:
                print(f"   ⚠️ 警告：Max值>1，未归一化")
            else:
                print(f"   ✅ 数据正常")
        print()

print("=" * 70 + "\n")