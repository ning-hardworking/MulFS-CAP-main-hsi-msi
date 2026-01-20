import scipy.io as sio
import numpy as np
from pathlib import Path

# 检查Z_reconst文件夹
z_dir = r"D:/datas/CAVEdata/Z_reconst"
z_files = sorted(Path(z_dir).glob("*.mat"))

print("=" * 70)
print(f"🔍 检查 Z_reconst 文件夹:")
print(f"   路径: {z_dir}")
print(f"   文件数量: {len(z_files)}")
print("=" * 70)

if len(z_files) == 0:
    print("❌ 错误：文件夹为空！")
else:
    # 检查前3个文件
    for i, file_path in enumerate(z_files[:3]):
        print(f"\n📄 文件 {i + 1}: {file_path.name}")

        mat_data = sio.loadmat(str(file_path))

        # 打印所有键
        print(f"   Keys: {[k for k in mat_data.keys() if not k.startswith('__')]}")

        # 读取数据
        valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if len(valid_keys) > 0:
            data = mat_data[valid_keys[0]]
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")
            print(f"   Min: {data.min():.6f}")
            print(f"   Max: {data.max():.6f}")
            print(f"   Mean: {data.mean():.6f}")

            # 检查是否全零
            if data.max() == 0 and data.min() == 0:
                print(f"   ❌ 警告：数据全为0！")
            else:
                print(f"   ✅ 数据正常")
        else:
            print(f"   ❌ 未找到有效数据键")

print("\n" + "=" * 70)