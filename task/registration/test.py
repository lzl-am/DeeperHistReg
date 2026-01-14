import os
import sys

import h5py
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(parent_dir)

from deeperhistreg.dhr_input_output.dhr_loaders import OpenSlideLoader, LoadMode


# region 特征提取尺度
feature_path = "../../test/F2025-00089_patches.h5"

with h5py.File(feature_path, 'r') as f:
    # 读取坐标数据
    coords = f['coords'][:]  # numpy array, shape (20965, 2)
    
    # 读取元信息
    coords_attrs = dict(f['coords'].attrs)
    print(f"coords attrs: {coords_attrs}")
    patch_size = f['coords'].attrs['patch_size']
    magnification = f['coords'].attrs['target_magnification']
    slide_name = f['coords'].attrs['name']
    
    print(f"切片名: {slide_name}")
    print(f"Patch数量: {len(coords)}")
    print(f"Patch大小: {patch_size}x{patch_size}")
    print(f"放大倍数: {magnification}x")
    print(f"前5个坐标:\n{coords[:5]}")
# endregion

# region 数据加载
source_loader = OpenSlideLoader(
    image_path="../../test/2025-03-05_15_04_40.svs",
    mode=LoadMode.NUMPY,
)
print(f'source金字塔层数: {source_loader.num_levels}')
print(f'source各层分辨率: {source_loader.resolutions}')

target_loader = OpenSlideLoader(
    image_path="../../test/F2025-00089.svs",
    mode=LoadMode.NUMPY,
)
print(f'target金字塔层数: {target_loader.num_levels}')
print(f'target各层分辨率: {target_loader.resolutions}')

load_level = 1
source = source_loader.load_level(level=load_level)
target = target_loader.load_level(level=load_level)
print(f"Source shape: {source.shape}")
print(f"Target shape: {target.shape}")
# endregion

# region 配准前数据展示
plt.figure(dpi=200)
plt.imshow(source)
plt.axis('off')
plt.savefig('../../test/source.png', bbox_inches='tight', pad_inches=0)
plt.close()

plt.figure(dpi=200)
plt.imshow(target)
plt.axis('off')
plt.savefig('../../test/target.png', bbox_inches='tight', pad_inches=0)
plt.close()

source_patch = source_loader.load_region(
    level=0,
    offset=(1000, 10000), 
    shape=(256, 256)
)
plt.figure(dpi=200)
plt.imshow(source_patch)
plt.axis('off')
plt.savefig('../../test/source_patch.png', bbox_inches='tight', pad_inches=0)
plt.close()

target_patch = target_loader.load_region(
    level=0,
    offset=(1000, 10000), 
    shape=(256, 256)
)
plt.figure(dpi=200)
plt.imshow(target_patch)
plt.axis('off')
plt.savefig('../../test/target_patch.png', bbox_inches='tight', pad_inches=0)
plt.close()
# endregion