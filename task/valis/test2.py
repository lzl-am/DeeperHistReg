import torch
import time
import os
import pyvips
from valis import registration

slide_src_dir = "../../test/data"
results_dst_dir = "../../test/valis_registration2"

start = time.time()
registrar = registration.Valis(slide_src_dir, results_dst_dir)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()
stop = time.time()

registered_slide_dst_dir = os.path.join("../../test/valis_registration2", registrar.name)
os.makedirs(registered_slide_dst_dir, exist_ok=True)

start = time.time()

for src_f in registrar.get_sorted_img_f_list():
    slide_obj = registrar.get_slide(src_f)
    
    # 获取 warp 后的 pyvips 图像
    warped_slide = slide_obj.warp_slide(level=0, non_rigid=True, crop=True)
    
    dst_f = os.path.join(registered_slide_dst_dir, slide_obj.name + ".svs")
    
    # 保存为 SVS 兼容的金字塔 JPEG TIFF
    warped_slide.tiffsave(
        dst_f,
        compression="jpeg",       # SVS 通常用 JPEG 压缩
        Q=90,                      # JPEG 质量
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,              # 生成图像金字塔
        bigtiff=True,              # 支持大文件
        depth="onetile",           # 金字塔层级方式
        subifd=True,               # 将子金字塔写入 SubIFD（SVS 标准）
    )
    print(f"Saved: {dst_f}")

stop = time.time()
print(f"Saving {registrar.size} slides took {(stop-start)/60:.2f} minutes")

registration.kill_jvm()

# 运行报错，则需要手动加载libstdc++
# export LD_PRELOAD=/home/lzl/.conda/envs/tiatoolbox/lib/libstdc++.so.6