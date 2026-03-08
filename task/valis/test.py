import torch
import time
import os
from valis import registration

slide_src_dir = "../../test/data"
results_dst_dir = "../../test/valis_registration"

# 计时与创建配准器
start = time.time()
registrar = registration.Valis(slide_src_dir, results_dst_dir)

# 执行配准
rigid_registrar, non_rigid_registrar, error_df = registrar.register()
stop = time.time()
elapsed = stop - start

# 设置输出路径并保存配准后的切片
registered_slide_dst_dir = os.path.join("../../test/valis_registration", registrar.name)
start = time.time()
registrar.warp_and_save_slides(registered_slide_dst_dir)
stop = time.time()
elapsed = stop - start
print(f"saving {registrar.size} slides took {elapsed/60} minutes")

# 关闭 JVM
registration.kill_jvm()

# 运行报错，则需要手动加载libstdc++
# export LD_PRELOAD=/home/lzl/.conda/envs/tiatoolbox/lib/libstdc++.so.6