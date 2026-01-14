import os
import sys
import pathlib
from typing import Union

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(parent_dir)

import deeperhistreg
from deeperhistreg.dhr_pipeline.registration_params import default_initial, default_initial_nonrigid_high_resolution

# region 定义路径
source_path = "../../test/F2025-00089.svs"
target_path = "../../test/2025-03-05_15_04_40.svs"
output_path = "../../test/he2ihc_output"  # 输出目录
# endregion

# region 配置参数
source_path = pathlib.Path(source_path)
target_path = pathlib.Path(target_path)
output_path = pathlib.Path(output_path)

# 获取默认参数
registration_params: dict = default_initial()

# 关键：SVS 格式需要使用 'openslide' loader，而不是 'pil'
registration_params['loading_params']['loader'] = 'openslide'

# 其他配置
save_displacement_field: bool = True           # 保存位移场（用于后续 landmarks/分割变换）
copy_target: bool = False                      # 复制 target 到输出目录
delete_temporary_results: bool = False         # 保留临时结果（调试用）
case_name: str = f"{source_path.stem}_{target_path.stem}"
temporary_path: Union[str, pathlib.Path] = output_path / f"{source_path.stem}_{target_path.stem}_TEMP"
# endregion

# region 创建配置字典
config = dict()
config['source_path'] = str(source_path)       # 转为字符串更稳定
config['target_path'] = str(target_path)
config['output_path'] = str(output_path)
config['registration_parameters'] = registration_params
config['case_name'] = case_name
config['save_displacement_field'] = save_displacement_field
config['copy_target'] = copy_target
config['delete_temporary_results'] = delete_temporary_results
config['temporary_path'] = str(temporary_path)
# endregion

# region 执行配准
print(f"Source: {source_path}")
print(f"Target: {target_path}")
print(f"Output: {output_path}")
print("开始配准...")

deeperhistreg.run_registration(**config)

print("配准完成！")
# endregion