import cv2
import numpy as np
import shutil
from pathlib import Path
from scipy import ndimage
from skimage import color, exposure, measure, morphology
from matplotlib import pyplot as plt
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.tools.registration.wsi_registration import (
    AffineWSITransformer,
    DFBRegister,
    apply_affine_transformation,
    apply_bspline_transform,
    estimate_bspline_transform,
    match_histograms,
)


# region 辅助函数
def preprocess_image(image: np.ndarray) -> np.ndarray:
    # 将RGB图像转换为灰度图像
    image = color.rgb2gray(image)

    # 对比度增强:
    # - 计算图像第0.5和99.5百分位数的像素值
    # - 将这个范围线性拉伸到[0,1],裁剪极端值,提高主要内容的对比度
    image = exposure.rescale_intensity(
        image,
        in_range=tuple(np.percentile(image, (0.5, 99.5))),
    )

    # 将归一化的[0,1]浮点值缩放到[0,255]的8位整数范围
    image = image * 255
    return image.astype(np.uint8)

def post_processing_mask(mask: np.ndarray) -> np.ndarray:
    """对二值掩码进行清理，只保留最大的连通区域"""
    # 填充掩码中的小孔洞
    mask = ndimage.binary_fill_holes(mask, structure=np.ones((3, 3))).astype(int)

    # 定义期望的标签数
    # 背景(0) + 前景(1) = 2个标签
    num_unique_labels = 2

    # 对二值掩码进行连通区域标记
    # 背景(0) + N个对象，共N+1个唯一标签
    label_img = measure.label(mask)
    
    if len(np.unique(label_img)) > num_unique_labels:
        regions = measure.regionprops(label_img)

        mask = mask.astype(bool)
        all_area = [i.area for i in regions]
        second_max = max([i for i in all_area if i != max(all_area)])

        # 移除所有面积 < second_max+1 的区域
        # 结果:只保留最大的连通区域
        mask = morphology.remove_small_objects(mask, min_size=second_max+1)

    return mask.astype(np.uint8)

def print_wsi_metadata(reader, name="WSI"):
    """打印WSI的完整元数据信息"""
    info = reader.info
    
    print(f"\n{'='*50}")
    print(f"{name} 元数据信息")
    print(f"{'='*50}")
    print(f"文件路径: {info.file_path}")
    print(f"切片尺寸 (W×H): {info.slide_dimensions[0]} × {info.slide_dimensions[1]}")
    print(f"MPP (微米/像素): {info.mpp}")
    print(f"物镜倍率: {info.objective_power}×")
    print(f"\n金字塔层级信息:")
    print(f"  层级数量: {info.level_count}")
    for i, (dims, downsample) in enumerate(zip(info.level_dimensions, 
                                                 info.level_downsamples)):
        print(f"  Level {i}: {dims[0]}×{dims[1]}, 下采样: {downsample:.2f}×")
# endregion


# region 数据配置
registration_level = 2

he_path = "/data/med/MMR/MMR-HCMI-HE-SVS/F2025-00089.svs"
ihc_path = "/backup/lzl/data/MMR/IHC/2025-03-05_15_04_40.svs"

output_dir = Path("../../test/simpleitk")
output_dir.mkdir(exist_ok=True)

he_img_gray_path = output_dir / "HE_level2_gray.png"
ihc_img_gray_path = output_dir / "IHC_level2_gray.png"
he_img_gray_match_path = output_dir / "HE_level2_gray_match.png"
ihc_img_gray_match_path = output_dir / "IHC_level2_gray_match.png"

segment_dir = output_dir / "segment"
# endregion

# region 图像预览
he_reader = WSIReader.open(he_path)
ihc_reader = WSIReader.open(ihc_path)

print_wsi_metadata(he_reader, "H&E")
print_wsi_metadata(ihc_reader, "IHC")

print("="*50)
print("图像数据预览")
print("="*50)

# 返回 numpy 数组，包含图像的像素数据
he_img_level2 = he_reader.read_region(
    location=(0, 0),
    size=he_reader.info.level_dimensions[registration_level],
    level=registration_level,
)
print("H&E 层级2 图像尺寸:", he_img_level2.shape)

ihc_img_level2 = ihc_reader.read_region(
    location=(0, 0),
    size=ihc_reader.info.level_dimensions[registration_level],
    level=registration_level,
)
print("IHC 层级2 图像尺寸:", ihc_img_level2.shape)

_, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(he_img_level2)
axs[0].set_title("H&E Level 2")
axs[1].imshow(ihc_img_level2)
axs[1].set_title("IHC Level 2")
plt.savefig(output_dir / "H&E_IHC_level2_visual.png", dpi=300, bbox_inches='tight')
plt.close()
# endregion

# region RGB图像转换为灰度图像
# 单通道灰度图像
he_img_gray_single = preprocess_image(he_img_level2)
ihc_img_gray_single = preprocess_image(ihc_img_level2)
he_img_gray = np.repeat(np.expand_dims(he_img_gray_single, axis=2), 3, axis=2)
ihc_img_gray = np.repeat(np.expand_dims(ihc_img_gray_single, axis=2), 3, axis=2)

cv2.imwrite(str(he_img_gray_path), he_img_gray)
cv2.imwrite(str(ihc_img_gray_path), ihc_img_gray)

_, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(he_img_gray_single, cmap="gray")
axs[0].set_title("H&E Level 2 (Gray)")
axs[1].imshow(ihc_img_gray_single, cmap="gray")
axs[1].set_title("IHC Level 2 (Gray)")
plt.savefig(output_dir / "H&E_IHC_level2_gray_visual.png", dpi=300, bbox_inches='tight')
plt.close()


# 可选：直方图匹配，但不要用于组织分割，分布变化会导致错误的匹配
# 使IHC图像的亮度分布与H&E图像一致
he_img_gray_single_match, ihc_img_gray_single_match = match_histograms(
    he_img_gray_single, 
    ihc_img_gray_single
)
he_img_gray_match = np.repeat(np.expand_dims(he_img_gray_single_match, axis=2), 3, axis=2)
ihc_img_gray_match = np.repeat(np.expand_dims(ihc_img_gray_single_match, axis=2), 3, axis=2)

cv2.imwrite(str(he_img_gray_match_path), he_img_gray_match)
cv2.imwrite(str(ihc_img_gray_match_path), ihc_img_gray_match)

_, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(he_img_gray_single_match, cmap="gray")
axs[0].set_title("H&E Level 2 (Gray)")
axs[1].imshow(ihc_img_gray_single_match, cmap="gray")
axs[1].set_title("IHC Level 2 (Gray)")
plt.savefig(output_dir / "H&E_IHC_level2_gray_match_visual.png", dpi=300, bbox_inches='tight')
plt.close()
# endregion

# region 组织分割
segmentor = SemanticSegmentor(
    pretrained_model="unet_tissue_mask_tsef",
    num_loader_workers=4,
    batch_size=4,
)

if segment_dir.exists() and segment_dir.is_dir():
    try:
        # rmtree 可删除非空目录（包含所有子文件/子目录），rmdir() 仅能删除空目录
        shutil.rmtree(segment_dir)
        print(f"目录 {segment_dir} 已成功删除")
    except Exception as e:
        print(f"删除目录 {segment_dir} 时出错: {e}")
else:
    print(f"目录 {segment_dir} 不存在，无需删除")

# 对存储的灰度图像进行组织分割
# 灰度图像是level 2的存储结果
# resolution=1.0, units="baseline" 表示以PNG的原始尺寸进行处理
segment_result = segmentor.predict(
    [
        str(he_img_gray_path),
        str(ihc_img_gray_path),
    ],
    save_dir=segment_dir,
    mode="tile",
    resolution=1.0,
    units="baseline",
    patch_input_shape=[1024, 1024],
    patch_output_shape=[512, 512],
    stride_shape=[512, 512],
    device="cuda",
    crash_on_exception=True,
)

# 加载Level 2的组织分割结果
# Shape: (H, W, 3) - 3个类别的概率图
# 类别0: 背景（玻片白色区域）
# 类别1: 边界/不确定区域
# 类别2: 组织（染色区域）
he_mask = np.load(segment_result[0][1] + ".raw.0.npy")
ihc_mask = np.load(segment_result[1][1] + ".raw.0.npy")

print("="*50)
print("组织分割原始输出形状")
print("="*50)

print("H&E 组织分割原始输出形状:", he_mask.shape)
print("IHC 组织分割原始输出形状:", ihc_mask.shape)

TISSUE_CLASS_INDEX = 2

# 从概率图转换为二值mask
he_mask = np.argmax(he_mask, axis=-1) == TISSUE_CLASS_INDEX
ihc_mask = np.argmax(ihc_mask, axis=-1) == TISSUE_CLASS_INDEX

# 对二值掩码进行清理，只保留最大的连通区域
he_mask = post_processing_mask(he_mask)
ihc_mask = post_processing_mask(ihc_mask)

_, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(he_mask, cmap="gray")
axs[0].set_title("HE Mask")
axs[1].imshow(ihc_mask, cmap="gray")
axs[1].set_title("IHC Mask")
plt.savefig(output_dir / "H&E_IHC_level2_mask_visual.png", dpi=300, bbox_inches='tight')
plt.close()
# endregion

# region DFBR 粗配准
dfbr = DFBRegister()

# 返回值是一个 3x3 的仿射变换矩阵
# [[a, b, tx],
# [c, d, ty],
# [0, 0, 1 ]]
# a, b, c, d: 控制旋转、缩放、剪切
# tx, ty: 控制平移
dfbr_transformer = dfbr.register(
    fixed_img=he_img_gray,
    moving_img=ihc_img_gray,
    fixed_mask=he_mask,
    moving_mask=ihc_mask,
)

# 原始 IHC 图像
original_ihc = cv2.warpAffine(
    ihc_img_gray_single,
    # 创建一个 2x3 的单位仿射变换矩阵，这意味着不进行任何变换
    # 只是将 IHC 图像调整到与 HE 图像相同的尺寸
    np.eye(2, 3),
    # [::-1]将 (H, W) 转换为 (W, H)，OpenCV 要求的输入格式
    he_img_gray.shape[:2][::-1],
)
# 配准后的 IHC 图像
dfbr_registered_ihc = cv2.warpAffine(
    ihc_img_gray_single,
    dfbr_transformer[0:-1],
    he_img_gray.shape[:2][::-1],
)
# 配准后的 IHC mask
dfbr_registered_ihc_mask = cv2.warpAffine(
    ihc_mask,
    dfbr_transformer[0:-1],  # 只取前 2x3 的部分，忽略最后一行 [0, 0, 1]
    he_img_gray.shape[:2][::-1],
)

# 将配准前的移动图像、固定图像和配准前的移动图像堆叠在一起，用于可视化配准前的叠加效果
before_overlay = np.dstack((original_ihc, he_img_gray_single, original_ihc))
# 将配准后的移动图像、固定图像和配准后的移动图像堆叠在一起，用于可视化配准后的叠加效果
dfbr_overlay = np.dstack((dfbr_registered_ihc, he_img_gray_single, dfbr_registered_ihc))

_, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(before_overlay, cmap="gray")
axs[0].set_title("Overlay Before Registration")
axs[1].imshow(dfbr_overlay, cmap="gray")
axs[1].set_title("Overlay After DFBR")
plt.savefig(output_dir / "H&E_IHC_level2_dfbr_visual.png", dpi=300, bbox_inches='tight')
plt.close()

# 应用变换矩阵到 RGB 图像
affine_matrix = dfbr_transformer[0:-1]
dfbr_registered_ihc_img_level2 = cv2.warpAffine(
    ihc_img_level2,
    affine_matrix,
    (he_img_level2.shape[1], he_img_level2.shape[0]),  # (width, height)
    flags=cv2.INTER_LINEAR,  # 线性插值，在变换过程中平滑地计算新像素值
    borderMode=cv2.BORDER_CONSTANT,  # 边界处理模式：变换后超出范围的区域用常量填充
    borderValue=(0, 0, 0)  # 填充值为黑色
)

_, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(he_img_level2)
axs[0].set_title("Fixed Image (RGB)")
axs[1].imshow(dfbr_registered_ihc_img_level2)
axs[1].set_title("Registered Moving Image (RGB)")
plt.savefig(output_dir / "H&E_IHC_level2_dfbr_rgb_visual.png", dpi=300, bbox_inches='tight')
plt.close()

# 在 level 0 进行仿射变换
he_img_level0 = he_reader.read_region(
    location=(0, 0),
    size=he_reader.info.level_dimensions[0],
    level=0,
)
ihc_img_level0 = ihc_reader.read_region(
    location=(0, 0),
    size=ihc_reader.info.level_dimensions[0],
    level=0,
)

ihc_img_registered_level0 = apply_affine_transformation(
    fixed_img=he_img_level0,
    moving_img=ihc_img_level0,
    transformer=dfbr_transformer,
)
# endregion

# region SimpleITK 非刚性配准
# endregion
