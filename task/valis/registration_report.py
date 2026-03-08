"""
配准效果静态报告生成器
生成类似参考图的可视化：
  - 顶部：所有切片全图缩略图
  - 中部/底部：同一 ROI 在不同分辨率(level)下的 Patch 对比

用法：
# 自动在组织中心取 ROI，导出默认分辨率层级
python registration_report.py \
    --slide_dir ../../test/valis_registration2/data \
    --out_dir ../../test/valis_report

# 指定 ROI 坐标和多个层级
python registration_report.py --slide_dir /path/to/svs \
    --x 12000 --y 8000 --w 2048 --h 2048 \
    --levels 0 1 2 \
    --out_dir ./report

# 多个 ROI（逗号分隔）
python registration_report.py --slide_dir /path/to/svs \
    --rois "12000,8000,2048,2048" "5000,3000,1024,1024"
"""
import os
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import openslide
from skimage.color import rgb2gray
from skimage import filters

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# 颜色方案
# ──────────────────────────────────────────────────────────────
THEME = {
    "bg":       "#f7f7f5",
    "panel_bg": "#ffffff",
    "title_fg": "#1a1a2e",
    "label_fg": "#333355",
    "border":   "#ccccdd",
    "accent":   "#c0392b",
}

FIXED_COLOR   = "#2c3e50"
MOVING_COLORS = ["#8e44ad", "#27ae60", "#2980b9", "#d35400", "#16a085"]


# ──────────────────────────────────────────────────────────────
# 优化 1：缩略图缓存
# key = (slide_path, max_dim)，同一张图只读一次
# ──────────────────────────────────────────────────────────────
_thumb_cache: dict = {}

def get_thumbnail_cached(slide_path: str, max_dim: int = 512):
    """读取缩略图并缓存，相同 path+max_dim 只读一次"""
    key = (slide_path, max_dim)
    if key not in _thumb_cache:
        slide = openslide.OpenSlide(slide_path)
        w, h = slide.dimensions
        scale = max_dim / max(w, h)
        tw = max(1, int(w * scale))
        th = max(1, int(h * scale))
        thumb = slide.get_thumbnail((tw, th))
        _thumb_cache[key] = (np.array(thumb.convert("RGB")), scale)
    return _thumb_cache[key]


# ──────────────────────────────────────────────────────────────
# 优化 2：直接读取 native level，避免 level-0 解码
# ──────────────────────────────────────────────────────────────
def read_region_native(slide_path: str, x: int, y: int,
                       w: int, h: int, level: int) -> np.ndarray:
    """
    在指定 level 直接读取 native pyramid tile。

    参数 x, y, w, h 均为 level-0 坐标/尺寸。
    openslide.read_region 的 location=(x,y) 永远是 level-0 坐标，
    size=(lw, lh) 是目标 level 下的像素数，
    openslide 内部直接从对应 level 的 tile 读取，
    不会从 level-0 解码再缩放。
    """
    slide = openslide.OpenSlide(slide_path)   # 每次新建，线程安全
    ds = slide.level_downsamples[level]
    lw = max(1, round(w / ds))
    lh = max(1, round(h / ds))
    img = slide.read_region((x, y), level, (lw, lh))
    return np.array(img.convert("RGB"))


# ──────────────────────────────────────────────────────────────
# 优化 3：并行预取所有 patch
# ──────────────────────────────────────────────────────────────
def prefetch_patches(slide_paths: list, roi_x: int, roi_y: int,
                     roi_w: int, roi_h: int, levels: list,
                     max_workers: int = 8) -> dict:
    """
    并行读取 len(slides) × len(levels) 个 patch。
    返回 dict: {(slide_idx, level): np.ndarray}
    """
    tasks = [
        (i, path, level)
        for i, path in enumerate(slide_paths)
        for level in levels
    ]

    results = {}

    def _read(args):
        i, path, level = args
        arr = read_region_native(path, roi_x, roi_y, roi_w, roi_h, level)
        return (i, level), arr

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_read, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                key, arr = future.result()
                results[key] = arr
            except Exception as e:
                t = futures[future]
                print(f"  ✗ 读取失败 slide={t[0]} level={t[2]}: {e}")
                results[(t[0], t[2])] = None

    return results


# ──────────────────────────────────────────────────────────────
# 其他工具函数
# ──────────────────────────────────────────────────────────────

def load_slides(slide_dir: str) -> list:
    exts = (".svs", ".tiff", ".tif", ".ndpi", ".scn", ".mrxs")
    files = sorted([
        os.path.join(slide_dir, f)
        for f in os.listdir(slide_dir)
        if f.lower().endswith(exts)
    ])
    if not files:
        raise FileNotFoundError(f"未在 {slide_dir} 找到切片文件")

    slides = []
    for f in files:
        try:
            s = openslide.OpenSlide(f)
            slides.append({"name": os.path.basename(f), "path": f, "slide": s})
            print(f"  ✓ {os.path.basename(f)}  "
                  f"levels={s.level_count}  dim={s.dimensions}")
        except Exception as e:
            print(f"  ✗ 跳过 {os.path.basename(f)}: {e}")

    if not slides:
        raise RuntimeError("没有可读取的切片")
    return slides


def auto_roi(slide: openslide.OpenSlide, target_size: int = 2048):
    """在组织密度最高的区域自动选取 ROI 左上角坐标"""
    thumb, scale = get_thumbnail_cached(
        slide.get_path() if hasattr(slide, "get_path") else "",
        max_dim=256
    )
    # fallback：直接从 slide 读缩略图
    w0, h0 = slide.dimensions
    s = 256 / max(w0, h0)
    t = slide.get_thumbnail((max(1, int(w0 * s)), max(1, int(h0 * s))))
    thumb = np.array(t.convert("RGB"))
    scale = s

    gray = rgb2gray(thumb)
    tissue = gray < filters.threshold_otsu(gray)
    ys, xs = np.where(tissue)
    cx = int(xs.mean()) if len(xs) else thumb.shape[1] // 2
    cy = int(ys.mean()) if len(ys) else thumb.shape[0] // 2

    lv0_cx = int(cx / scale)
    lv0_cy = int(cy / scale)
    W0, H0 = slide.dimensions
    x = int(np.clip(lv0_cx - target_size // 2, 0, max(0, W0 - target_size)))
    y = int(np.clip(lv0_cy - target_size // 2, 0, max(0, H0 - target_size)))
    return x, y


def level_to_magnification(slide: openslide.OpenSlide, level: int):
    try:
        obj_power = float(slide.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
        return obj_power / slide.level_downsamples[level]
    except Exception:
        return None


def get_mpp(slide: openslide.OpenSlide) -> float:
    try:
        return float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0))
    except Exception:
        return 0.0


def short_name(name: str, max_len: int = 28) -> str:
    stem = os.path.splitext(name)[0]
    return stem if len(stem) <= max_len else stem[:max_len - 3] + "..."


def auto_levels(slide: openslide.OpenSlide, max_levels: int = 4) -> list:
    n = slide.level_count
    if n <= max_levels:
        return list(range(n))
    step = max(1, (n - 1) // (max_levels - 1))
    lvs = sorted(set([0] + list(range(step, n - 1, step)) + [n - 1]))
    return lvs[:max_levels]


# ──────────────────────────────────────────────────────────────
# 绘图：一张完整报告
# ──────────────────────────────────────────────────────────────

def make_report(slides: list, roi_x: int, roi_y: int,
                roi_w: int, roi_h: int, levels: list,
                out_path: str, title: str = "Registration Report",
                thumbnail_max: int = 512, max_workers: int = 8):
    """
    生成一张完整的配准报告图。

    布局：
      Row 0       : 所有切片全图缩略图（N 列）
      Row 1..N_lv : 每个 level 一行，每列一个切片的 Patch
    """
    n_slides = len(slides)
    n_levels = len(levels)
    slide_paths = [info["path"] for info in slides]

    # ── 优化：并行预取所有 patch ──────────────────────────────
    print(f"  并行读取 {n_slides} × {n_levels} 个 patch（workers={max_workers}）...")
    patches = prefetch_patches(
        slide_paths, roi_x, roi_y, roi_w, roi_h, levels,
        max_workers=max_workers
    )

    # ── 优化：预取缩略图（已缓存，重复调用无额外 IO）────────────
    thumbs = {}
    for i, info in enumerate(slides):
        thumbs[i] = get_thumbnail_cached(info["path"], max_dim=thumbnail_max)

    # ── 计算图尺寸 ────────────────────────────────────────────
    thumb_h = 3.2
    patch_h = 3.8
    col_w   = 3.5
    fig_w = col_w * n_slides + 0.6
    fig_h = thumb_h + patch_h * n_levels + 1.2

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=130,
                     facecolor=THEME["bg"])

    height_ratios = [thumb_h] + [patch_h] * n_levels
    gs = gridspec.GridSpec(
        1 + n_levels, n_slides,
        figure=fig,
        hspace=0.05, wspace=0.04,
        left=0.03, right=0.97,
        top=0.93, bottom=0.06,
        height_ratios=height_ratios,
    )

    ref_slide = slides[0]["slide"]

    # ── 顶部：全图缩略图 ──────────────────────────────────────
    for col, info in enumerate(slides):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(THEME["panel_bg"])

        thumb, scale = thumbs[col]
        ax.imshow(thumb, aspect="equal")

        # ROI 框
        rect = Rectangle(
            (roi_x * scale, roi_y * scale),
            roi_w * scale, roi_h * scale,
            linewidth=1.8, edgecolor=THEME["accent"],
            facecolor="none", linestyle="--", zorder=5
        )
        ax.add_patch(rect)

        mpp  = get_mpp(info["slide"])
        sub  = f"MPP={mpp:.3f} µm" if mpp > 0 else ""
        color = FIXED_COLOR if col == 0 else MOVING_COLORS[(col - 1) % len(MOVING_COLORS)]
        prefix = "Fixed" if col == 0 else f"Moving [{col}]"
        ax.set_title(f"{prefix}\n{short_name(info['name'])}\n{sub}",
                     fontsize=7.5, color=color, pad=4,
                     fontweight="bold" if col == 0 else "normal")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_edgecolor(THEME["border"])
            sp.set_linewidth(0.8)

    # ── Patch 行：每个 level ──────────────────────────────────
    for row_idx, level in enumerate(levels):
        ds  = ref_slide.level_downsamples[level]
        mag = level_to_magnification(ref_slide, level)
        mag_str = f"×{mag:.1f}" if mag else f"Level {level}"
        mpp_lv  = get_mpp(ref_slide) * ds if get_mpp(ref_slide) > 0 else 0

        row_label = (f"Level {level}  {mag_str}\n"
                     f"ds=×{ds:.0f}"
                     + (f"  MPP={mpp_lv:.2f}" if mpp_lv > 0 else ""))

        for col, info in enumerate(slides):
            ax = fig.add_subplot(gs[row_idx + 1, col])
            ax.set_facecolor(THEME["panel_bg"])

            patch = patches.get((col, level))
            if patch is not None:
                ax.imshow(patch, aspect="equal", interpolation="lanczos")
            else:
                ax.text(0.5, 0.5, "读取失败",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="red")
                ax.set_facecolor("#fff0f0")

            color  = FIXED_COLOR if col == 0 else MOVING_COLORS[(col - 1) % len(MOVING_COLORS)]
            prefix = "Fixed Tile" if col == 0 else "Moving Tile"
            ax.set_title(prefix, fontsize=7.5, color=color, pad=3)

            if col == 0:
                ax.set_ylabel(row_label, fontsize=6.5,
                              color=THEME["label_fg"], labelpad=4,
                              rotation=90, va="center")

            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_edgecolor(THEME["border"])
                sp.set_linewidth(0.6)

    # ── 标题 & 页脚 ───────────────────────────────────────────
    fig.text(0.5, 0.975, title,
             ha="center", va="top", fontsize=11, fontweight="bold",
             color=THEME["title_fg"])

    fig.text(0.5, 0.012,
             f"ROI origin=({roi_x},{roi_y})  size={roi_w}×{roi_h} px (level-0)  |  "
             f"Levels: {levels}  |  Slides: {n_slides}",
             ha="center", va="bottom", fontsize=6.5,
             color="#888899", style="italic")

    # ── 保存 ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ 已保存: {out_path}")


# ──────────────────────────────────────────────────────────────
# 批量：多个 ROI
# ──────────────────────────────────────────────────────────────

def batch_report(slides: list, rois: list, levels: list,
                 out_dir: str, prefix: str = "reg",
                 max_workers: int = 8):
    os.makedirs(out_dir, exist_ok=True)

    for i, (x, y, w, h) in enumerate(rois):
        out_f = os.path.join(out_dir, f"{prefix}_roi{i:02d}_x{x}_y{y}.png")
        title = f"Registration Report  |  ROI #{i}  ({x},{y})  {w}×{h} px"
        print(f"\n生成 ROI #{i}  ({x},{y}) {w}×{h} ...")
        make_report(slides, x, y, w, h,
                    levels=levels,
                    out_path=out_f,
                    title=title,
                    max_workers=max_workers)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="配准效果静态报告生成器（优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--slide_dir", required=True,
                   help="配准后切片所在目录")
    p.add_argument("--out_dir", default="./reg_report",
                   help="报告输出目录（默认 ./reg_report）")
    p.add_argument("--prefix", default="reg",
                   help="输出文件名前缀")

    # 单个 ROI
    p.add_argument("--x", type=int, default=None)
    p.add_argument("--y", type=int, default=None)
    p.add_argument("--w", type=int, default=2048)
    p.add_argument("--h", type=int, default=2048)

    # 多个 ROI
    p.add_argument("--rois", nargs="+", default=None,
                   help='多 ROI，格式 "x,y,w,h"，例: --rois "1000,2000,1024,1024"')

    # 分辨率层级
    p.add_argument("--levels", nargs="+", type=int, default=None,
                   help="展示的层级，例 --levels 0 1 2（默认自动选取最多4个）")

    p.add_argument("--thumbnail_max", type=int, default=512,
                   help="缩略图最大边长（像素，默认512）")
    p.add_argument("--n_auto_rois", type=int, default=1,
                   help="自动选取 ROI 数量（未指定 --x/--y/--rois 时生效）")
    p.add_argument("--workers", type=int, default=8,
                   help="并行读取线程数（默认8）")

    return p.parse_args()


def main():
    args = parse_args()

    print("\n═══ 配准报告生成器（优化版）═══")
    print(f"切片目录: {args.slide_dir}")

    slides = load_slides(args.slide_dir)
    ref_slide = slides[0]["slide"]
    W0, H0 = ref_slide.dimensions
    print(f"参考切片尺寸 (level-0): {W0} × {H0}")

    # ── 层级 ─────────────────────────────────────────────────
    levels = args.levels if args.levels else auto_levels(ref_slide)
    print(f"展示层级: {levels}  (共 {ref_slide.level_count} 个层级)")

    # ── ROI ──────────────────────────────────────────────────
    if args.rois:
        rois = []
        for s in args.rois:
            parts = [int(v.strip()) for v in s.split(",")]
            assert len(parts) == 4, f"ROI 格式错误: {s}"
            rois.append(tuple(parts))
    elif args.x is not None and args.y is not None:
        rois = [(args.x, args.y, args.w, args.h)]
    else:
        print(f"未指定 ROI，自动选取 {args.n_auto_rois} 个...")
        rois = []
        for _ in range(args.n_auto_rois):
            x, y = auto_roi(ref_slide, target_size=args.w)
            rois.append((x, y, args.w, args.h))
            print(f"  自动 ROI: ({x}, {y})  {args.w}×{args.h}")

    print(f"\n输出目录: {args.out_dir}")
    print(f"ROI 数量: {len(rois)}  |  并行线程: {args.workers}\n")

    batch_report(
        slides=slides,
        rois=rois,
        levels=levels,
        out_dir=args.out_dir,
        prefix=args.prefix,
        max_workers=args.workers,
    )

    print(f"\n完成！共生成 {len(rois)} 张报告图。")


if __name__ == "__main__":
    main()
