"""
配准效果查看工具
支持多分辨率、区域选择、多种可视化模式
用法：
python registration_viewer.py \
    --slide_dir ../../test/valis_registration2/data \
    --static \
    --out_dir ../../test/valis_report


    python registration_viewer.py --slide_dir /path/to/registered_svs
    python registration_viewer.py --slide_dir /path/to/registered_svs --level 1 --x 1000 --y 1000 --w 512 --h 512
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons, RectangleSelector
from matplotlib.gridspec import GridSpec
import openslide
from skimage import exposure
from skimage.color import rgb2gray
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_slides(slide_dir):
    """加载目录下所有 SVS / ome.tiff 文件"""
    exts = (".svs", ".tiff", ".tif", ".ndpi", ".scn", ".mrxs")
    files = sorted([
        os.path.join(slide_dir, f)
        for f in os.listdir(slide_dir)
        if f.lower().endswith(exts)
    ])
    if not files:
        raise FileNotFoundError(f"在 {slide_dir} 下未找到切片文件")

    slides = []
    for f in files:
        try:
            s = openslide.OpenSlide(f)
            slides.append((os.path.basename(f), s))
            print(f"  已加载: {os.path.basename(f)}  层级数={s.level_count}  "
                  f"尺寸={s.dimensions}")
        except Exception as e:
            print(f"  [跳过] {os.path.basename(f)}: {e}")

    if not slides:
        raise RuntimeError("没有可读取的切片文件")
    return slides


def read_region(slide, level, x, y, w, h):
    """从切片读取指定区域，返回 numpy RGBA→RGB"""
    # x, y 是 level-0 坐标
    img = slide.read_region((x, y), level, (w, h))
    return np.array(img.convert("RGB"))


def get_thumbnail(slide, max_dim=512):
    """获取缩略图"""
    w, h = slide.dimensions
    scale = max_dim / max(w, h)
    tw, th = max(1, int(w * scale)), max(1, int(h * scale))
    thumb = slide.get_thumbnail((tw, th))
    return np.array(thumb.convert("RGB")), scale


def overlay_blend(imgs, alpha=None):
    """将多张 RGB 图混合为一张彩色叠加图"""
    n = len(imgs)
    if alpha is None:
        alpha = 1.0 / n

    # 每张图染不同颜色后叠加
    cmap_colors = plt.cm.get_cmap("tab10")
    result = np.zeros((*imgs[0].shape[:2], 3), dtype=np.float32)
    for i, img in enumerate(imgs):
        gray = rgb2gray(img.astype(np.float32) / 255.0)
        color = np.array(cmap_colors(i)[:3])
        colored = gray[..., np.newaxis] * color[np.newaxis, np.newaxis, :]
        result += colored * alpha

    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)


def checkerboard(img1, img2, grid=8):
    """棋盘格拼接两张图"""
    h, w = img1.shape[:2]
    result = img1.copy()
    cell_h = max(1, h // grid)
    cell_w = max(1, w // grid)
    for r in range(grid):
        for c in range(grid):
            if (r + c) % 2 == 0:
                r0, r1 = r * cell_h, min((r + 1) * cell_h, h)
                c0, c1 = c * cell_w, min((c + 1) * cell_w, w)
                result[r0:r1, c0:c1] = img2[r0:r1, c0:c1]
    return result


def difference_map(img1, img2):
    """差值图（绝对差）"""
    g1 = rgb2gray(img1.astype(np.float32) / 255.0)
    g2 = rgb2gray(img2.astype(np.float32) / 255.0)
    diff = np.abs(g1 - g2)
    return diff


# ─────────────────────────────────────────────
# 主查看器
# ─────────────────────────────────────────────

class RegistrationViewer:
    """
    交互式配准效果查看器

    操作说明：
      - 左图（缩略图）：点击选择 ROI 中心，拖拽可选取矩形区域
      - 右侧面板：显示各切片在选定区域的详细视图
      - Level 滑块：切换分辨率层级
      - 模式按钮：Single / Overlay / Checkerboard / Difference
      - Prev/Next 按钮：切换参考切片对（用于 Checkerboard / Diff）
    """

    def __init__(self, slides, init_level=0, init_x=None, init_y=None,
                 init_w=512, init_h=512):
        self.slides = slides          # list of (name, OpenSlide)
        self.n = len(slides)
        self.level = init_level
        self.region_w = init_w
        self.region_h = init_h
        self.mode = "single"          # single | overlay | checker | diff
        self.ref_pair = (0, 1)        # 用于 checker / diff 的两张图索引

        # 用 level-0 坐标记录 ROI 左上角
        ref_slide = slides[0][1]
        W0, H0 = ref_slide.dimensions
        self.roi_x = init_x if init_x is not None else W0 // 2 - init_w // 2
        self.roi_y = init_y if init_y is not None else H0 // 2 - init_h // 2

        self._build_ui()
        self._update_all()

    # ── 构建 UI ──────────────────────────────

    def _build_ui(self):
        ref_slide = self.slides[0][1]
        max_level = ref_slide.level_count - 1

        fig = plt.figure(figsize=(18, 10), facecolor="#1a1a2e")
        fig.canvas.manager.set_window_title("Registration Viewer")
        self.fig = fig

        # 主布局：左侧导航 + 右侧图像网格
        gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[1, 3],
                            left=0.03, right=0.97, top=0.93, bottom=0.12,
                            wspace=0.05)

        # 左侧：缩略图 + 控件
        gs_left = gs_outer[0, 0].subgridspec(4, 1, hspace=0.4,
                                              height_ratios=[4, 0.6, 0.6, 1.2])
        self.ax_thumb = fig.add_subplot(gs_left[0])

        # Level 滑块
        ax_level = fig.add_subplot(gs_left[1])
        ax_level.set_facecolor("#16213e")
        self.slider_level = Slider(ax_level, "Level", 0, max_level,
                                   valinit=self.level, valstep=1,
                                   color="#e94560", track_color="#0f3460")
        self.slider_level.label.set_color("white")
        self.slider_level.valtext.set_color("#e94560")
        self.slider_level.on_changed(self._on_level_change)

        # 区域大小滑块
        ax_size = fig.add_subplot(gs_left[2])
        ax_size.set_facecolor("#16213e")
        self.slider_size = Slider(ax_size, "ROI Size", 128, 2048,
                                  valinit=self.region_w, valstep=128,
                                  color="#0f3460", track_color="#16213e")
        self.slider_size.label.set_color("white")
        self.slider_size.valtext.set_color("#e94560")
        self.slider_size.on_changed(self._on_size_change)

        # 模式选择
        ax_radio = fig.add_subplot(gs_left[3])
        ax_radio.set_facecolor("#16213e")
        self.radio = RadioButtons(ax_radio,
                                  ("Single", "Overlay", "Checkerboard", "Difference"),
                                  activecolor="#e94560")
        for label in self.radio.labels:
            label.set_color("white")
            label.set_fontsize(8)
        self.radio.on_clicked(self._on_mode_change)

        # 右侧：图像网格
        cols = min(self.n, 4)
        rows = (self.n + cols - 1) // cols
        gs_right = gs_outer[0, 1].subgridspec(rows, cols, hspace=0.08, wspace=0.04)
        self.ax_imgs = []
        for i in range(self.n):
            r, c = divmod(i, cols)
            ax = fig.add_subplot(gs_right[r, c])
            ax.set_facecolor("#0f3460")
            self.ax_imgs.append(ax)

        # 如果格子多于切片，隐藏多余的
        for i in range(self.n, rows * cols):
            r, c = divmod(i, cols)
            try:
                ax = fig.add_subplot(gs_right[r, c])
                ax.set_visible(False)
            except Exception:
                pass

        # 底部按钮
        ax_btn_prev = fig.add_axes([0.40, 0.02, 0.08, 0.04])
        ax_btn_next = fig.add_axes([0.50, 0.02, 0.08, 0.04])
        ax_btn_save = fig.add_axes([0.62, 0.02, 0.10, 0.04])

        btn_style = dict(color="#16213e", hovercolor="#e94560")
        self.btn_prev = Button(ax_btn_prev, "◀ Prev Pair", **btn_style)
        self.btn_next = Button(ax_btn_next, "Next Pair ▶", **btn_style)
        self.btn_save = Button(ax_btn_save, "💾 Save PNG", **btn_style)

        for btn in [self.btn_prev, self.btn_next, self.btn_save]:
            btn.label.set_color("white")
            btn.label.set_fontsize(8)

        self.btn_prev.on_clicked(self._on_prev_pair)
        self.btn_next.on_clicked(self._on_next_pair)
        self.btn_save.on_clicked(self._on_save)

        # 标题
        self.title_text = fig.text(0.5, 0.97, "", ha="center", va="top",
                                   color="white", fontsize=11,
                                   fontfamily="monospace")

        # 状态栏
        self.status_text = fig.text(0.03, 0.02, "", ha="left", va="bottom",
                                    color="#aaaaaa", fontsize=8,
                                    fontfamily="monospace")

        # 缩略图点击 & 框选
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.rect_selector = RectangleSelector(
            self.ax_thumb, self._on_rect_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords="pixels",
            interactive=False,
            props=dict(edgecolor="#e94560", facecolor="none", linewidth=1.5)
        )

    # ── 事件处理 ─────────────────────────────

    def _on_level_change(self, val):
        self.level = int(val)
        self._update_all()

    def _on_size_change(self, val):
        self.region_w = int(val)
        self.region_h = int(val)
        self._update_all()

    def _on_mode_change(self, label):
        self.mode = label.lower()
        self._update_all()

    def _on_prev_pair(self, event):
        a, b = self.ref_pair
        b = (b - 1) % self.n
        if b == a:
            b = (b - 1) % self.n
        self.ref_pair = (a, b)
        self._update_all()

    def _on_next_pair(self, event):
        a, b = self.ref_pair
        b = (b + 1) % self.n
        if b == a:
            b = (b + 1) % self.n
        self.ref_pair = (a, b)
        self._update_all()

    def _on_click(self, event):
        if event.inaxes != self.ax_thumb:
            return
        if event.button != 1:
            return
        # 将缩略图坐标转回 level-0 坐标
        thumb_img, scale = get_thumbnail(self.slides[0][1])
        tx, ty = event.xdata, event.ydata
        if tx is None or ty is None:
            return
        lv0_x = int(tx / scale) - self.region_w // 2
        lv0_y = int(ty / scale) - self.region_h // 2
        self._set_roi(lv0_x, lv0_y)

    def _on_rect_select(self, eclick, erelease):
        """鼠标框选区域"""
        _, scale = get_thumbnail(self.slides[0][1])
        x1 = min(eclick.xdata, erelease.xdata)
        y1 = min(eclick.ydata, erelease.ydata)
        x2 = max(eclick.xdata, erelease.xdata)
        y2 = max(eclick.ydata, erelease.ydata)

        lv0_x = int(x1 / scale)
        lv0_y = int(y1 / scale)
        new_w = max(128, int((x2 - x1) / scale))
        new_h = max(128, int((y2 - y1) / scale))

        self.region_w = new_w
        self.region_h = new_h
        # 同步 slider
        self.slider_size.set_val(min(new_w, 2048))
        self._set_roi(lv0_x, lv0_y)

    def _on_save(self, event):
        fname = f"registration_L{self.level}_x{self.roi_x}_y{self.roi_y}.png"
        self.fig.savefig(fname, dpi=150, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())
        print(f"已保存: {fname}")
        self.status_text.set_text(f"已保存: {fname}")
        self.fig.canvas.draw_idle()

    # ── ROI 设置 ─────────────────────────────

    def _set_roi(self, x, y):
        ref_slide = self.slides[0][1]
        W0, H0 = ref_slide.dimensions
        self.roi_x = int(np.clip(x, 0, W0 - self.region_w))
        self.roi_y = int(np.clip(y, 0, H0 - self.region_h))
        self._update_all()

    # ── 更新绘图 ─────────────────────────────

    def _update_all(self):
        self._update_thumbnail()
        self._update_regions()
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_thumbnail(self):
        ax = self.ax_thumb
        ax.clear()
        ax.set_facecolor("#0f3460")

        thumb, scale = get_thumbnail(self.slides[0][1])
        ax.imshow(thumb, aspect="equal")
        ax.set_title(f"导航图  [{self.slides[0][0]}]",
                     color="white", fontsize=7, pad=3)
        ax.tick_params(colors="gray", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        # 绘制 ROI 矩形
        rx = self.roi_x * scale
        ry = self.roi_y * scale
        rw = self.region_w * scale
        rh = self.region_h * scale
        rect = patches.Rectangle((rx, ry), rw, rh,
                                  linewidth=1.5, edgecolor="#e94560",
                                  facecolor="none", linestyle="--")
        ax.add_patch(rect)

    def _update_regions(self):
        ref_slide = self.slides[0][1]
        lv_factor = int(ref_slide.level_downsamples[self.level])

        # level 下的坐标和尺寸
        lv_x = self.roi_x // lv_factor
        lv_y = self.roi_y // lv_factor
        lv_w = max(32, self.region_w // lv_factor)
        lv_h = max(32, self.region_h // lv_factor)

        # 读取所有切片该区域
        imgs = []
        for name, slide in self.slides:
            try:
                img = read_region(slide, self.level,
                                  self.roi_x, self.roi_y, lv_w, lv_h)
                imgs.append(img)
            except Exception as e:
                # 填充黑图
                imgs.append(np.zeros((lv_h, lv_w, 3), dtype=np.uint8))
                print(f"  读取 {name} 失败: {e}")

        mode = self.mode

        for i, ax in enumerate(self.ax_imgs):
            ax.clear()
            ax.set_facecolor("#0f3460")

            if mode == "single":
                ax.imshow(imgs[i], aspect="equal")
                ax.set_title(self.slides[i][0], color="white",
                             fontsize=7, pad=2)

            elif mode == "overlay":
                blended = overlay_blend(imgs)
                ax.imshow(blended, aspect="equal")
                ax.set_title("Overlay (all)", color="white", fontsize=7, pad=2)
                if i > 0:
                    ax.set_visible(False)
                    continue
                else:
                    ax.set_visible(True)

            elif mode == "checkerboard":
                a, b = self.ref_pair
                img_a = imgs[a] if a < len(imgs) else imgs[0]
                img_b = imgs[b] if b < len(imgs) else imgs[-1]
                # 统一大小
                h = min(img_a.shape[0], img_b.shape[0])
                w = min(img_a.shape[1], img_b.shape[1])
                ck = checkerboard(img_a[:h, :w], img_b[:h, :w])
                ax.imshow(ck, aspect="equal")
                pair_title = (f"Checker: {self.slides[a][0].split('.')[0]} /"
                              f" {self.slides[b][0].split('.')[0]}")
                ax.set_title(pair_title, color="white", fontsize=7, pad=2)
                if i > 0:
                    ax.set_visible(False)
                    continue
                else:
                    ax.set_visible(True)

            elif mode == "difference":
                a, b = self.ref_pair
                img_a = imgs[a] if a < len(imgs) else imgs[0]
                img_b = imgs[b] if b < len(imgs) else imgs[-1]
                h = min(img_a.shape[0], img_b.shape[0])
                w = min(img_a.shape[1], img_b.shape[1])
                diff = difference_map(img_a[:h, :w], img_b[:h, :w])
                im = ax.imshow(diff, cmap="hot", vmin=0, vmax=1, aspect="equal")
                pair_title = (f"Diff: {self.slides[a][0].split('.')[0]} vs"
                              f" {self.slides[b][0].split('.')[0]}")
                ax.set_title(pair_title, color="white", fontsize=7, pad=2)
                if i > 0:
                    ax.set_visible(False)
                    continue
                else:
                    ax.set_visible(True)
                    # colorbar
                    try:
                        self.fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
                    except Exception:
                        pass

            for spine in ax.spines.values():
                spine.set_edgecolor("#e94560" if i == 0 else "#334466")
                spine.set_linewidth(1.2 if i == 0 else 0.5)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            ax.set_visible(True)

        # overlay / checker / diff 只用第一个 ax，其他隐藏
        if mode in ("overlay", "checkerboard", "difference"):
            for ax in self.ax_imgs[1:]:
                ax.set_visible(False)
            self.ax_imgs[0].set_visible(True)
        else:
            for ax in self.ax_imgs:
                ax.set_visible(True)

    def _update_title(self):
        ref_slide = self.slides[0][1]
        ds = ref_slide.level_downsamples[self.level]
        W0, H0 = ref_slide.dimensions
        mpp = ref_slide.properties.get(openslide.PROPERTY_NAME_MPP_X, "?")

        title = (f"Level {self.level}  |  Downsample ×{ds:.0f}  |  "
                 f"ROI origin=({self.roi_x}, {self.roi_y})  "
                 f"size=({self.region_w}×{self.region_h}) px@L0  |  "
                 f"MPP={mpp} µm/px  |  Slides: {self.n}")
        self.title_text.set_text(title)

        status = (f"Mode: {self.mode.upper()}  |  "
                  f"Pair: {self.slides[self.ref_pair[0]][0].split('.')[0]} ↔ "
                  f"{self.slides[self.ref_pair[1]][0].split('.')[0]}  |  "
                  f"点击缩略图选择区域，拖拽框选 ROI")
        self.status_text.set_text(status)

    def show(self):
        plt.show()


# ─────────────────────────────────────────────
# 静态批量保存（无交互模式）
# ─────────────────────────────────────────────

def static_report(slides, levels, x, y, w, h, out_dir="."):
    """在指定层级下保存配准对比图，不弹窗"""
    os.makedirs(out_dir, exist_ok=True)
    modes = ["single", "overlay", "checkerboard", "difference"]
    n = len(slides)

    for level in levels:
        ref_slide = slides[0][1]
        ds = int(ref_slide.level_downsamples[level])
        lv_w = max(32, w // ds)
        lv_h = max(32, h // ds)

        imgs = []
        for name, slide in slides:
            try:
                img = read_region(slide, level, x, y, lv_w, lv_h)
            except Exception:
                img = np.zeros((lv_h, lv_w, 3), dtype=np.uint8)
            imgs.append((name, img))

        for mode in modes:
            cols = min(n, 4)
            rows = max(1, (n + cols - 1) // cols)

            if mode in ("overlay", "checkerboard", "difference"):
                fig, axes = plt.subplots(1, 1, figsize=(6, 6),
                                         facecolor="#1a1a2e")
                axes = [axes]
            else:
                fig, axes = plt.subplots(rows, cols,
                                         figsize=(5 * cols, 5 * rows),
                                         facecolor="#1a1a2e")
                axes = np.array(axes).flatten().tolist()

            fig.suptitle(f"Level {level} (×{ds})  |  Mode: {mode.upper()}  |  "
                         f"ROI ({x},{y}) {w}×{h}px",
                         color="white", fontsize=10, y=0.98)

            if mode == "single":
                for i, (name, img) in enumerate(imgs):
                    axes[i].imshow(img)
                    axes[i].set_title(name, color="white", fontsize=8)
                    axes[i].axis("off")
                for j in range(n, len(axes)):
                    axes[j].set_visible(False)

            elif mode == "overlay":
                blended = overlay_blend([im for _, im in imgs])
                axes[0].imshow(blended)
                axes[0].set_title("Overlay (all slides)", color="white", fontsize=9)
                axes[0].axis("off")

            elif mode == "checkerboard":
                img_a = imgs[0][1]
                img_b = imgs[1][1] if len(imgs) > 1 else imgs[0][1]
                h2 = min(img_a.shape[0], img_b.shape[0])
                w2 = min(img_a.shape[1], img_b.shape[1])
                ck = checkerboard(img_a[:h2, :w2], img_b[:h2, :w2])
                axes[0].imshow(ck)
                axes[0].set_title(f"Checkerboard: {imgs[0][0]} / {imgs[1][0]}",
                                  color="white", fontsize=8)
                axes[0].axis("off")

            elif mode == "difference":
                img_a = imgs[0][1]
                img_b = imgs[1][1] if len(imgs) > 1 else imgs[0][1]
                h2 = min(img_a.shape[0], img_b.shape[0])
                w2 = min(img_a.shape[1], img_b.shape[1])
                diff = difference_map(img_a[:h2, :w2], img_b[:h2, :w2])
                im = axes[0].imshow(diff, cmap="hot", vmin=0, vmax=1)
                axes[0].set_title(f"Difference: {imgs[0][0]} vs {imgs[1][0]}",
                                  color="white", fontsize=8)
                axes[0].axis("off")
                fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)

            for ax in axes:
                ax.set_facecolor("#0f3460")

            plt.tight_layout()
            out_f = os.path.join(out_dir,
                                 f"reg_L{level}_{mode}_x{x}_y{y}.png")
            fig.savefig(out_f, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  保存: {out_f}")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="配准效果查看工具")
    p.add_argument("--slide_dir", required=True,
                   help="配准后 SVS / ome.tiff 所在目录")
    p.add_argument("--level", type=int, default=0,
                   help="初始分辨率层级（0=最高分辨率）")
    p.add_argument("--x", type=int, default=None,
                   help="ROI 左上角 X 坐标（level-0）")
    p.add_argument("--y", type=int, default=None,
                   help="ROI 左上角 Y 坐标（level-0）")
    p.add_argument("--w", type=int, default=512,
                   help="ROI 宽度（level-0 像素，默认 512）")
    p.add_argument("--h", type=int, default=512,
                   help="ROI 高度（level-0 像素，默认 512）")
    p.add_argument("--static", action="store_true",
                   help="非交互模式：批量保存所有层级的对比图")
    p.add_argument("--out_dir", default="./reg_report",
                   help="静态模式下的输出目录")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n加载切片目录: {args.slide_dir}")
    slides = load_slides(args.slide_dir)
    print(f"共加载 {len(slides)} 个切片\n")

    if args.static:
        # 非交互：保存所有层级
        ref_slide = slides[0][1]
        all_levels = list(range(ref_slide.level_count))
        W0, H0 = ref_slide.dimensions
        x = args.x if args.x is not None else W0 // 2 - args.w // 2
        y = args.y if args.y is not None else H0 // 2 - args.h // 2
        print(f"静态报告模式，输出到: {args.out_dir}")
        static_report(slides, all_levels, x, y, args.w, args.h, args.out_dir)
        print("完成！")
    else:
        # 交互模式
        print("启动交互查看器...")
        print("  - 点击左侧缩略图选择区域")
        print("  - 拖拽框选精确 ROI")
        print("  - 右侧面板切换模式和分辨率")
        viewer = RegistrationViewer(
            slides,
            init_level=args.level,
            init_x=args.x,
            init_y=args.y,
            init_w=args.w,
            init_h=args.h,
        )
        viewer.show()


if __name__ == "__main__":
    main()
