"""设计线稿预处理：带浅灰填充的线稿 -> 纯白底黑线（渲染控制图）。

`to_bw_control` / `prep_iso_reference` 也供 generate_showcase.py 使用：把整体结构
等距示意图做成第二路几何控制图。示意图是控制图，不是风格提示。

用法:
  python prep_control.py sketch1.png sketch2.png --out-dir output/control --thresh 218

阈值取值：背景近纯白(240+)、线为灰(150-210)时用 218 左右；阈值过低会把浅灰
结构线当背景丢光。右缘尺寸标注等杂物可先在别处白掉（--erase-box 源图坐标）。
"""
import argparse
import os

import cv2
import numpy as np


def to_bw_control(img_bgr, thresh=218):
    """浅灰线稿 -> 白底黑线。与本脚本命令行同一阈值逻辑。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return cv2.medianBlur(bw, 3)


def edges_to_control(img_bgr):
    """彩色或填色等距图 -> 白底黑线。MistoLine 吃线稿，色块不能当结构。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    med = float(np.median(gray))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, max(lo + 1, 1.33 * med)))
    edges = cv2.Canny(gray, lo, hi)
    # 略加粗，细线缩到 1024 后仍能被控制网看见
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return 255 - edges


def auto_iso_control(img_bgr, thresh=218):
    """黑线等距/解剖图留深色笔划；浅灰线稿走阈值；其余抽边缘。

    返回 (白底黑线 uint8, 方法名)。方法名仅用于日志，方便核对几何有没有被涂实。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    # 低饱和深色才是墨线。纯红/纯蓝填充亮度也很低，不能当成结构涂黑。
    ink = (gray < 80) & (sat < 50)
    ink_ratio = float(ink.mean())
    if 0.002 <= ink_ratio <= 0.25:
        return np.where(ink, 0, 255).astype(np.uint8), "dark-stroke"
    if float(sat.mean()) < 18:
        bw = to_bw_control(img_bgr, thresh)
        if float((bw < 128).mean()) <= 0.45:
            return bw, "thresh"
    return edges_to_control(img_bgr), "edges"


def read_bgr_on_white(src):
    """读图并把透明底合成白底。HTML 导出的等距 PNG 常带 alpha，直接 imread 会变黑底。"""
    raw = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise SystemExit(f"读不到图: {src}")
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.shape[2] == 4:
        bgr = raw[:, :, :3].astype(np.float32)
        alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
        comp = bgr * alpha + 255.0 * (1.0 - alpha)
        return comp.astype(np.uint8)
    return raw[:, :, :3]


def _clip_box(box, width, height, flag):
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        raise SystemExit(
            f"{flag} {x1},{y1},{x2},{y2} 超出图像 {width}x{height} 或不是正矩形")
    return x1, y1, x2, y2


def prep_iso_reference(src, thresh=218, crop_box=None, erase_boxes=None, method="auto"):
    """等距结构示意图 -> 白底黑线控制图（未缩放，裁切在源图像素坐标）。

    erase_boxes 与 crop_box 都是源图坐标，先抹再裁。method: auto / thresh / edges / dark-stroke。
    返回 (uint8 单通道, 实际使用的方法名)。
    """
    img = read_bgr_on_white(src)
    height, width = img.shape[:2]
    for box in erase_boxes or []:
        x1, y1, x2, y2 = _clip_box(box, width, height, "--iso-erase-box")
        img[y1:y2, x1:x2] = 255
    if crop_box is not None:
        x1, y1, x2, y2 = _clip_box(crop_box, width, height, "--iso-box")
        img = img[y1:y2, x1:x2]
    if method == "thresh":
        return to_bw_control(img, thresh), "thresh"
    if method == "edges":
        return edges_to_control(img), "edges"
    if method == "dark-stroke":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.where(gray < 90, 0, 255).astype(np.uint8), "dark-stroke"
    if method == "auto":
        return auto_iso_control(img, thresh)
    raise SystemExit(f"未知等距预处理方法: {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+")
    ap.add_argument("--out-dir", default="output/control")
    ap.add_argument("--thresh", type=int, default=218)
    ap.add_argument("--erase-box", action="append", default=[],
                    help="源图坐标白块 x1,y1,x2,y2（抹标注，作用于全部输入）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for src in args.src:
        img = cv2.imread(src)
        if img is None:
            print("SKIP (unreadable):", src)
            continue
        for box in args.erase_box:
            x1, y1, x2, y2 = [int(v) for v in box.split(",")]
            img[y1:y2, x1:x2] = 255
        bw = to_bw_control(img, args.thresh)
        name = "ctrl_" + os.path.splitext(os.path.basename(src))[0] + ".png"
        dst = os.path.join(args.out_dir, name)
        cv2.imwrite(dst, bw)
        black = float((bw < 128).mean())
        print(f"saved {dst}  thresh={args.thresh}  黑像素占比 {black:.3f}")


if __name__ == "__main__":
    main()
