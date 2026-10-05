"""拓扑图预处理：去标注/引线/剖面线 -> 白底黑线方形控制图。

用法（坐标需逐图人工调整，先跑一次看 debug 叠框图再修正）:
  python prep_control.py input/topo.png --out output/control_clean.png \
      --erase-box 2,42,96,80 --erase-box 44,116,114,146 \
      --erase-line 114,134,150,150 \
      --hatch-zone 140,86,390,114 \
      --debug output/debug_boxes.png

坐标系约定：
  --erase-box / --erase-line / --margin-rect 用源图像素坐标；
  --hatch-zone / --final-white 用输出画布坐标（--size 边长的方形图）。
"""
import argparse
import os

import cv2
import numpy as np


def rects(arg):
    out = []
    for item in (arg or []):
        v = [int(x) for x in item.split(",")]
        if len(v) != 4:
            raise SystemExit(f"坐标需 4 个数字: {item}")
        out.append(tuple(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", help="叠框调试图输出路径")
    ap.add_argument("--erase-box", action="append", help="标注文字框（inpaint）")
    ap.add_argument("--erase-line", action="append", help="引线段走廊 x1,y1,x2,y2")
    ap.add_argument("--margin-rect", action="append", help="边缘空白带，内部孤立连通域抹白")
    ap.add_argument("--hatch-zone", action="append", help="剖面线清除区（画布坐标）")
    ap.add_argument("--final-white", action="append", help="画布坐标定点白块")
    ap.add_argument("--thresh", type=int, default=170)
    ap.add_argument("--size", type=int, default=512, help="输出方形画布边长")
    ap.add_argument("--hatch-k", type=int, default=7)
    args = ap.parse_args()

    im = cv2.imread(args.src)
    if im is None:
        raise SystemExit(f"读不到图: {args.src}")
    boxes = rects(args.erase_box)
    segs = rects(args.erase_line)
    margins = rects(args.margin_rect)
    hatches = rects(args.hatch_zone)
    whites = rects(args.final_white)

    # 1) inpaint 抹除标注文字与引线
    if boxes or segs:
        mask = np.zeros(im.shape[:2], np.uint8)
        for x1, y1, x2, y2 in boxes:
            mask[y1:y2, x1:x2] = 255
        for x1, y1, x2, y2 in segs:
            cv2.line(mask, (x1, y1), (x2, y2), 255, 5)
        im = cv2.inpaint(im, mask, 5, cv2.INPAINT_TELEA)

    # 2) 二值化
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, args.thresh, 255, cv2.THRESH_BINARY)

    # 3) 边缘空白带内的孤立连通域（标注/引线残余）抹白
    if margins:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(255 - bw, connectivity=8)
        for i in range(1, n):
            x, y, w2, h2, _a = stats[i]
            for rx1, ry1, rx2, ry2 in margins:
                if x >= rx1 and y >= ry1 and x + w2 <= rx2 and y + h2 <= ry2:
                    bw[y:y + h2, x:x + w2] = 255
                    break

    # 4) 方形画布居中 + 缩放
    h, w = bw.shape
    s = max(h, w)
    canvas = np.full((s, s), 255, np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = bw
    ctrl = cv2.resize(canvas, (args.size, args.size), interpolation=cv2.INTER_AREA)

    # 5) 画布坐标系：定点白块 + 剖面线清除（横/竖开运算保留结构线，抽掉斜线）
    for x1, y1, x2, y2 in whites:
        ctrl[y1:y2, x1:x2] = 255
    K = args.hatch_k
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (K, 1))
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, K))
    black = (ctrl < 128).astype(np.uint8) * 255
    for zx1, zy1, zx2, zy2 in hatches:
        roi = black[zy1:zy2, zx1:zx2]
        horiz = cv2.morphologyEx(roi, cv2.MORPH_OPEN, hker)
        vert = cv2.morphologyEx(roi, cv2.MORPH_OPEN, vker)
        keep = cv2.max(horiz, vert)
        ctrl[zy1:zy2, zx1:zx2] = np.where(keep > 0, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, ctrl)
    print("saved", args.out)

    if args.debug:
        dbg = im.copy()
        for b in boxes:
            cv2.rectangle(dbg, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
        for g in segs:
            cv2.line(dbg, (g[0], g[1]), (g[2], g[3]), (0, 0, 255), 1)
        cv2.imwrite(args.debug, dbg)
        print("saved", args.debug)


if __name__ == "__main__":
    main()
