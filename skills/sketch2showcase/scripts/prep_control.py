"""设计线稿预处理：带浅灰填充的线稿 -> 纯白底黑线（渲染控制图）。

用法:
  python prep_control.py sketch1.png sketch2.png --out-dir output/control --thresh 218

阈值取值：背景近纯白(240+)、线为灰(150-210)时用 218 左右；阈值过低会把浅灰
结构线当背景丢光。右缘尺寸标注等杂物可先在别处白掉（--erase-box 源图坐标）。
"""
import argparse
import os

import cv2


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
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(g, args.thresh, 255, cv2.THRESH_BINARY)
        bw = cv2.medianBlur(bw, 3)
        name = "ctrl_" + os.path.splitext(os.path.basename(src))[0] + ".png"
        dst = os.path.join(args.out_dir, name)
        cv2.imwrite(dst, bw)
        black = float((bw < 128).mean())
        print(f"saved {dst}  thresh={args.thresh}  黑像素占比 {black:.3f}")


if __name__ == "__main__":
    main()
