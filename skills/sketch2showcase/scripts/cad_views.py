"""cadquery 三视图投影引擎：从模型文件生成正交线稿（白底黑线 PNG）。

用法:
  python cad_views.py my_machine.py --out-dir output/cad --views front,side,top

模型文件约定（见 references 内说明或文件尾注释）：
  - 必须定义 build() -> cadquery 形体（Workplane / Shape / list 均可）
  - 单位 mm，Z 轴向上，原点在底面投影中心
  - 环境依赖：cadquery（建议独立 venv：pip install cadquery）

原理：cadquery 导出 SVG（OCCT HLR 隐藏线消除），再解析 SVG 线段栅格化为 PNG。
"""
import argparse
import importlib.util
import os
import re

import cadquery as cq
from cadquery import exporters
from PIL import Image, ImageDraw

MARGIN = 50


def load_build(model_path):
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build"):
        raise SystemExit("模型文件必须定义 build() 函数")
    return mod.build


def to_compound(result):
    if isinstance(result, cq.Workplane):
        return result.val()
    if isinstance(result, (list, tuple)):
        shapes = [x.val() if isinstance(x, cq.Workplane) else x for x in result]
        return cq.Compound.makeCompound(shapes)
    return result


def svg_to_png(svg_path, png_path, canvas):
    """CQ 导出的 SVG（无 viewBox，几何在 scale/translate 变换组内，M/L 线段）-> 方形 PNG。"""
    txt = open(svg_path, encoding="utf-8").read()
    m = re.search(r'scale\(([-\d.eE]+),\s*([-\d.eE]+)\)\s*translate\(([-\d.eE]+),\s*([-\d.eE]+)\)', txt)
    if m is None:
        raise ValueError("未找到 scale/translate 变换: " + svg_path)
    sx, sy, tx, ty = [float(v) for v in m.groups()]

    segs = []
    for d in re.findall(r'<path d="([^"]+)"', txt):
        cur = None
        for cmd, rest in re.findall(r'([ML])([^ML]*)', d):
            nums = [float(v) for v in re.findall(r'[-\d.eE]+', rest)]
            # 变换顺序：先 translate 后 scale（SVG 右向左作用）
            pts = [((nums[i] + tx) * sx, (nums[i + 1] + ty) * sy)
                   for i in range(0, len(nums), 2)]
            if cmd == "M":
                cur = pts[0]
                pts = pts[1:]
            for p in pts:
                segs.append((cur[0], cur[1], p[0], p[1]))
                cur = p

    if not segs:
        raise ValueError("SVG 无几何线段: " + svg_path)
    xs = [v for s in segs for v in (s[0], s[2])]
    ys = [v for s in segs for v in (s[1], s[3])]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    scale = (canvas - 2 * MARGIN) / max(x1 - x0, y1 - y0)
    # SVG y 向下，翻转并垂直居中（内容不满幅时避免沉底）
    voff = ((canvas - 2 * MARGIN) - (y1 - y0) * scale) / 2
    fx = lambda x: MARGIN + (x - x0) * scale  # noqa: E731
    fy = lambda y: MARGIN + voff + (y1 - y) * scale  # noqa: E731

    img = Image.new("RGB", (canvas, canvas), "white")
    draw = ImageDraw.Draw(img)
    for seg in segs:
        draw.line([fx(seg[0]), fy(seg[1]), fx(seg[2]), fy(seg[3])],
                  fill="black", width=2)
    img.save(png_path)
    print("saved", png_path)


# 投影方向语义坑：CQ SVG 导出仅在 projectionDir=(0,0,1) 时屏幕竖直才对应世界 Z，
# 其余方向会带 90° 旋转。因此固定用 (0,0,1)，靠投影前旋转模型摆正视图。
VIEW_ROTS = {
    "front": [((1, 0, 0), 90)],
    "side": [((0, 0, 1), -90), ((1, 0, 0), 90)],
    "top": [],
    "bottom": [((0, 0, 1), 180)],
    "back": [((0, 0, 1), 180), ((1, 0, 0), 90)],
    "left": [((0, 0, 1), 90), ((1, 0, 0), 90)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="模型文件（定义 build()）")
    ap.add_argument("--out-dir", default="output/cad")
    ap.add_argument("--views", default="front,side,top")
    ap.add_argument("--canvas", type=int, default=1200)
    ap.add_argument("--stroke", type=float, default=0.3, help="SVG 描边宽（mm）")
    args = ap.parse_args()

    build = load_build(args.model)
    comp0 = to_compound(build())
    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.views.split(","):
        if name not in VIEW_ROTS:
            raise SystemExit(f"未知视图 {name}，可选: {','.join(VIEW_ROTS)}")
        comp = comp0
        for axis, ang in VIEW_ROTS[name]:
            comp = comp.rotate((0, 0, 0), axis, ang)
        svg = os.path.join(args.out_dir, f"view_{name}.svg")
        png = os.path.join(args.out_dir, f"view_{name}.png")
        exporters.export(comp, svg,
                         opt={"projectionDir": (0, 0, 1), "showHidden": False,
                              "showAxes": False, "strokeWidth": args.stroke})
        svg_to_png(svg, png, args.canvas)
    print("ALL VIEWS EXPORTED")


if __name__ == "__main__":
    main()
