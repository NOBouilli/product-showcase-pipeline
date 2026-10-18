"""线稿控制图 -> 产品展示图（SDXL RealVisXL + MistoLine 线稿控制网）。

用法:
  # ortho（默认）：只有主线稿，和旧版相同
  python generate_showcase.py ctrl_front.png --product "hydrostatic testing machine \
      for fire extinguisher cylinders" --view "front view" --seeds 101,202

  # overall：主线稿 + 整体结构等距示意图，双几何控制，出整体实物渲染
  python generate_showcase.py ctrl_front.png --iso-ref iso_overview.png --mode overall \
      --product "hydrostatic testing machine for fire extinguisher cylinders" \
      --view "isometric three-quarter view" --seeds 101

  # core：等距示意图可按框裁到核心区，再加线稿（也可 --control-box 裁线稿）
  python generate_showcase.py ctrl_front.png --iso-ref iso_overview.png --mode core \
      --iso-box 120,80,400,360 --product "hydrostatic testing machine" \
      --view "close-up of the core assembly" --seeds 101

  python generate_showcase.py ctrl_exploded.png --style clay --view "exploded view"

--mode ortho    仅主线稿。输出 <out-dir>/<控制图名>_<style>_<seed>.png
--mode overall  必须 --iso-ref。输出 <控制图名>_overall_<style>_<seed>.png
--mode core     必须 --iso-ref。输出 <控制图名>_core_<style>_<seed>.png

等距示意图会先做成白底黑线，再和主线稿一起进 MultiControlNet（同一份 MistoLine
权重走两路）。几何只来自控制图；提示词只写材质、灯光和取景，不写零件清单。
SDXL 原生 1024。RTX 5070 Ti 单图约 1 分钟（模型加载约 1.5 分钟）。
"""
import argparse
import os

from PIL import Image, ImageOps

STYLES = {
    "render": {
        "pos": ("industrial product photography of {product}, "
                "{view}, studio softbox lighting, seamless light gray background, "
                "photorealistic, sharp focus, high detail, stainless steel and matte plastic "
                "materials, unbranded, plain blank surfaces, professional product render"),
        "neg": ("sketch, line drawing, drawing, cartoon, illustration, painting, "
                "(dark background:1.3), cluttered, (text:1.4), letters, numbers, "
                "(logo:1.4), (sticker:1.3), (nameplate:1.4), (label:1.3), brand, emblem, "
                "watermark, signature, blurry, deformed, lowres"),
    },
    "clay": {
        "pos": ("3d clay render of {product}, {view}, "
                "matte white ceramic material, soft studio lighting, light gray gradient "
                "background, industrial design concept model, octane render, high detail"),
        "neg": ("sketch, line drawing, cartoon, illustration, photo texture, text, "
                "logo, sticker, nameplate, label, watermark, "
                "dark background, blurry, deformed, lowres"),
    },
}

# 只约束取景，不点名零件。结构交给控制图。
MODE_POS = {
    "ortho": "",
    "overall": ", full assembled product, entire machine in frame",
    "core": ", close-up of the core assembly region shown in the control drawings",
}
MODE_NEG = {
    "ortho": "",
    "overall": ", extra components, missing parts, redesigned geometry, altered layout",
    "core": ", extra components, missing parts, redesigned geometry, wide establishing shot",
}
# 等距图负责布局，默认比线稿略强；core 两路接近，避免裁切区被线稿或等距单方拽走。
DEFAULT_ISO_SCALE = {"overall": 0.85, "core": 0.75}


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("control", help="线稿控制图（白底黑线）")
    ap.add_argument("--product", required=True, help="产品名（英文提示词，进渲染提示）")
    ap.add_argument("--style", default="render", choices=STYLES)
    ap.add_argument("--view", default="three-quarter view",
                    help='视图提示词，如 "front view" / "side view" / "top view" / "exploded view"')
    ap.add_argument("--mode", default="ortho", choices=("ortho", "overall", "core"),
                    help="ortho=仅主线稿（默认，兼容旧行为）；"
                         "overall=整机实物渲染，必须 --iso-ref；"
                         "core=核心结构实物渲染，必须 --iso-ref，可用 --iso-box")
    ap.add_argument("--iso-ref", default=None,
                    help="整体结构等距示意图。overall/core 必填；先转白底黑线，作第二路几何控制，不是风格参考")
    ap.add_argument("--iso-box", default=None,
                    help="等距示意图源图像素裁切 x1,y1,x2,y2。core 用来圈核心区；overall 可用来去白边")
    ap.add_argument("--iso-erase-box", action="append", default=[],
                    help="等距示意图源图坐标白块 x1,y1,x2,y2（抹标注，可重复；先于 --iso-box）")
    ap.add_argument("--iso-thresh", type=int, default=218,
                    help="等距图走阈值法时的灰度阈值，含义同 prep_control.py")
    ap.add_argument("--iso-method", default="auto",
                    choices=("auto", "thresh", "edges", "dark-stroke"),
                    help="等距图转线稿：auto 按图像选；thresh 浅灰线；dark-stroke 黑线；edges 彩色填色图")
    ap.add_argument("--iso-scale", type=float, default=None,
                    help="等距控制 conditioning scale。默认 overall=0.85，core=0.75。线稿仍用 --scale")
    ap.add_argument("--control-box", default=None,
                    help="主线稿源图像素裁切 x1,y1,x2,y2。core 可复用细节区；不传则整张线稿进控制网")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out-dir", default="output/showcase")
    ap.add_argument("--seeds", default="101,202,303")
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--scale", type=float, default=0.7,
                    help="主线稿 controlnet_conditioning_scale，MistoLine 建议 0.6-0.9")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--size", type=int, default=1024)
    return ap


def parse_box(text, flag):
    try:
        vals = [int(v) for v in text.split(",")]
    except ValueError:
        raise SystemExit(f"{flag} 需要整数坐标 x1,y1,x2,y2，收到: {text}")
    if len(vals) != 4:
        raise SystemExit(f"{flag} 需要 4 个整数 x1,y1,x2,y2，收到: {text}")
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise SystemExit(f"{flag} 需要 x2>x1 且 y2>y1，收到: {text}")
    return x1, y1, x2, y2


def validate_args(args):
    """路径和模式约束。不过模型、不读图像素，方便 --help 之外的参数错误立刻失败。"""
    if not os.path.isfile(args.control):
        raise SystemExit(f"找不到控制图: {args.control}")
    iso_flags = bool(args.iso_ref or args.iso_box or args.iso_erase_box or args.control_box
                     or args.iso_scale is not None or args.iso_method != "auto"
                     or args.iso_thresh != 218)
    if args.mode == "ortho":
        if iso_flags:
            raise SystemExit(
                "mode=ortho 不使用等距示意图。去掉 --iso-ref/--iso-box/--control-box 等参数，"
                "或改成 --mode overall / --mode core")
        return
    if not args.iso_ref:
        raise SystemExit(f"mode={args.mode} 必须提供 --iso-ref（整体结构等距示意图）")
    if not os.path.isfile(args.iso_ref):
        raise SystemExit(f"找不到等距示意图: {args.iso_ref}")
    if args.iso_box:
        parse_box(args.iso_box, "--iso-box")
    if args.control_box:
        parse_box(args.control_box, "--control-box")
    for box in args.iso_erase_box:
        parse_box(box, "--iso-erase-box")


def render_filename(stem, mode, style, seed):
    if mode == "ortho":
        return f"{stem}_{style}_{seed}.png"
    return f"{stem}_{mode}_{style}_{seed}.png"


def build_prompts(args):
    preset = STYLES[args.style]
    pos = preset["pos"].format(product=args.product, view=args.view) + MODE_POS[args.mode]
    neg = preset["neg"] + MODE_NEG[args.mode]
    return pos, neg


def fit_square(img, size):
    """等比放进方形白底。拉伸会改等距比例，整体/核心渲染不用拉伸。"""
    img = img.convert("RGB")
    width, height = img.size
    scale = size / float(max(width, height))
    nw = max(1, int(round(width * scale)))
    nh = max(1, int(round(height * scale)))
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def crop_pil(img, box, flag):
    x1, y1, x2, y2 = box
    width, height = img.size
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise SystemExit(f"{flag} {x1},{y1},{x2},{y2} 超出图像 {width}x{height}")
    return img.crop((x1, y1, x2, y2))


def load_pipeline(models_dir, multi):
    """16GB 内存/显存：controlnet device_map=cuda，管线 device_map=balanced。

    不能 .to("cuda") 前在 CPU 上攒齐 SDXL。overall/core 第二路复用同一 MistoLine，
    不再加载第二套 ControlNet 权重。
    """
    import torch
    from torch import nn
    from diffusers import (
        ControlNetModel, MultiControlNetModel, StableDiffusionXLControlNetPipeline,
        UniPCMultistepScheduler)

    class TiedControlNet(nn.Module):
        """第二路控制只转发到已加载的 MistoLine。list 引用不会把权重再注册一次。"""

        def __init__(self, net):
            super().__init__()
            self._net_ref = [net]

        def forward(self, *args, **kwargs):
            return self._net_ref[0](*args, **kwargs)

    mistoline = ControlNetModel.from_pretrained(
        os.path.join(models_dir, "mistoline"), torch_dtype=torch.float16,
        variant="fp16", low_cpu_mem_usage=True, device_map="cuda")
    if multi:
        # 同一份线稿控制网吃两张图：主线稿 + 等距结构。不要 from_pretrained 第二次。
        controlnet = MultiControlNetModel([mistoline, TiedControlNet(mistoline)])
    else:
        controlnet = mistoline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        os.path.join(models_dir, "realvisxl_v5"), controlnet=controlnet,
        torch_dtype=torch.float16, variant="fp16",
        safety_checker=None, requires_safety_checker=False,
        image_encoder=None, feature_extractor=None, low_cpu_mem_usage=True,
        device_map="balanced")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def prepare_ortho_control(path, size):
    from diffusers.utils import load_image
    return load_image(path).convert("RGB").resize((size, size))


def prepare_dual_controls(args):
    """主线稿（可裁）+ 等距示意图（白底黑线，可裁），都等比放进方形画布。"""
    from prep_control import prep_iso_reference

    line = ImageOps.exif_transpose(Image.open(args.control)).convert("RGB")
    if args.control_box:
        line = crop_pil(line, parse_box(args.control_box, "--control-box"), "--control-box")
    line = fit_square(line, args.size)

    iso_box = parse_box(args.iso_box, "--iso-box") if args.iso_box else None
    erase = [parse_box(box, "--iso-erase-box") for box in args.iso_erase_box]
    bw, method = prep_iso_reference(
        args.iso_ref, thresh=args.iso_thresh, crop_box=iso_box,
        erase_boxes=erase, method=args.iso_method)
    iso = fit_square(Image.fromarray(bw).convert("RGB"), args.size)
    ratio = float((bw < 128).mean())
    return line, iso, method, ratio


def main():
    ap = build_parser()
    args = ap.parse_args()
    validate_args(args)

    multi = args.mode != "ortho"
    pos, neg = build_prompts(args)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.control))[0]
    # 先做好控制图再加载 SDXL：等距图裁切/读图失败时不必占显存。
    if multi:
        line, iso, method, ratio = prepare_dual_controls(args)
        iso_scale = DEFAULT_ISO_SCALE[args.mode] if args.iso_scale is None else args.iso_scale
        line_path = os.path.join(args.out_dir, f"{stem}_{args.mode}_line_ctrl.png")
        iso_path = os.path.join(args.out_dir, f"{stem}_{args.mode}_iso_ctrl.png")
        line.save(line_path)
        iso.save(iso_path)
        print(f"iso prep {method}  黑像素占比 {ratio:.3f}  "
              f"line_scale={args.scale}  iso_scale={iso_scale}")
        print("saved", line_path)
        print("saved", iso_path)
        image = [line, iso]
        scale = [args.scale, iso_scale]
        extra = {
            "control_guidance_start": [0.0, 0.0],
            "control_guidance_end": [1.0, 1.0],
        }
    else:
        image = prepare_ortho_control(args.control, args.size)
        scale = args.scale
        extra = {}

    import torch
    pipe = load_pipeline(args.models_dir, multi)
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        g = torch.Generator("cuda").manual_seed(seed)
        img = pipe(pos, negative_prompt=neg, image=image,
                   num_inference_steps=args.steps, guidance_scale=args.cfg,
                   controlnet_conditioning_scale=scale, generator=g, **extra).images[0]
        name = os.path.join(args.out_dir, render_filename(stem, args.mode, args.style, seed))
        img.save(name)
        print("saved", name)
    print("ALL GENERATED")


if __name__ == "__main__":
    main()
