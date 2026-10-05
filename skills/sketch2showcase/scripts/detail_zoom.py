"""细节放大图：从展示图裁出局部，用 x4 放大模型重绘。

用法:
  python detail_zoom.py showcase.png --box 380,120,760,500 --out-dir output/zoom

box 为原图像素坐标 (x1,y1,x2,y2)，裁剪区建议 400-600px 见方（输出即其 4 倍边长）。
"""
import argparse
import os

import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--box", required=True, help="x1,y1,x2,y2 原图像素坐标")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out-dir", default="output/zoom")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=9.0)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--prompt", default="high detail industrial product, sharp focus, clean surfaces")
    args = ap.parse_args()

    x1, y1, x2, y2 = [int(v) for v in args.box.split(",")]
    src = Image.open(args.src).convert("RGB")
    crop = src.crop((x1, y1, x2, y2))
    print("crop size:", crop.size, flush=True)

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        os.path.join(args.models_dir, "x4upscale"), torch_dtype=torch.float16,
        variant="fp16")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    pipe.to("cuda")

    g = torch.Generator("cuda").manual_seed(args.seed)
    out = pipe(prompt=args.prompt, image=crop, num_inference_steps=args.steps,
               guidance_scale=args.cfg, generator=g).images[0]

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.src))[0]
    name = os.path.join(args.out_dir, f"{stem}_zoom_{x1}_{y1}.png")
    out.save(name)
    print("saved", name, out.size, flush=True)


if __name__ == "__main__":
    main()
