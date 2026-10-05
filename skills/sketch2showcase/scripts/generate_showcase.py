"""线稿控制图 -> 产品展示图（SDXL RealVisXL + MistoLine 线稿控制网）。

用法:
  python generate_showcase.py ctrl_front.png --product "hydrostatic testing machine \
      for fire extinguisher cylinders" --view "front view" --seeds 101,202
  python generate_showcase.py ctrl_exploded.png --style clay --view "exploded view"

输出 <out-dir>/<控制图名>_<style>_<seed>.png。SDXL 原生 1024x1024，
RTX 5070 Ti 单图约 1 分钟（模型加载约 1.5 分钟）。
"""
import argparse
import os

import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control", help="线稿控制图（白底黑线）")
    ap.add_argument("--product", required=True, help="产品名（英文提示词，进渲染提示）")
    ap.add_argument("--style", default="render", choices=STYLES)
    ap.add_argument("--view", default="three-quarter view",
                    help='视图提示词，如 "front view" / "side view" / "top view" / "exploded view"')
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out-dir", default="output/showcase")
    ap.add_argument("--seeds", default="101,202,303")
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--scale", type=float, default=0.7,
                    help="controlnet_conditioning_scale，MistoLine 建议 0.6-0.9")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    controlnet = ControlNetModel.from_pretrained(
        os.path.join(args.models_dir, "mistoline"), torch_dtype=torch.float16,
        variant="fp16", low_cpu_mem_usage=True, device_map="cuda")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        os.path.join(args.models_dir, "realvisxl_v5"), controlnet=controlnet,
        torch_dtype=torch.float16, variant="fp16",
        safety_checker=None, requires_safety_checker=False,
        image_encoder=None, feature_extractor=None, low_cpu_mem_usage=True,
        device_map="balanced")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    control = load_image(args.control).convert("RGB").resize((args.size, args.size))
    preset = STYLES[args.style]
    pos = preset["pos"].format(product=args.product, view=args.view)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.control))[0]
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        g = torch.Generator("cuda").manual_seed(seed)
        img = pipe(pos, negative_prompt=preset["neg"], image=control,
                   num_inference_steps=args.steps, guidance_scale=args.cfg,
                   controlnet_conditioning_scale=args.scale, generator=g).images[0]
        name = os.path.join(args.out_dir, f"{stem}_{args.style}_{seed}.png")
        img.save(name)
        print("saved", name)
    print("ALL GENERATED")


if __name__ == "__main__":
    main()
