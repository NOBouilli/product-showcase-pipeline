"""控制图 -> 设计线稿（ControlNet lineart + SD1.5 系底模）。

用法:
  python generate.py output/control_clean.png \
      --prompt "product design sketch of ..., front view, black ink linework on white paper" \
      --out output/sketch.png --seeds 101,202,303

建议参数：--cfg 5.5-7、--scale 0.85-1.0、--steps 30-32、512x512。
"""
import argparse
import os

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

NEG_DEFAULT = ("(dark background:1.4), (black background:1.4), (gray background:1.3), "
               "colored background, inverted colors, hatching, crosshatch, stripes, "
               "photo, photorealistic, 3d render, gradients, gray fill, color, "
               "frame, border, dimension lines, arrows, ruler, "
               "text, letters, numbers, watermark, signature, blurry, deformed")


def load_controlnet(path):
    try:
        return ControlNetModel.from_pretrained(path, torch_dtype=torch.float16, variant="fp16")
    except Exception:  # noqa: BLE001
        return ControlNetModel.from_pretrained(path, torch_dtype=torch.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control", help="控制图（白底黑线）")
    ap.add_argument("--prompt", required=True, help="产品本体描述：结构、材质、视角")
    ap.add_argument("--neg", default=NEG_DEFAULT)
    ap.add_argument("--base", default="models/sd15", help="底模目录（如 DreamShaper-8）")
    ap.add_argument("--lineart", default="models/lineart")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="101,202,303")
    ap.add_argument("--cfg", type=float, default=6.2)
    ap.add_argument("--scale", type=float, default=1.0, help="controlnet_conditioning_scale")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    controlnet = load_controlnet(args.lineart)
    try:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            args.base, controlnet=controlnet, torch_dtype=torch.float16,
            variant="fp16", safety_checker=None, requires_safety_checker=False)
    except Exception:  # noqa: BLE001
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            args.base, controlnet=controlnet, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    control = load_image(args.control).convert("RGB").resize((args.size, args.size))
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.out))[0]

    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        g = torch.Generator("cuda").manual_seed(seed)
        img = pipe(args.prompt, negative_prompt=args.neg, image=control,
                   num_inference_steps=args.steps, guidance_scale=args.cfg,
                   controlnet_conditioning_scale=args.scale, generator=g).images[0]
        name = os.path.join(out_dir, f"{stem}_{seed}.png")
        img.save(name)
        print("saved", name)
    print("ALL GENERATED")


if __name__ == "__main__":
    main()
