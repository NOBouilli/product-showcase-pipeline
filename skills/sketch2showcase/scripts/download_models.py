"""下载 sketch2showcase 管线模型（RealVisXL_V5 + MistoLine + x4 放大器，约 12GB）。

用法: python download_models.py --root models
"""
import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="models", help="模型存放根目录")
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)

    def get(repo, sub, allow, ignore=()):
        local = os.path.join(args.root, sub)
        print(f"==> {repo} -> {local}", flush=True)
        snapshot_download(repo, local_dir=local, allow_patterns=allow,
                          ignore_patterns=list(ignore))
        print(f"<== {repo} done", flush=True)

    # 1) SDXL 真实感底模（diffusers 完整格式只取 fp16；根目录单文件整模跳过省 10GB）
    get("SG161222/RealVisXL_V5.0", "realvisxl_v5",
        allow=["model_index.json", "*/*.json", "*/*.txt", "*/*.fp16.safetensors"],
        ignore=["RealVisXL_V5.0_fp16.safetensors", "RealVisXL_V5.0_fp32.safetensors",
                "*.bin", "*.png", "*.jpg", "*.jpeg", "*ip-adapter*"])
    # 2) MistoLine：通用线稿 ControlNet（SDXL 版）
    get("TheMistoAI/MistoLine", "mistoline",
        allow=["config.json", "diffusion_pytorch_model.fp16.safetensors"])
    # 3) x4 放大器（细节放大图；SUPIR 权重 gated，用这个轻量替代）
    get("stabilityai/stable-diffusion-x4-upscaler", "x4upscale",
        allow=["model_index.json", "*/*.json", "*/*.txt", "*/*.fp16.safetensors"],
        ignore=["*.bin", "*.md"])
    print("SHOWCASE MODELS DOWNLOADED")


if __name__ == "__main__":
    main()
