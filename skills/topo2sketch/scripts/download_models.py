"""下载 topo2sketch 管线模型（SD1.5 底模 + lineart ControlNet，约 6GB）。

用法: python download_models.py --root models
走 hf-mirror；HF_HUB_DISABLE_XET=1 规避新版 huggingface_hub 的 Xet 401。
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

    jobs = [
        ("stable-diffusion-v1-5/stable-diffusion-v1-5", "sd15",
         ["model_index.json", "*/*.json", "*/*.txt", "*/*.fp16.safetensors"],
         ["v1-5-pruned*", "safety_checker/*"]),
        ("lllyasviel/control_v11p_sd15_lineart", "lineart",
         ["*.json", "*/*.json", "*.safetensors", "*/*.safetensors"], []),
    ]
    for repo, sub, allow, ignore in jobs:
        local = os.path.join(args.root, sub)
        print(f"==> {repo} -> {local}", flush=True)
        snapshot_download(repo, local_dir=local, allow_patterns=allow,
                          ignore_patterns=ignore)
        print(f"<== {repo} done", flush=True)
    print("MODELS DOWNLOADED")


if __name__ == "__main__":
    main()
