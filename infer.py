from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Optional

import torch

from wan_model import (
    DDPMSchedule,
    FrozenT5TextEncoder,
    WanDiT,
    WanVAE,
    default_device,
    sample_with_cfg,
)


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}


def align_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def save_video_tensor(video: torch.Tensor, out_path: Path, fps: int = 8) -> None:
    # Input is expected in C,T,H,W and normalized to [-1, 1].
    import imageio.v3 as iio
    import numpy as np

    x = ((video.clamp(-1, 1) + 1.0) * 127.5).byte()
    frames = x.permute(1, 2, 3, 0).contiguous().cpu().numpy().astype(np.uint8)  # T,H,W,C
    if out_path.suffix.lower() == ".gif":
        iio.imwrite(out_path, frames, loop=0, duration=1000 / fps)
    else:
        iio.imwrite(out_path, frames, fps=fps)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Inference script for Wan-style DDPM video generation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=root / "soccertrack_data")
    parser.add_argument("--prompt_from_data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=root / "sample.mp4")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--t5_model", type=str, default="google/t5-v1_1-large")
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def prompt_from_soccertrack(data_dir: Path, seed: int) -> str:
    media = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if not media:
        raise RuntimeError(f"No videos found in {data_dir}")
    rng = random.Random(seed)
    picked = media[rng.randrange(len(media))]
    caption_file = picked.with_suffix(".txt")
    if caption_file.exists():
        txt = caption_file.read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return picked.stem.replace("_", " ").replace("-", " ").strip()


def main() -> None:
    args = parse_args()
    if args.prompt is None and not args.prompt_from_data:
        raise ValueError("Set --prompt or use --prompt_from_data")

    device = default_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["model_config"]

    # Rebuild model topology from checkpoint config, then load weights.
    vae = WanVAE(
        in_channels=3,
        base_channels=cfg["base_channels"],
        latent_channels=cfg["latent_channels"],
    ).to(device)
    dit = WanDiT(
        latent_channels=cfg["latent_channels"],
        model_dim=cfg["model_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        patch_size=tuple(cfg.get("patch_size", (1, 2, 2))),
        t5_dim=cfg["t5_dim"],
    ).to(device)

    vae.load_state_dict(ckpt["vae_state_dict"])
    dit.load_state_dict(ckpt["dit_state_dict"])
    vae.eval()
    dit.eval()

    if args.prompt_from_data:
        prompt = prompt_from_soccertrack(args.data_dir, args.seed)
    else:
        prompt = args.prompt

    text_dtype = torch.float16 if device.type == "cuda" else torch.float32
    text_encoder = FrozenT5TextEncoder(model_name=args.t5_model, dtype=text_dtype).to(device)
    prompts = [prompt] * args.batch_size
    text_emb, _ = text_encoder.encode_prompts(prompts, device=device)
    text_emb = text_emb.to(device=device, dtype=torch.float32)

    # Latent shape must match VAE compression factors exactly: time/4, space/8.
    frames = align_up(args.frames, 4)
    height = align_up(args.height, 8)
    width = align_up(args.width, 8)
    latent_shape = (
        args.batch_size,
        cfg["latent_channels"],
        frames // 4,
        height // 8,
        width // 8,
    )

    schedule = DDPMSchedule(num_train_steps=args.diffusion_steps, device=device)
    # DDPM reverse process with CFG generates latent video, then VAE decodes pixels.
    latents = sample_with_cfg(
        model=dit,
        schedule=schedule,
        shape=latent_shape,
        text_embeddings=text_emb,
        guidance_scale=args.cfg_scale,
        num_steps=args.sample_steps,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        videos = vae.decode(latents.float())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_video_tensor(videos[0], args.output, fps=args.fps)
    print(f"prompt: {prompt}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
