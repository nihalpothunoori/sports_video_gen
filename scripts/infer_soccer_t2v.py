#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soccer_t2v.config import AppConfig
from soccer_t2v.inference import generate_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for Soccer T2V MVP.")
    parser.add_argument("prompt", type=str)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "soccer_t2v_mvp.yaml"))
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--dit_ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default=str(ROOT / "output" / "soccer_t2v_mvp.mp4"))
    parser.add_argument("--init_video", type=str, default="", help="Optional source clip for clip-conditioned generation.")
    parser.add_argument("--denoise_strength", type=float, default=0.25, help="0 keeps source latent, 1 uses full noise.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = AppConfig.from_file(args.config)
    out = generate_video(
        cfg=cfg,
        prompt=args.prompt,
        vae_ckpt=args.vae_ckpt,
        dit_ckpt=args.dit_ckpt,
        out_path=args.out,
        init_video_path=args.init_video or None,
        denoise_strength=args.denoise_strength,
        seed=args.seed,
        device=args.device,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
