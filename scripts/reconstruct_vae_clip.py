#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_video, write_video

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soccer_t2v.config import AppConfig
from soccer_t2v.models import SoccerVAE


def load_preprocess_video(path: Path, frames: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    video, _, _ = read_video(str(path), pts_unit="sec")
    if video.numel() == 0 or video.shape[0] == 0:
        raise ValueError(f"Failed to decode frames from {path}")
    video = video.float() / 127.5 - 1.0
    video = video.permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]
    if video.shape[0] < frames:
        pad = video[-1:].repeat(frames - video.shape[0], 1, 1, 1)
        video = torch.cat([video, pad], dim=0)
    elif video.shape[0] > frames:
        idx = torch.linspace(0, video.shape[0] - 1, frames).long()
        video = video[idx]
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1,C,T,H,W]
    video = F.interpolate(video, size=(frames, height, width), mode="trilinear", align_corners=False)
    return video.to(device)


def to_uint8_t_h_w_c(video_1c_t_h_w: torch.Tensor) -> torch.Tensor:
    x = video_1c_t_h_w[0].permute(1, 2, 3, 0).contiguous()
    x = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu()
    return x


def match_time_dim(video: torch.Tensor, target_t: int) -> torch.Tensor:
    t = video.shape[2]
    if t == target_t:
        return video
    if t > target_t:
        return video[:, :, :target_t]
    pad = video[:, :, -1:].repeat(1, 1, target_t - t, 1, 1)
    return torch.cat([video, pad], dim=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct one clip through trained VAE.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "soccer_t2v_mvp.yaml"))
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--input_clip", type=str, required=True)
    parser.add_argument("--out_recon", type=str, default=str(ROOT / "output" / "vae_recon.mp4"))
    parser.add_argument("--out_side_by_side", type=str, default=str(ROOT / "output" / "vae_side_by_side.mp4"))
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = AppConfig.from_file(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vae = SoccerVAE(base_channels=cfg.model.vae_base_channels, latent_channels=cfg.model.latent_channels).to(device).eval()
    payload = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(payload["model"], strict=True)

    inp = load_preprocess_video(
        Path(args.input_clip),
        frames=args.frames,
        height=args.height,
        width=args.width,
        device=device,
    )

    with torch.no_grad():
        mu, _ = vae.encode(inp)
        recon = vae.decode(mu).clamp(-1, 1)
        recon = match_time_dim(recon, inp.shape[2])

    inp_u8 = to_uint8_t_h_w_c(inp)
    recon_u8 = to_uint8_t_h_w_c(recon)
    side = torch.cat([inp_u8, recon_u8], dim=2)  # concat width

    out_recon = Path(args.out_recon)
    out_recon.parent.mkdir(parents=True, exist_ok=True)
    out_side = Path(args.out_side_by_side)
    out_side.parent.mkdir(parents=True, exist_ok=True)

    write_video(str(out_recon), recon_u8, fps=args.fps)
    write_video(str(out_side), side, fps=args.fps)

    print(f"Saved reconstruction: {out_recon}")
    print(f"Saved side-by-side:  {out_side}")


if __name__ == "__main__":
    main()
