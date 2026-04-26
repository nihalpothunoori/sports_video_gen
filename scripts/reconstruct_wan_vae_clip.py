#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torchvision.io import read_video, write_video

ROOT = Path(__file__).resolve().parents[1]
DIFFSYNTH_ROOT = ROOT / "scripts" / "DiffSynth-Studio"
sys.path.insert(0, str(DIFFSYNTH_ROOT))

from huggingface_hub import hf_hub_download

from diffsynth.models.wan_video_vae import WanVideoVAE


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
    parser = argparse.ArgumentParser(description="Reconstruct one clip using pretrained Wan VAE.")
    parser.add_argument("--input_clip", type=str, required=True)
    parser.add_argument("--out_recon", type=str, default=str(ROOT / "output" / "wan_vae_recon.mp4"))
    parser.add_argument("--out_side_by_side", type=str, default=str(ROOT / "output" / "wan_vae_side_by_side.mp4"))
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vae_repo", type=str, default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--vae_file", type=str, default="Wan2.1_VAE.pth")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    inp = load_preprocess_video(
        Path(args.input_clip),
        frames=args.frames,
        height=args.height,
        width=args.width,
        device=device,
    )

    vae_path = hf_hub_download(repo_id=args.vae_repo, filename=args.vae_file)
    if vae_path.endswith(".safetensors"):
        state = load_file(vae_path)
    else:
        raw = torch.load(vae_path, map_location="cpu")
        if isinstance(raw, dict) and "model_state" in raw:
            raw = raw["model_state"]
        # Wan pth files store bare module keys; WanVideoVAE expects "model.*".
        state = {("model." + k if not k.startswith("model.") else k): v for k, v in raw.items()}

    vae = WanVideoVAE().to(device).eval()
    missing, unexpected = vae.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in Wan VAE state dict: {unexpected[:8]}")
    if missing:
        print(f"Warning: missing keys when loading Wan VAE: {len(missing)}")

    with torch.no_grad():
        latents = vae.encode(inp, device=device, tiled=False)
        recon = vae.decode(latents, device=device, tiled=False)
        recon = match_time_dim(recon, inp.shape[2]).clamp(-1, 1)

    inp_u8 = to_uint8_t_h_w_c(inp)
    recon_u8 = to_uint8_t_h_w_c(recon)
    side = torch.cat([inp_u8, recon_u8], dim=2)

    out_recon = Path(args.out_recon)
    out_recon.parent.mkdir(parents=True, exist_ok=True)
    out_side = Path(args.out_side_by_side)
    out_side.parent.mkdir(parents=True, exist_ok=True)

    write_video(str(out_recon), recon_u8, fps=args.fps)
    write_video(str(out_side), side, fps=args.fps)

    print(f"Loaded Wan VAE weights: {vae_path}")
    print(f"Saved reconstruction: {out_recon}")
    print(f"Saved side-by-side:  {out_side}")


if __name__ == "__main__":
    main()
