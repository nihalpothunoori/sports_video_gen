#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_video

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soccer_t2v.config import AppConfig


def _load_video(path: str, frames: int) -> torch.Tensor:
    v, _, _ = read_video(path, pts_unit="sec")
    v = v.float() / 255.0
    if v.shape[0] > frames:
        idx = torch.linspace(0, v.shape[0] - 1, frames).long()
        v = v[idx]
    elif v.shape[0] < frames:
        pad = v[-1:].repeat(frames - v.shape[0], 1, 1, 1)
        v = torch.cat([v, pad], dim=0)
    return v


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick MVP eval with PSNR proxy.")
    parser.add_argument("--generated", type=str, required=True)
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "soccer_t2v_mvp.yaml"))
    args = parser.parse_args()

    cfg = AppConfig.from_file(args.config)
    g = _load_video(args.generated, cfg.data.frames)
    r = _load_video(args.reference, cfg.data.frames)
    print(f"PSNR: {psnr(g, r):.4f} dB")


if __name__ == "__main__":
    main()
