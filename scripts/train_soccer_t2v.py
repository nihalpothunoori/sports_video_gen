#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soccer_t2v.config import AppConfig
from soccer_t2v.trainer import SoccerT2VTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Soccer T2V MVP from scratch.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "soccer_t2v_mvp.yaml"))
    parser.add_argument("--phase", type=str, choices=["vae", "dit"], required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="", help="Override checkpoint output directory.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override total training steps.")
    parser.add_argument("--train_fraction", type=float, default=None, help="Fraction of train manifest to use (0..1).")
    parser.add_argument("--val_fraction", type=float, default=None, help="Fraction of val manifest to use (0..1).")
    parser.add_argument("--vae_ckpt", type=str, default="", help="Path to trained VAE checkpoint for DIT phase.")
    parser.add_argument(
        "--use_mu_target",
        action="store_true",
        help="Use deterministic VAE mean latent as flow target (recommended for early DIT training).",
    )
    parser.add_argument(
        "--use_sampled_target",
        action="store_true",
        help="Use stochastic sampled VAE latent target instead of mean.",
    )
    parser.add_argument(
        "--latent_scale",
        type=float,
        default=None,
        help="Scale factor applied to VAE latent targets during DIT training.",
    )
    parser.add_argument(
        "--motion_focus_weight",
        type=float,
        default=None,
        help="Extra reconstruction weight on high-motion regions during VAE training.",
    )
    parser.add_argument(
        "--perceptual_weight",
        type=float,
        default=None,
        help="VGG perceptual reconstruction weight for VAE training.",
    )
    parser.add_argument(
        "--perceptual_num_frames",
        type=int,
        default=None,
        help="Number of temporal frames sampled per clip for perceptual loss.",
    )
    args = parser.parse_args()

    cfg = AppConfig.from_file(args.config)
    if args.vae_ckpt:
        cfg.train.vae_checkpoint = args.vae_ckpt
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.use_mu_target:
        cfg.train.use_mu_target = True
    if args.use_sampled_target:
        cfg.train.use_mu_target = False
    if args.latent_scale is not None:
        cfg.train.latent_scale = args.latent_scale
    if args.motion_focus_weight is not None:
        cfg.train.motion_focus_weight = args.motion_focus_weight
    if args.perceptual_weight is not None:
        cfg.train.perceptual_weight = args.perceptual_weight
    if args.perceptual_num_frames is not None:
        cfg.train.perceptual_num_frames = args.perceptual_num_frames
    if args.train_fraction is not None:
        cfg.data.train_fraction = args.train_fraction
    if args.val_fraction is not None:
        cfg.data.val_fraction = args.val_fraction
    trainer = SoccerT2VTrainer(cfg=cfg, phase=args.phase, device=args.device)
    trainer.train()


if __name__ == "__main__":
    main()
