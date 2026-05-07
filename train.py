from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from nlp_mapper import SoccerTrackPromptMapper
from wan_model import (
    DDPMSchedule,
    FrozenT5TextEncoder,
    KLBetaRamp,
    VAELoss,
    WanDiT,
    WanVAE,
    default_device,
    diffusion_training_step,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}


class VideoTextDataset(Dataset):
    # Dataset expects media files and optional same-name .txt captions.
    def __init__(self, root: Path, frames: int, height: int, width: int) -> None:
        self.root = root
        self.frames = frames
        self.height = height
        self.width = width
        self.samples = self._collect(root)
        if not self.samples:
            raise RuntimeError(f"No media files found in {root}")

    def _collect(self, root: Path) -> List[Path]:
        # Expects SoccerTrack clips already exported as short per-play/per-segment files.
        files: List[Path] = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS.union(VIDEO_EXTS):
                files.append(p)
        files.sort()
        return files

    def _read_caption(self, media_path: Path) -> str:
        # Prefer explicit caption files; fallback to a cleaned filename.
        txt = media_path.with_suffix(".txt")
        if txt.exists():
            return txt.read_text(encoding="utf-8").strip()
        return media_path.stem.replace("_", " ").replace("-", " ").strip()

    def _load_image(self, path: Path) -> torch.Tensor:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        x = torch.from_numpy(np.array(img)).float() / 255.0
        x = x.permute(2, 0, 1)  # C,H,W
        # Expand a still image to a pseudo-video so the model path stays uniform.
        x = x.unsqueeze(1).repeat(1, self.frames, 1, 1)  # C,T,H,W
        return x

    def _load_video(self, path: Path) -> torch.Tensor:
        try:
            from torchvision.io import read_video
        except ImportError as exc:
            raise ImportError("Install torchvision to read video files") from exc

        frames, _, _ = read_video(str(path), pts_unit="sec")
        if frames.numel() == 0:
            raise RuntimeError(f"Could not decode video: {path}")
        frames = frames.float() / 255.0  # T,H,W,C
        total = frames.shape[0]
        # Uniform frame sampling keeps clip length fixed across variable input videos.
        idx = torch.linspace(0, total - 1, self.frames).long().clamp(0, total - 1)
        frames = frames.index_select(0, idx)
        x = frames.permute(3, 0, 1, 2).contiguous()  # C,T,H,W
        return x

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to T,C,H,W so interpolate can resize all frames in one call.
        c, t, h, w = x.shape
        x = x.permute(1, 0, 2, 3).contiguous()  # T,C,H,W
        x = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=False)
        x = x.permute(1, 0, 2, 3).contiguous()  # C,T,H,W
        return x

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path = self.samples[index]
        if path.suffix.lower() in IMAGE_EXTS:
            x = self._load_image(path)
        else:
            x = self._load_video(path)
        x = self._resize(x)
        x = x * 2.0 - 1.0
        caption = self._read_caption(path)
        return x, caption


def collate_batch(batch: Sequence[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    videos = torch.stack([item[0] for item in batch], dim=0)
    captions = [item[1] for item in batch]
    return videos, captions


def validate(
    vae: WanVAE,
    dit: WanDiT,
    text_encoder: FrozenT5TextEncoder,
    schedule: DDPMSchedule,
    loader: DataLoader,
    vae_loss_fn: VAELoss,
    beta_ramp: KLBetaRamp,
    step: int,
    device: torch.device,
    prompt_mapper: Optional[SoccerTrackPromptMapper] = None,
) -> Tuple[float, float]:
    # Validation runs both objectives: VAE reconstruction and DDPM epsilon prediction.
    vae.eval()
    dit.eval()
    total_vae = 0.0
    total_diff = 0.0
    n = 0

    with torch.no_grad():
        for videos, prompts in loader:
            videos = videos.to(device=device, dtype=torch.float32)
            if prompt_mapper is not None:
                prompts = [prompt_mapper.normalize_prompt(p)[0] for p in prompts]
            recon, mu, logvar = vae(videos)
            beta = beta_ramp(step)
            vae_loss, _ = vae_loss_fn(recon, videos, mu, logvar, beta=beta)

            # Reuse encoder mean for a deterministic latent path during validation.
            latents = mu
            text_emb, _ = text_encoder.encode_prompts(prompts, device=device)
            text_emb = text_emb.to(device=device, dtype=torch.float32)
            diff_loss, _ = diffusion_training_step(
                dit,
                schedule=schedule,
                clean_latents=latents,
                text_embeddings=text_emb,
                cfg_dropout_prob=0.0,
            )

            bs = videos.size(0)
            total_vae += float(vae_loss.item()) * bs
            total_diff += float(diff_loss.item()) * bs
            n += bs

    vae.train()
    dit.train()
    return total_vae / max(n, 1), total_diff / max(n, 1)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train Wan-style VAE+DiT with DDPM objective")
    parser.add_argument("--data_dir", type=Path, default=root / "soccertrack_data")
    parser.add_argument("--output_dir", type=Path, default=root / "checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr_vae", type=float, default=2e-4)
    parser.add_argument("--lr_dit", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--t5_model", type=str, default="google/t5-v1_1-large")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--normalize_captions", action="store_true")
    parser.add_argument("--disable_llm_mapper", action="store_true")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--llm_api_base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--llm_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--latent_channels", type=int, default=16)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = default_device()
    print(f"Using device: {device}")

    # Fixed 80/20 split so validation tracks generalization from the same source set.
    dataset = VideoTextDataset(
        root=args.data_dir,
        frames=args.frames,
        height=args.height,
        width=args.width,
    )
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    if n_train == 0 or n_val == 0:
        raise RuntimeError(f"Need at least 2 samples for 80/20 split. Found {n_total}.")
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
        drop_last=False,
    )

    text_dtype = torch.float16 if device.type == "cuda" else torch.float32
    text_encoder = FrozenT5TextEncoder(model_name=args.t5_model, dtype=text_dtype).to(device)
    t5_dim = text_encoder.hidden_size

    vae = WanVAE(
        in_channels=3,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
    ).to(device)
    dit = WanDiT(
        latent_channels=args.latent_channels,
        model_dim=args.model_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        patch_size=(1, 2, 2),
        t5_dim=t5_dim,
    ).to(device)

    schedule = DDPMSchedule(num_train_steps=args.diffusion_steps, device=device)
    vae_loss_fn = VAELoss()
    beta_ramp = KLBetaRamp(beta_start=0.0, beta_end=1e-4, warmup_steps=50_000)
    prompt_mapper: Optional[SoccerTrackPromptMapper] = None
    if args.normalize_captions:
        prompt_mapper = SoccerTrackPromptMapper(
            use_llm=not args.disable_llm_mapper,
            model=args.llm_model,
            api_base_url=args.llm_api_base_url,
            api_key=os.getenv(args.llm_api_key_env),
        )

    vae_opt = torch.optim.AdamW(vae.parameters(), lr=args.lr_vae, weight_decay=args.weight_decay)
    dit_opt = torch.optim.AdamW(dit.parameters(), lr=args.lr_dit, weight_decay=args.weight_decay)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        vae.train()
        dit.train()
        train_vae = 0.0
        train_diff = 0.0
        seen = 0

        for videos, prompts in train_loader:
            videos = videos.to(device=device, dtype=torch.float32)
            bs = videos.size(0)
            if prompt_mapper is not None:
                prompts = [prompt_mapper.normalize_prompt(p)[0] for p in prompts]

            # Step 1: optimize VAE reconstruction + KL with beta ramp.
            vae_opt.zero_grad(set_to_none=True)
            recon, mu, logvar = vae(videos)
            beta = beta_ramp(global_step)
            vae_loss, _ = vae_loss_fn(recon, videos, mu, logvar, beta=beta)
            vae_loss.backward()
            vae_opt.step()

            # Step 2: freeze VAE path and train DiT on noised latents.
            with torch.no_grad():
                mu_detached, _ = vae.encode(videos)
                latents = mu_detached
                text_emb, _ = text_encoder.encode_prompts(prompts, device=device)
                text_emb = text_emb.to(device=device, dtype=torch.float32)

            dit_opt.zero_grad(set_to_none=True)
            diff_loss, _ = diffusion_training_step(
                dit,
                schedule=schedule,
                clean_latents=latents,
                text_embeddings=text_emb,
                cfg_dropout_prob=args.cfg_dropout,
            )
            diff_loss.backward()
            dit_opt.step()

            train_vae += float(vae_loss.item()) * bs
            train_diff += float(diff_loss.item()) * bs
            seen += bs
            global_step += 1

        val_vae, val_diff = validate(
            vae=vae,
            dit=dit,
            text_encoder=text_encoder,
            schedule=schedule,
            loader=val_loader,
            vae_loss_fn=vae_loss_fn,
            beta_ramp=beta_ramp,
            step=global_step,
            device=device,
            prompt_mapper=prompt_mapper,
        )
        train_vae /= max(seen, 1)
        train_diff /= max(seen, 1)

        print(
            f"epoch={epoch} "
            f"train_vae={train_vae:.5f} train_diff={train_diff:.5f} "
            f"val_vae={val_vae:.5f} val_diff={val_diff:.5f}"
        )

        if epoch % args.save_every == 0:
            # Keep enough config in checkpoint to fully rebuild for inference.
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "vae_state_dict": vae.state_dict(),
                "dit_state_dict": dit.state_dict(),
                "model_config": {
                    "base_channels": args.base_channels,
                    "latent_channels": args.latent_channels,
                    "model_dim": args.model_dim,
                    "depth": args.depth,
                    "num_heads": args.num_heads,
                    "patch_size": (1, 2, 2),
                    "t5_dim": t5_dim,
                },
                "train_config": vars(args),
            }
            out_path = args.output_dir / f"wan_epoch_{epoch:03d}.pt"
            torch.save(ckpt, out_path)
            print(f"saved checkpoint: {out_path}")


if __name__ == "__main__":
    main()
