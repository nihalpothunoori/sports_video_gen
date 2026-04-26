from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from soccer_t2v.config import AppConfig
from soccer_t2v.data import create_dataloaders
from soccer_t2v.losses import VGGPerceptualLoss, flow_matching_loss, vae_loss
from soccer_t2v.models import SimpleTokenizer, SoccerVAE, TextConditionEncoder, VideoDiT
from soccer_t2v.utils.checkpoint import load_latest_checkpoint, save_checkpoint
from soccer_t2v.utils.logging_utils import setup_logger
from soccer_t2v.utils.seed import set_seed


@dataclass
class TrainingState:
    step: int = 0
    epoch: int = 0
    update_step: int = 0


class SoccerT2VTrainer:
    def __init__(self, cfg: AppConfig, phase: str, device: str = "cuda") -> None:
        self.cfg = cfg
        self.phase = phase
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger(f"soccer_t2v_{phase}", "/home/nihal/Documents/cachehacks/logs")
        set_seed(cfg.train.seed)

        self.train_loader, self.val_loader = create_dataloaders(cfg.data, cfg.train.batch_size)
        self.tokenizer = SimpleTokenizer(vocab_size=cfg.model.text_vocab_size, max_len=cfg.model.text_max_len)
        self.vae = SoccerVAE(base_channels=cfg.model.vae_base_channels, latent_channels=cfg.model.latent_channels).to(self.device)
        self.text_encoder = TextConditionEncoder(
            vocab_size=cfg.model.text_vocab_size, dim=cfg.model.text_dim, max_len=cfg.model.text_max_len
        ).to(self.device)
        self.dit = VideoDiT(
            latent_channels=cfg.model.latent_channels,
            text_dim=cfg.model.text_dim,
            dim=cfg.model.dit_dim,
            depth=cfg.model.dit_depth,
            heads=cfg.model.dit_heads,
            patch_t=cfg.model.patch_size_t,
            patch_h=cfg.model.patch_size_h,
            patch_w=cfg.model.patch_size_w,
        ).to(self.device)

        if phase == "vae":
            encoder_params = list(self.vae.encoder.parameters())
            decoder_params = list(self.vae.decoder.parameters())
            params = [
                {
                    "params": encoder_params,
                    "lr": cfg.train.learning_rate * cfg.train.encoder_lr_scale,
                },
                {
                    "params": decoder_params,
                    "lr": cfg.train.learning_rate,
                },
            ]
        elif phase == "dit":
            if cfg.train.vae_checkpoint:
                payload = torch.load(cfg.train.vae_checkpoint, map_location=self.device)
                self.vae.load_state_dict(payload["model"], strict=True)
                self.logger.info("Loaded VAE checkpoint for DIT phase: %s", cfg.train.vae_checkpoint)
            else:
                self.logger.warning("DIT phase started without a VAE checkpoint; this hurts convergence.")
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False
            params = list(self.dit.parameters()) + list(self.text_encoder.parameters())
        else:
            raise ValueError("phase must be 'vae' or 'dit'")

        self.optimizer = AdamW(params, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.train.precision == "fp16" and self.device.type == "cuda"))
        total_updates = max(1, cfg.train.max_steps // max(1, cfg.train.grad_accum_steps))
        self.scheduler = None
        if cfg.train.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_updates,
                eta_min=cfg.train.learning_rate * cfg.train.min_lr_ratio,
            )
        self.autocast_dtype = torch.bfloat16 if cfg.train.precision == "bf16" else torch.float16
        self.state = TrainingState()
        self.perceptual_loss = None
        if self.phase == "vae" and float(cfg.train.perceptual_weight) > 0.0:
            try:
                self.perceptual_loss = VGGPerceptualLoss().to(self.device).eval()
                self.logger.info(
                    "Enabled VGG perceptual loss: weight=%.3f frames=%d",
                    float(cfg.train.perceptual_weight),
                    int(cfg.train.perceptual_num_frames),
                )
            except Exception as exc:
                self.logger.warning("Perceptual loss disabled (load failure): %s", exc)
        self.out_dir = Path(cfg.train.output_dir) / phase
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path = self.out_dir / "tokenizer_vocab.json"
        if self.phase == "dit" and self.tokenizer_path.exists():
            self.tokenizer.load_json(self.tokenizer_path)
            self.logger.info("Loaded tokenizer vocab from %s", self.tokenizer_path)

    def maybe_resume(self) -> None:
        payload = load_latest_checkpoint(self.out_dir, self._train_module(), self.optimizer, self.scaler)
        if payload is None:
            return
        self.state.step = int(payload.get("step", 0))
        extra = payload.get("extra", {})
        self.state.epoch = int(extra.get("epoch", 0))
        self.state.update_step = int(extra.get("update_step", 0))
        if self.scheduler is not None and extra.get("scheduler") is not None:
            self.scheduler.load_state_dict(extra["scheduler"])
        if self.phase == "dit" and extra.get("tokenizer") is not None:
            self.tokenizer.load_dict(extra["tokenizer"])
        self.logger.info(
            "Resumed from step=%s update_step=%s epoch=%s",
            self.state.step,
            self.state.update_step,
            self.state.epoch,
        )

    def _train_module(self) -> torch.nn.Module:
        if self.phase == "vae":
            return self.vae
        return torch.nn.ModuleDict({"text": self.text_encoder, "dit": self.dit})

    def _vae_step(self, batch: dict[str, torch.Tensor | list[str]]) -> torch.Tensor:
        video = batch["video"].to(self.device, non_blocking=True)
        warmup_steps = max(1, int(self.cfg.train.kl_warmup_steps))
        kl_progress = min(1.0, float(self.state.step) / warmup_steps)
        if self.cfg.train.kl_schedule == "sigmoid":
            s = max(0.1, float(self.cfg.train.kl_sigmoid_steepness))
            x = (kl_progress - 0.5) * s
            kl_progress = 1.0 / (1.0 + math.exp(-x))
        elif self.cfg.train.kl_schedule == "linear":
            pass
        else:
            raise ValueError(f"Unsupported KL schedule: {self.cfg.train.kl_schedule}")
        kl_weight = self.cfg.train.kl_start_weight + (
            self.cfg.train.kl_base_weight - self.cfg.train.kl_start_weight
        ) * kl_progress
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
            recon, mu, logvar, _ = self.vae(video)
            loss, stats = vae_loss(
                recon,
                video,
                mu,
                logvar,
                perceptual=self.perceptual_loss,
                kl_weight=kl_weight,
                motion_focus_weight=float(self.cfg.train.motion_focus_weight),
                perceptual_weight=float(self.cfg.train.perceptual_weight),
                perceptual_num_frames=int(self.cfg.train.perceptual_num_frames),
            )
        current_lr = self.optimizer.param_groups[0]["lr"]
        mu_mean = float(mu.mean().item())
        mu_std = float(mu.std().item())
        logvar_mean = float(logvar.mean().item())
        logvar_std = float(logvar.std().item())
        self.logger.info(
            "step=%d loss=%.6f l1=%.6f w_l1=%.6f perc=%.6f kl=%.6f kl_w=%.8f motion_w=%.3f perc_w=%.3f lr=%.7f mu_mean=%.4f mu_std=%.4f logvar_mean=%.4f logvar_std=%.4f",
            self.state.step,
            stats["loss"],
            stats["l1"],
            stats["weighted_l1"],
            stats["perceptual"],
            stats["kl"],
            kl_weight,
            float(self.cfg.train.motion_focus_weight),
            float(self.cfg.train.perceptual_weight),
            current_lr,
            mu_mean,
            mu_std,
            logvar_mean,
            logvar_std,
        )
        return loss

    def _flow_step(self, batch: dict[str, torch.Tensor | list[str]]) -> torch.Tensor:
        video = batch["video"].to(self.device, non_blocking=True)
        prompts = batch["prompt"]
        with torch.no_grad():
            mu, logvar = self.vae.encode(video)
            if self.cfg.train.use_mu_target:
                z1 = mu
            else:
                z1 = self.vae.reparameterize(mu, logvar)
        scale = max(float(self.cfg.train.latent_scale), 1e-6)
        z1 = z1 / scale
        z0 = torch.randn_like(z1)
        t = torch.sigmoid(torch.randn(video.shape[0], device=self.device))
        zt = t.view(-1, 1, 1, 1, 1) * z1 + (1 - t).view(-1, 1, 1, 1, 1) * z0
        target_v = z1 - z0

        token_ids, attn_mask = self.tokenizer.encode_batch(prompts, self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
            text_emb = self.text_encoder(token_ids, attn_mask)
            pred_v = self.dit(zt, text_emb, attn_mask, t)
            loss = flow_matching_loss(pred_v, target_v)
        self.logger.info(
            "step=%d update_step=%d flow_loss=%.6f",
            self.state.step,
            self.state.update_step,
            float(loss.item()),
        )
        return loss

    def train(self) -> None:
        self.maybe_resume()
        grad_accum = self.cfg.train.grad_accum_steps
        pbar = tqdm(total=self.cfg.train.max_steps, initial=self.state.step, desc=f"train-{self.phase}")

        while self.state.step < self.cfg.train.max_steps:
            for batch in self.train_loader:
                if self.state.step >= self.cfg.train.max_steps:
                    break
                if self.phase == "vae":
                    loss = self._vae_step(batch)
                else:
                    loss = self._flow_step(batch)
                if not torch.isfinite(loss):
                    self.logger.warning("Non-finite loss at step=%d; skipping batch", self.state.step)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state.step += 1
                    pbar.update(1)
                    continue
                loss = loss / grad_accum
                self.scaler.scale(loss).backward()

                if (self.state.step + 1) % grad_accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self._train_module().parameters(), self.cfg.train.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.state.update_step += 1
                    self.logger.info(
                        "optimizer_update=%d micro_step=%d",
                        self.state.update_step,
                        self.state.step + 1,
                    )

                self.state.step += 1
                pbar.update(1)

                if self.state.step % self.cfg.train.save_every == 0:
                    save_checkpoint(
                        self.out_dir,
                        self.state.step,
                        self._train_module(),
                        self.optimizer,
                        self.scaler,
                        extra={
                            "epoch": self.state.epoch,
                            "phase": self.phase,
                            "update_step": self.state.update_step,
                            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                            "tokenizer": self.tokenizer.to_dict() if self.phase == "dit" else None,
                            "data_frames": int(self.cfg.data.frames),
                        },
                    )
                    if self.phase == "dit":
                        self.tokenizer.save_json(self.tokenizer_path)
            self.state.epoch += 1
        pbar.close()
