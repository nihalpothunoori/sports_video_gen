from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision.models import VGG16_Weights, vgg16
        except Exception as exc:  # pragma: no cover - defensive import path
            raise RuntimeError("torchvision VGG16 is required for perceptual loss") from exc

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        self.blocks = nn.ModuleList(
            [
                vgg[:4],   # relu1_2
                vgg[4:9],  # relu2_2
                vgg[9:16],  # relu3_3
            ]
        )
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Inputs are expected in [-1, 1], shape [N, 3, H, W]
        pred = (pred + 1.0) * 0.5
        target = (target + 1.0) * 0.5
        pred = F.interpolate(pred, size=(224, 224), mode="bilinear", align_corners=False)
        target = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = pred.new_tensor(0.0)
        x = pred
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + F.l1_loss(x, y)
        return loss / len(self.blocks)


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    perceptual: nn.Module | None = None,
    l1_weight: float = 3.0,
    kl_weight: float = 3e-6,
    motion_focus_weight: float = 0.0,
    perceptual_weight: float = 0.0,
    perceptual_num_frames: int = 4,
) -> tuple[torch.Tensor, dict[str, float]]:
    l1 = F.l1_loss(recon, target)
    weighted_l1 = l1
    if motion_focus_weight > 0.0:
        # Emphasize moving regions so the VAE allocates capacity to players over static pitch/stadium.
        motion = (target[:, :, 1:] - target[:, :, :-1]).abs().mean(dim=1, keepdim=True)  # [B,1,T-1,H,W]
        motion = torch.cat([motion[:, :, :1], motion], dim=2)  # [B,1,T,H,W]
        motion = motion / (motion.amax(dim=(2, 3, 4), keepdim=True) + 1e-6)
        pixel_weight = 1.0 + motion_focus_weight * motion
        weighted_l1 = (pixel_weight * (recon - target).abs()).mean()

    perceptual_loss = l1.new_tensor(0.0)
    if perceptual is not None and perceptual_weight > 0.0:
        t = recon.shape[2]
        k = max(1, min(int(perceptual_num_frames), t))
        idx = torch.linspace(0, t - 1, steps=k, device=recon.device).long()
        # [B,C,T,H,W] -> [B*k,C,H,W]
        recon_2d = recon[:, :, idx].permute(0, 2, 1, 3, 4).reshape(-1, recon.shape[1], recon.shape[3], recon.shape[4])
        target_2d = target[:, :, idx].permute(0, 2, 1, 3, 4).reshape(-1, target.shape[1], target.shape[3], target.shape[4])
        perceptual_loss = perceptual(recon_2d, target_2d)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = l1_weight * weighted_l1 + perceptual_weight * perceptual_loss + kl_weight * kl
    return total, {
        "l1": float(l1.item()),
        "weighted_l1": float(weighted_l1.item()),
        "perceptual": float(perceptual_loss.item()),
        "kl": float(kl.item()),
        "loss": float(total.item()),
    }


def flow_matching_loss(
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(pred_v, target_v)
