from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_video, write_video

from soccer_t2v.config import AppConfig
from soccer_t2v.models import SimpleTokenizer, SoccerVAE, TextConditionEncoder, VideoDiT


@torch.no_grad()
def generate_video(
    cfg: AppConfig,
    prompt: str,
    vae_ckpt: str | Path,
    dit_ckpt: str | Path,
    out_path: str | Path,
    init_video_path: str | Path | None = None,
    denoise_strength: float = 0.25,
    seed: int = 42,
    device: str = "cuda",
) -> Path:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    vae = SoccerVAE(base_channels=cfg.model.vae_base_channels, latent_channels=cfg.model.latent_channels).to(dev).eval()
    text_encoder = TextConditionEncoder(
        vocab_size=cfg.model.text_vocab_size, dim=cfg.model.text_dim, max_len=cfg.model.text_max_len
    ).to(dev).eval()
    dit = VideoDiT(
        latent_channels=cfg.model.latent_channels,
        text_dim=cfg.model.text_dim,
        dim=cfg.model.dit_dim,
        depth=cfg.model.dit_depth,
        heads=cfg.model.dit_heads,
        patch_t=cfg.model.patch_size_t,
        patch_h=cfg.model.patch_size_h,
        patch_w=cfg.model.patch_size_w,
    ).to(dev).eval()
    tokenizer = SimpleTokenizer(vocab_size=cfg.model.text_vocab_size, max_len=cfg.model.text_max_len)

    vae_payload = torch.load(vae_ckpt, map_location=dev)
    vae.load_state_dict(vae_payload["model"], strict=True)
    dit_payload = torch.load(dit_ckpt, map_location=dev)
    model_state = dit_payload["model"]
    extra = dit_payload.get("extra", {})
    if isinstance(extra, dict):
        tok_payload = extra.get("tokenizer")
        if isinstance(tok_payload, dict):
            tokenizer.load_dict(tok_payload)
    trained_frames = int(extra.get("data_frames", cfg.data.frames)) if isinstance(extra, dict) else cfg.data.frames
    if trained_frames != cfg.data.frames:
        cfg.data.frames = trained_frames
    if "text" in model_state and "dit" in model_state:
        text_encoder.load_state_dict(model_state["text"], strict=True)
        dit.load_state_dict(model_state["dit"], strict=True)
    else:
        # Backward compatibility for checkpoints saved from ModuleDict.
        text_state = {}
        dit_state = {}
        for key, value in model_state.items():
            if key.startswith("text."):
                text_state[key[len("text.") :]] = value
            elif key.startswith("dit."):
                dit_state[key[len("dit.") :]] = value
        if not text_state or not dit_state:
            raise KeyError(
                "Could not parse DIT checkpoint model state. "
                "Expected keys {'text','dit'} or prefixed keys 'text.*' and 'dit.*'."
            )
        text_encoder.load_state_dict(text_state, strict=True)
        dit.load_state_dict(dit_state, strict=True)

    bsz = 1
    t = max(1, cfg.data.frames // 4)
    h = cfg.data.height // 8
    w = cfg.data.width // 8
    x = torch.randn((bsz, cfg.model.latent_channels, t, h, w), device=dev)

    if init_video_path is not None:
        video, _, _ = read_video(str(init_video_path), pts_unit="sec")
        if video.numel() == 0 or video.shape[0] == 0:
            raise ValueError(f"Could not decode frames from init video: {init_video_path}")
        video = video.float() / 127.5 - 1.0
        video = video.permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]
        if video.shape[0] < cfg.data.frames:
            pad = video[-1:].repeat(cfg.data.frames - video.shape[0], 1, 1, 1)
            video = torch.cat([video, pad], dim=0)
        elif video.shape[0] > cfg.data.frames:
            idx = torch.linspace(0, video.shape[0] - 1, cfg.data.frames).long()
            video = video[idx]
        video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1,C,T,H,W]
        video = F.interpolate(video, size=(cfg.data.frames, cfg.data.height, cfg.data.width), mode="trilinear", align_corners=False)
        video = video.to(dev)
        mu, _ = vae.encode(video)
        if x.shape != mu.shape:
            x = torch.randn_like(mu)
        denoise_strength = float(max(0.0, min(1.0, denoise_strength)))
        x = (1.0 - denoise_strength) * mu + denoise_strength * x
    token_ids, attn_mask = tokenizer.encode_batch([prompt], dev)
    text_emb = text_encoder(token_ids, attn_mask)

    steps = cfg.train.num_inference_steps
    for i in range(steps, 0, -1):
        t_val = torch.full((bsz,), i / steps, device=dev)
        v = dit(x, text_emb, attn_mask, t_val)
        x = x - v / steps

    video = vae.decode(x).clamp(-1, 1)
    video = ((video[0].permute(1, 2, 3, 0) + 1) * 127.5).to(torch.uint8).cpu()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(out_path), video, fps=cfg.data.fps)
    return out_path
