from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# VAE blocks used for latent video compression.
class CausalConv3d(nn.Module):
    """3D conv with causal padding on time and symmetric spatial padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int, int] = 3,
        stride: int | Tuple[int, int, int] = 1,
        dilation: int | Tuple[int, int, int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kt, kh, kw = self.kernel_size
        dt, dh, dw = self.dilation

        # Temporal padding is one-sided so each frame only sees current/past frames.
        t_left = (kt - 1) * dt
        h_pad = ((kh - 1) * dh) // 2
        w_pad = ((kw - 1) * dw) // 2
        x = F.pad(x, (w_pad, w_pad, h_pad, h_pad, t_left, 0))
        return self.conv(x)


class VAEResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 32) -> None:
        super().__init__()
        g1 = min(groups, in_channels)
        g2 = min(groups, out_channels)
        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else CausalConv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Upsample3d(nn.Module):
    def __init__(self, channels: int, scale_factor: Tuple[int, int, int]) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.proj = CausalConv3d(channels, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.proj(x)


class VAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 16,
        temporal_downsample: Sequence[int] = (2, 2, 1),
    ) -> None:
        super().__init__()
        if len(temporal_downsample) != len(channel_mult) - 1:
            raise ValueError("temporal_downsample must have len(channel_mult)-1 entries")

        self.in_conv = CausalConv3d(in_channels, base_channels, kernel_size=3)
        blocks = []
        ch = base_channels
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(VAEResBlock(ch, out_ch))
                ch = out_ch
            if level < len(channel_mult) - 1:
                # Downsample spatially each stage; temporal stride is configurable per stage.
                blocks.append(
                    CausalConv3d(
                        ch,
                        ch,
                        kernel_size=3,
                        stride=(temporal_downsample[level], 2, 2),
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.mid = nn.Sequential(VAEResBlock(ch, ch), VAEResBlock(ch, ch))
        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = CausalConv3d(ch, latent_channels * 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.mid(x)
        x = self.out_conv(F.silu(self.out_norm(x)))
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 16,
        temporal_upsample: Sequence[int] = (1, 2, 2),
    ) -> None:
        super().__init__()
        if len(temporal_upsample) != len(channel_mult) - 1:
            raise ValueError("temporal_upsample must have len(channel_mult)-1 entries")

        ch = base_channels * channel_mult[-1]
        self.in_conv = CausalConv3d(latent_channels, ch, kernel_size=1)
        self.mid = nn.Sequential(VAEResBlock(ch, ch), VAEResBlock(ch, ch))

        blocks = []
        rev_mults = list(channel_mult[::-1])
        for level, mult in enumerate(rev_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(VAEResBlock(ch, out_ch))
                ch = out_ch
            if level < len(rev_mults) - 1:
                # Decoder mirrors encoder and restores temporal/spatial resolution.
                blocks.append(Upsample3d(ch, (temporal_upsample[level], 2, 2)))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = CausalConv3d(ch, out_channels, kernel_size=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(z)
        x = self.mid(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_conv(F.silu(self.out_norm(x)))
        return torch.tanh(x)


class WanVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = VAEEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
        )
        self.decoder = VAEDecoder(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Standard VAE sampling: z = mu + sigma * eps.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(video)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(video)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


@dataclass
class KLBetaRamp:
    beta_start: float = 0.0
    beta_end: float = 1e-4
    warmup_steps: int = 50_000

    def __call__(self, step: int) -> float:
        # KL weight ramps from beta_start to beta_end over warmup_steps.
        if self.warmup_steps <= 0:
            return self.beta_end
        alpha = min(max(step, 0) / float(self.warmup_steps), 1.0)
        return self.beta_start + (self.beta_end - self.beta_start) * alpha


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).mean()


class VAELoss(nn.Module):
    def __init__(self, recon_weight: float = 1.0) -> None:
        super().__init__()
        self.recon_weight = recon_weight

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        recon_loss = F.l1_loss(recon, target)
        kl_loss = kl_divergence(mu, logvar)
        total = self.recon_weight * recon_loss + beta * kl_loss
        stats = {
            "loss": float(total.detach().item()),
            "recon_loss": float(recon_loss.detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
            "beta": float(beta),
        }
        return total, stats


class FrozenT5TextEncoder(nn.Module):
    """Frozen T5 encoder used to build text conditioning tokens."""

    def __init__(
        self,
        model_name: str = "google/t5-v1_1-large",
        max_length: int = 256,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoTokenizer, T5EncoderModel
        except ImportError as exc:
            raise ImportError("Install transformers to use FrozenT5TextEncoder") from exc

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name, torch_dtype=dtype)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.d_model)

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts: Sequence[str],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or next(self.encoder.parameters()).device
        toks = self.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = self.encoder(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"])
        return out.last_hidden_state, toks["attention_mask"]


# DiT internals: timestep embedding, 3D RoPE, and attention blocks.
class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.float()
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=timesteps.device, dtype=timesteps.dtype) / half
        )
        args = timesteps[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(-2)


class RoPE3D(nn.Module):
    """3D RoPE over the latent token grid (time, height, width)."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self._cache: Dict[Tuple[int, int, int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _build_axis(self, pos: torch.Tensor, pair_count: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if pair_count == 0:
            empty = torch.zeros(pos.shape[0], 0, device=pos.device, dtype=dtype)
            return empty, empty
        inv = torch.exp(
            -math.log(self.base) * torch.arange(pair_count, device=pos.device, dtype=dtype) / max(pair_count, 1)
        )
        ang = pos[:, None].to(dtype) * inv[None, :]
        cos = torch.repeat_interleave(torch.cos(ang), 2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(ang), 2, dim=-1)
        return cos, sin

    def get_cos_sin(
        self,
        t_hw: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (t_hw[0], t_hw[1], t_hw[2], device, dtype)
        if key in self._cache:
            return self._cache[key]

        t, h, w = t_hw
        rot_dim = (self.head_dim // 2) * 2
        pair_total = rot_dim // 2
        t_pairs = pair_total // 3
        h_pairs = pair_total // 3
        w_pairs = pair_total - t_pairs - h_pairs

        # Build token coordinates in the same traversal order as patchify.
        t_idx, h_idx, w_idx = torch.meshgrid(
            torch.arange(t, device=device),
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        t_pos = t_idx.reshape(-1)
        h_pos = h_idx.reshape(-1)
        w_pos = w_idx.reshape(-1)

        cos_t, sin_t = self._build_axis(t_pos, t_pairs, dtype)
        cos_h, sin_h = self._build_axis(h_pos, h_pairs, dtype)
        cos_w, sin_w = self._build_axis(w_pos, w_pairs, dtype)
        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)

        if rot_dim < self.head_dim:
            pad = self.head_dim - rot_dim
            cos = F.pad(cos, (0, pad), value=1.0)
            sin = F.pad(sin, (0, pad), value=0.0)

        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        self._cache[key] = (cos, sin)
        return cos, sin

    def apply(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (rotate_half(x) * sin)


class DiTBlock(nn.Module):
    """DiT block with timestep modulation and text KV concatenation."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_kv_text = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = attn_dropout

        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.modulation = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        text_emb: torch.Tensor,
        t_emb: torch.Tensor,
        rope: RoPE3D,
        token_grid_t_hw: Tuple[int, int, int],
    ) -> torch.Tensor:
        # adaLN-style shift/scale/gate values are generated from timestep embeddings.
        mod = self.modulation(F.silu(t_emb)).unsqueeze(1)
        a1, b1, g1, a2, b2, g2 = torch.chunk(mod, 6, dim=-1)

        x1 = self.norm1(x) * (1.0 + a1) + b1
        q = self._shape_heads(self.to_q(x1))
        kv_video = self.to_kv(x1)
        k_video, v_video = torch.chunk(kv_video, 2, dim=-1)
        k_video = self._shape_heads(k_video)
        v_video = self._shape_heads(v_video)

        kv_text = self.to_kv_text(text_emb)
        k_text, v_text = torch.chunk(kv_text, 2, dim=-1)
        k_text = self._shape_heads(k_text)
        v_text = self._shape_heads(v_text)

        cos, sin = rope.get_cos_sin(token_grid_t_hw, q.device, q.dtype)
        q = rope.apply(q, cos, sin)
        k_video = rope.apply(k_video, cos, sin)

        # Video queries attend to both video KV and text KV in one attention call.
        k = torch.cat([k_video, k_text], dim=2)
        v = torch.cat([v_video, v_text], dim=2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn = attn.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[1], self.dim)
        x = x + g1 * self.proj(attn)

        x2 = self.norm2(x) * (1.0 + a2) + b2
        x = x + g2 * self.ffn(x2)
        return x


class WanDiT(nn.Module):
    def __init__(
        self,
        latent_channels: int = 16,
        model_dim: int = 1536,
        depth: int = 24,
        num_heads: int = 24,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        t5_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.model_dim = model_dim
        self.patch_size = patch_size

        patch_dim = latent_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.patch_in = nn.Linear(patch_dim, model_dim)
        self.patch_out = nn.Linear(model_dim, patch_dim)
        self.in_norm = nn.LayerNorm(model_dim)
        self.out_norm = nn.LayerNorm(model_dim)

        self.timestep_embed = TimestepEmbedding(model_dim)
        self.text_proj = nn.Linear(t5_dim, model_dim)
        self.null_text_embedding = nn.Parameter(torch.zeros(1, 1, model_dim))

        self.blocks = nn.ModuleList(
            [DiTBlock(model_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.rope = RoPE3D(head_dim=model_dim // num_heads)

    def _patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        # Converts latent volume into a flat token sequence for transformer blocks.
        b, c, t, h, w = x.shape
        pt, ph, pw = self.patch_size
        if t % pt != 0 or h % ph != 0 or w % pw != 0:
            raise ValueError("latent size must be divisible by patch_size")

        tg, hg, wg = t // pt, h // ph, w // pw
        x = x.view(b, c, tg, pt, hg, ph, wg, pw)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.view(b, tg * hg * wg, pt * ph * pw * c)
        return x, (tg, hg, wg)

    def _unpatchify(self, x: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        # Restores transformer token sequence back to latent volume layout.
        b, n, _ = x.shape
        pt, ph, pw = self.patch_size
        tg, hg, wg = grid
        c = self.latent_channels
        if n != tg * hg * wg:
            raise ValueError("token count does not match grid")
        x = x.view(b, tg, hg, wg, pt, ph, pw, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x.view(b, c, tg * pt, hg * ph, wg * pw)

    def _prepare_text(
        self,
        text_embeddings: Optional[torch.Tensor],
        batch_size: int,
        cfg_dropout_prob: float,
        force_unconditional: bool,
    ) -> torch.Tensor:
        if text_embeddings is None:
            length = 1
            text_cond = self.null_text_embedding.expand(batch_size, length, -1)
            return text_cond

        if text_embeddings.shape[-1] == self.model_dim:
            text_cond = text_embeddings
        else:
            text_cond = self.text_proj(text_embeddings)

        if force_unconditional:
            return self.null_text_embedding.expand(batch_size, text_cond.size(1), -1)

        if cfg_dropout_prob <= 0.0:
            return text_cond

        # Classifier-free guidance training: randomly replace condition with null embedding.
        mask = (torch.rand(batch_size, 1, 1, device=text_cond.device) < cfg_dropout_prob).to(text_cond.dtype)
        null = self.null_text_embedding.expand(batch_size, text_cond.size(1), -1)
        return text_cond * (1.0 - mask) + null * mask

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Optional[torch.Tensor],
        cfg_dropout_prob: float = 0.0,
        force_unconditional: bool = False,
    ) -> torch.Tensor:
        b = noisy_latents.shape[0]
        tokens, grid = self._patchify(noisy_latents)
        x = self.patch_in(tokens)
        x = self.in_norm(x)

        t_emb = self.timestep_embed(timesteps)
        text_cond = self._prepare_text(
            text_embeddings=text_embeddings,
            batch_size=b,
            cfg_dropout_prob=cfg_dropout_prob,
            force_unconditional=force_unconditional,
        )

        for block in self.blocks:
            x = block(x, text_cond, t_emb, self.rope, grid)
        x = self.patch_out(self.out_norm(x))
        return self._unpatchify(x, grid)


# DDPM utilities for q(x_t|x_0), training loss, and reverse sampling.
@dataclass
class DDPMSchedule:
    num_train_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        # Precompute all scalar schedules once to avoid recomputing per step.
        device = self.device or default_device()
        self.betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_train_steps,
            device=device,
            dtype=self.dtype,
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device, dtype=self.dtype), self.alphas_cumprod[:-1]],
            dim=0,
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).clamp(min=1e-20)

    def to(self, device: torch.device) -> "DDPMSchedule":
        self.device = device
        self.__post_init__()
        return self

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        # Gather schedule values at timestep t and broadcast to latent shape.
        out = arr.index_select(0, t.long())
        while out.dim() < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def q_sample(
        self,
        clean_latents: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean_latents)
        # Forward noising process: x_t = sqrt(ab_t) * x_0 + sqrt(1-ab_t) * eps.
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, clean_latents.shape)
        sqrt_omb = self._extract(self.sqrt_one_minus_alphas_cumprod, t, clean_latents.shape)
        noisy = sqrt_ab * clean_latents + sqrt_omb * noise
        return noisy, noise


class DiffusionLoss(nn.Module):
    def forward(self, pred_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_noise, target_noise)


def diffusion_training_step(
    model: WanDiT,
    schedule: DDPMSchedule,
    clean_latents: torch.Tensor,
    text_embeddings: Optional[torch.Tensor],
    cfg_dropout_prob: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Randomly sample integer diffusion steps and train epsilon prediction.
    b = clean_latents.shape[0]
    t = torch.randint(
        0,
        schedule.num_train_steps,
        (b,),
        device=clean_latents.device,
        dtype=torch.long,
    )
    noisy_latents, target_noise = schedule.q_sample(clean_latents, t)
    t_norm = t.float() / max(schedule.num_train_steps - 1, 1)
    pred_noise = model(
        noisy_latents=noisy_latents,
        timesteps=t_norm,
        text_embeddings=text_embeddings,
        cfg_dropout_prob=cfg_dropout_prob,
    )
    loss = F.mse_loss(pred_noise, target_noise)
    return loss, {"diffusion_loss": float(loss.detach().item())}


@torch.no_grad()
def sample_with_cfg(
    model: WanDiT,
    schedule: DDPMSchedule,
    shape: Tuple[int, int, int, int, int],
    text_embeddings: Optional[torch.Tensor],
    guidance_scale: float = 6.0,
    num_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    device = device or default_device()
    z = torch.randn(shape, device=device, dtype=dtype)
    total_steps = schedule.num_train_steps if num_steps is None else num_steps
    step_indices = torch.linspace(
        schedule.num_train_steps - 1,
        0,
        total_steps,
        device=device,
    ).long()

    for i, t_step in enumerate(step_indices):
        t = torch.full((shape[0],), int(t_step.item()), device=device, dtype=torch.long)
        t_norm = t.float() / max(schedule.num_train_steps - 1, 1)

        eps_cond = model(
            noisy_latents=z,
            timesteps=t_norm,
            text_embeddings=text_embeddings,
            cfg_dropout_prob=0.0,
            force_unconditional=False,
        )
        eps_uncond = model(
            noisy_latents=z,
            timesteps=t_norm,
            text_embeddings=text_embeddings,
            cfg_dropout_prob=0.0,
            force_unconditional=True,
        )
        # Standard CFG interpolation between unconditional and conditional noise prediction.
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        alpha_t = schedule._extract(schedule.alphas, t, z.shape)
        sqrt_recip_alpha_t = schedule._extract(schedule.sqrt_recip_alphas, t, z.shape)
        sqrt_omb_t = schedule._extract(schedule.sqrt_one_minus_alphas_cumprod, t, z.shape)
        beta_t = schedule._extract(schedule.betas, t, z.shape)
        mean = sqrt_recip_alpha_t * (z - (beta_t / sqrt_omb_t) * eps)

        # Add posterior noise on all but the last reverse step.
        if i < len(step_indices) - 1:
            var_t = schedule._extract(schedule.posterior_variance, t, z.shape)
            z = mean + torch.sqrt(var_t) * torch.randn_like(z)
        else:
            z = mean
    return z


class WanVideoGenerator(nn.Module):
    """Top-level wrapper for VAE + DiT + text encoder."""

    def __init__(
        self,
        vae: WanVAE,
        dit: WanDiT,
        text_encoder: Optional[FrozenT5TextEncoder] = None,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.dit = dit
        self.text_encoder = text_encoder

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
        if self.text_encoder is None:
            raise RuntimeError("No text encoder attached")
        emb, _ = self.text_encoder.encode_prompts(prompts, device=device)
        return emb

    def forward_vae(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.vae(video)

    def forward_diffusion(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Optional[torch.Tensor],
        cfg_dropout_prob: float = 0.1,
    ) -> torch.Tensor:
        return self.dit(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            text_embeddings=text_embeddings,
            cfg_dropout_prob=cfg_dropout_prob,
        )
