from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any


@dataclass
class DataConfig:
    train_manifest: str = "/home/nihal/Documents/cachehacks/dataset_wan/manifest_train.jsonl"
    val_manifest: str = "/home/nihal/Documents/cachehacks/dataset_wan/manifest_val.jsonl"
    frames: int = 16
    height: int = 256
    width: int = 448
    fps: int = 16
    num_workers: int = 4
    train_fraction: float = 1.0
    val_fraction: float = 1.0


@dataclass
class ModelConfig:
    latent_channels: int = 16
    vae_base_channels: int = 64
    text_vocab_size: int = 8192
    text_max_len: int = 64
    text_dim: int = 256
    dit_dim: int = 512
    dit_depth: int = 8
    dit_heads: int = 8
    patch_size_t: int = 1
    patch_size_h: int = 2
    patch_size_w: int = 2


@dataclass
class TrainConfig:
    output_dir: str = "/home/nihal/Documents/cachehacks/checkpoints/soccer_t2v_mvp"
    batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    max_steps: int = 200000
    val_every: int = 500
    save_every: int = 500
    precision: str = "bf16"
    grad_clip_norm: float = 1.0
    seed: int = 42
    num_inference_steps: int = 30
    cfg_scale: float = 4.0
    vae_checkpoint: str = ""
    use_mu_target: bool = True
    latent_scale: float = 1.0
    kl_base_weight: float = 3e-6
    kl_start_weight: float = 0.0
    kl_warmup_steps: int = 3000
    kl_schedule: str = "sigmoid"
    kl_sigmoid_steepness: float = 10.0
    lr_scheduler: str = "cosine"
    min_lr_ratio: float = 0.1
    encoder_lr_scale: float = 0.5
    motion_focus_weight: float = 30.0
    perceptual_weight: float = 1.0
    perceptual_num_frames: int = 4


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def from_file(path: str | Path) -> "AppConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in {".json"}:
            payload: dict[str, Any] = json.loads(text)
        else:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "PyYAML is required for YAML config files. "
                    "Install with: pip install pyyaml"
                ) from exc
            payload = yaml.safe_load(text)
        return AppConfig(
            data=DataConfig(**payload.get("data", {})),
            model=ModelConfig(**payload.get("model", {})),
            train=TrainConfig(**payload.get("train", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
