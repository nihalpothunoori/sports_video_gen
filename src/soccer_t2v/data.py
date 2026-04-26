from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video

from soccer_t2v.config import DataConfig

BAD_VIDEO_LOG = Path("/home/nihal/Documents/cachehacks/logs/bad_videos.log")


ACTION_TEMPLATES = [
    "Show a soccer clip where the key action is {action}.",
    "Counterfactual soccer sequence: the player performs a {action}.",
    "Generate a realistic football play ending with a {action}.",
    "Tactical scenario in soccer with a decisive {action}.",
]


@dataclass
class Sample:
    video: torch.Tensor
    prompt: str
    action: str
    video_path: str


def _read_manifest(path: str | Path, fraction: float = 1.0, seed: int = 42) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    fraction = float(max(0.0, min(1.0, fraction)))
    if fraction <= 0.0:
        return []
    if fraction < 1.0:
        rng = random.Random(seed)
        rng.shuffle(rows)
        keep = max(1, int(len(rows) * fraction))
        rows = rows[:keep]
    return rows


def _normalize_action(action: str) -> str:
    return action.strip().lower()


def _make_prompt(action: str) -> str:
    template = random.choice(ACTION_TEMPLATES)
    return template.format(action=action)


def _resize_video(video_tchw: torch.Tensor, frames: int, height: int, width: int) -> torch.Tensor:
    if video_tchw.shape[0] < frames:
        pad_count = frames - video_tchw.shape[0]
        padding = video_tchw[-1:].repeat(pad_count, 1, 1, 1)
        video_tchw = torch.cat([video_tchw, padding], dim=0)
    elif video_tchw.shape[0] > frames:
        idx = torch.linspace(0, video_tchw.shape[0] - 1, frames).long()
        video_tchw = video_tchw[idx]

    video_cthw = video_tchw.permute(1, 0, 2, 3).contiguous().unsqueeze(0)
    video_cthw = F.interpolate(
        video_cthw,
        size=(frames, height, width),
        mode="trilinear",
        align_corners=False,
    )
    return video_cthw.squeeze(0)


class SoccerVideoDataset(Dataset[Sample]):
    def __init__(
        self,
        manifest_path: str | Path,
        cfg: DataConfig,
        deterministic_prompt: bool = False,
        fraction: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.rows = _read_manifest(manifest_path, fraction=fraction, seed=seed)
        self.cfg = cfg
        self.deterministic_prompt = deterministic_prompt

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _log_bad_video(path: str, error: Exception) -> None:
        BAD_VIDEO_LOG.parent.mkdir(parents=True, exist_ok=True)
        with BAD_VIDEO_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{path}\t{type(error).__name__}\t{error}\n")

    def __getitem__(self, idx: int) -> Sample:
        # Some clips may be partially corrupted; retry with a different sample.
        last_error: Optional[Exception] = None
        for attempt in range(8):
            row = self.rows[idx]
            path = row["video"]
            action = _normalize_action(row["prompt"])
            try:
                video, _, _ = read_video(path, pts_unit="sec")
                if video.numel() == 0 or video.shape[0] == 0:
                    raise ValueError("Empty decoded video frames")
                # torchvision returns [T, H, W, C] uint8
                video = video.float() / 127.5 - 1.0
                video = video.permute(0, 3, 1, 2).contiguous()  # [T,C,H,W]
                video = _resize_video(video, self.cfg.frames, self.cfg.height, self.cfg.width)
                prompt = (
                    f"Soccer tactical counterfactual: {action}."
                    if self.deterministic_prompt
                    else _make_prompt(action)
                )
                return Sample(video=video, prompt=prompt, action=action, video_path=path)
            except Exception as exc:
                last_error = exc
                self._log_bad_video(path, exc)
                if attempt == 7:
                    raise RuntimeError(f"Failed loading video after retries: {path}") from exc
                idx = random.randint(0, len(self.rows) - 1)
        raise RuntimeError("Failed to load sample after retries") from last_error


def collate_samples(samples: list[Sample]) -> dict[str, torch.Tensor | list[str]]:
    video = torch.stack([s.video for s in samples], dim=0)
    prompt = [s.prompt for s in samples]
    action = [s.action for s in samples]
    video_path = [s.video_path for s in samples]
    return {
        "video": video,
        "prompt": prompt,
        "action": action,
        "video_path": video_path,
    }


def create_dataloaders(cfg: DataConfig, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_ds = SoccerVideoDataset(
        cfg.train_manifest, cfg, fraction=cfg.train_fraction, seed=42
    )
    val_ds = SoccerVideoDataset(
        cfg.val_manifest, cfg, deterministic_prompt=True, fraction=cfg.val_fraction, seed=43
    )
    train_worker_args = {}
    val_worker_args = {}
    if cfg.num_workers > 0:
        train_worker_args = {"persistent_workers": True, "prefetch_factor": 4}
        val_worker_args = {"persistent_workers": True, "prefetch_factor": 2}

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_samples,
        drop_last=True,
        **train_worker_args,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_samples,
        drop_last=False,
        **val_worker_args,
    )
    return train_loader, val_loader
