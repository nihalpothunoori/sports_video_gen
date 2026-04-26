#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soccer_t2v.config import AppConfig
from soccer_t2v.data import create_dataloaders


def main() -> None:
    cfg = AppConfig.from_file(ROOT / "configs" / "soccer_t2v_mvp.yaml")
    cfg.data.num_workers = 0
    train_loader, val_loader = create_dataloaders(cfg.data, cfg.train.batch_size)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("train/video:", train_batch["video"].shape)
    print("train/prompt[0]:", train_batch["prompt"][0])
    print("val/video:", val_batch["video"].shape)
    print("val/prompt[0]:", val_batch["prompt"][0])


if __name__ == "__main__":
    main()
