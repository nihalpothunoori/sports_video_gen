#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_ffmpeg_faststart(input_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c",
        "copy",
        "-movflags",
        "faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Move MP4 moov atom to front (faststart).")
    parser.add_argument(
        "--root",
        type=str,
        default="/home/nihal/Documents/cachehacks/dataset_wan",
        help="Dataset root to scan for mp4 files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N mp4 files (0 means all).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print how many files would be processed.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    mp4_files = sorted(root.rglob("*.mp4"))
    if args.limit > 0:
        mp4_files = mp4_files[: args.limit]

    print(f"Found {len(mp4_files)} mp4 files under {root}")
    if args.dry_run:
        return

    ok = 0
    failed = 0
    for i, src in enumerate(mp4_files, start=1):
        tmp = src.with_suffix(".faststart.tmp.mp4")
        try:
            run_ffmpeg_faststart(src, tmp)
            tmp.replace(src)
            ok += 1
        except Exception as exc:
            failed += 1
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            print(f"[{i}/{len(mp4_files)}] FAIL: {src} ({exc})")
        if i % 200 == 0:
            print(f"Processed {i}/{len(mp4_files)} | ok={ok} fail={failed}")

    print(f"Done. ok={ok}, fail={failed}")


if __name__ == "__main__":
    main()
