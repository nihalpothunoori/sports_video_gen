#!/usr/bin/env python3
"""
STEP 3: Build train/val manifests from dataset_wan/.

For each event dir with both clip.mp4 and caption.txt:
  {"video": "/abs/path/clip.mp4", "prompt": "caption text"}

Splits 90/10 train/val, stratified by caption label.
Writes dataset_wan/manifest_train.jsonl and manifest_val.jsonl.
"""

import json, random
from collections import defaultdict
from pathlib import Path

DATASET_ROOT = Path('/home/nihal/Documents/cachehacks/dataset_wan')
TRAIN_OUT    = DATASET_ROOT / 'manifest_train.jsonl'
VAL_OUT      = DATASET_ROOT / 'manifest_val.jsonl'
VAL_RATIO    = 0.10
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)


def main() -> None:
    # Collect all valid entries, grouped by caption label for stratification
    by_label: dict[str, list[dict]] = defaultdict(list)
    skipped_no_clip    = 0
    skipped_no_caption = 0
    skipped_empty_cap  = 0

    clip_dirs = sorted(p.parent for p in DATASET_ROOT.rglob('clip.mp4'))
    print(f'Scanning {len(clip_dirs)} clip directories...')

    for event_dir in clip_dirs:
        clip_path    = event_dir / 'clip.mp4'
        caption_path = event_dir / 'caption.txt'

        if not clip_path.exists():
            print(f'  SKIP (no clip): {event_dir}')
            skipped_no_clip += 1
            continue

        if not caption_path.exists():
            print(f'  SKIP (no caption.txt): {event_dir}')
            skipped_no_caption += 1
            continue

        caption = caption_path.read_text().strip()
        if not caption:
            print(f'  SKIP (empty caption): {event_dir}')
            skipped_empty_cap += 1
            continue

        entry = {
            'video':  str(clip_path.resolve()),
            'prompt': caption,
        }
        by_label[caption].append(entry)

    total = sum(len(v) for v in by_label.values())
    print(f'\nValid entries: {total}')
    print(f'Skipped — no clip: {skipped_no_clip}, no caption: {skipped_no_caption}, empty caption: {skipped_empty_cap}')

    # Per-class counts
    print('\nPer-class counts:')
    for label, entries in sorted(by_label.items(), key=lambda x: -len(x[1])):
        print(f'  {label:<30} {len(entries):>5}')

    # Stratified 90/10 split
    train_entries: list[dict] = []
    val_entries:   list[dict] = []

    for label, entries in by_label.items():
        random.shuffle(entries)
        n_val = max(1, round(len(entries) * VAL_RATIO))
        val_entries.extend(entries[:n_val])
        train_entries.extend(entries[n_val:])

    # Shuffle final lists so they're not label-sorted
    random.shuffle(train_entries)
    random.shuffle(val_entries)

    # Write manifests
    TRAIN_OUT.write_text('\n'.join(json.dumps(e) for e in train_entries) + '\n')
    VAL_OUT.write_text('\n'.join(json.dumps(e) for e in val_entries) + '\n')

    print(f'\nTrain: {len(train_entries)} entries → {TRAIN_OUT}')
    print(f'Val:   {len(val_entries)} entries → {VAL_OUT}')

    # Show first 5 lines of each
    print('\n--- manifest_train.jsonl (first 5) ---')
    for line in TRAIN_OUT.read_text().splitlines()[:5]:
        print(' ', line)

    print('\n--- manifest_val.jsonl (first 5) ---')
    for line in VAL_OUT.read_text().splitlines()[:5]:
        print(' ', line)

    # Val class distribution (sanity check)
    val_dist: dict[str, int] = defaultdict(int)
    for e in val_entries:
        val_dist[e['prompt']] += 1
    print('\nVal class distribution:')
    for label, cnt in sorted(val_dist.items(), key=lambda x: -x[1]):
        pct = cnt / len(val_entries) * 100
        print(f'  {label:<30} {cnt:>4}  ({pct:.1f}%)')


if __name__ == '__main__':
    main()
