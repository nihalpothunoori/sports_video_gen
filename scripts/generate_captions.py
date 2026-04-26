#!/usr/bin/env python3
"""
STEP 2: Generate one-line caption.txt files from metadata.json event_class.

Caption = core action word(s) in plain English, all lowercase.
Structural prefixes (BALL_PLAYER_, PLAYER_SUCCESSFUL_) are stripped.
Meaningful modifiers (HIGH, FREE) are kept.

Idempotent: skips dirs that already have caption.txt.
"""

import json
from pathlib import Path

DATASET_ROOT = Path('/home/nihal/Documents/cachehacks/dataset_wan')

# Explicit map for all 12 SoccerTrack action classes.
# Keeps all 12 distinct; strips structural noise, keeps semantic content.
LABEL_MAP: dict[str, str] = {
    'PASS':                     'pass',
    'HIGH_PASS':                'high pass',
    'DRIVE':                    'drive',
    'CROSS':                    'cross',
    'SHOT':                     'shot',
    'HEADER':                   'header',
    'FREE_KICK':                'free kick',
    'THROW_IN':                 'throw in',
    'GOAL':                     'goal',
    'OUT':                      'out',
    'BALL_PLAYER_BLOCK':        'block',
    'PLAYER_SUCCESSFUL_TACKLE': 'tackle',
}


def label_to_caption(raw: str) -> str:
    # Normalize: uppercase + underscores (metadata may store spaces or underscores)
    key = raw.strip().upper().replace(' ', '_')
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    # Fallback: lowercase + spaces (handles any future labels)
    return raw.strip().lower().replace('_', ' ')


def main() -> None:
    written = 0
    already_existed = 0
    skipped_no_meta = 0
    skipped_no_clip = 0
    unknown_labels: set[str] = set()

    # Walk every directory that contains clip.mp4
    clip_dirs = sorted(
        p.parent for p in DATASET_ROOT.rglob('clip.mp4')
    )
    total = len(clip_dirs)
    print(f'Found {total} clip directories under {DATASET_ROOT}')

    for i, event_dir in enumerate(clip_dirs, 1):
        caption_path = event_dir / 'caption.txt'

        # Skip only if caption exists and already matches what we'd write
        if caption_path.exists():
            existing = caption_path.read_text().strip()
            if existing:
                # Peek at what we'd write; skip if already correct
                meta_path_peek = event_dir / 'metadata.json'
                if meta_path_peek.exists():
                    try:
                        peek = json.loads(meta_path_peek.read_text())
                        raw_peek = peek.get('event_class') or peek.get('label', '')
                        expected = label_to_caption(raw_peek) if raw_peek else ''
                        if existing == expected:
                            already_existed += 1
                            if i % 500 == 0:
                                print(f'  [{i}/{total}] ok: "{existing}" ({event_dir.name})')
                            continue
                    except Exception:
                        pass  # fall through to rewrite

        # Need clip (sanity) and metadata
        meta_path = event_dir / 'metadata.json'
        if not meta_path.exists():
            print(f'  SKIP (no metadata.json): {event_dir}')
            skipped_no_meta += 1
            continue

        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            print(f'  SKIP (bad metadata.json): {event_dir} — {e}')
            skipped_no_meta += 1
            continue

        # event_class field (fallback to label for older formats)
        raw_label = meta.get('event_class') or meta.get('label', '')
        if not raw_label:
            print(f'  SKIP (no event_class in metadata): {event_dir}')
            skipped_no_meta += 1
            continue

        caption = label_to_caption(raw_label)

        # Track any labels not in our explicit map
        if raw_label.strip().upper() not in LABEL_MAP:
            unknown_labels.add(raw_label)

        caption_path.write_text(caption + '\n')
        written += 1

        if i % 100 == 0:
            print(f'  [{i}/{total}] wrote "{caption}" ← {raw_label}  ({event_dir.name})')

    print()
    print('=== Caption Generation Summary ===')
    print(f'  Written:         {written}')
    print(f'  Already existed: {already_existed}')
    print(f'  Skipped (no meta/clip): {skipped_no_meta}')
    if unknown_labels:
        print(f'  Unknown labels (used fallback): {sorted(unknown_labels)}')

    # Show the caption→count distribution
    caption_counts: dict[str, int] = {}
    for p in DATASET_ROOT.rglob('caption.txt'):
        c = p.read_text().strip()
        caption_counts[c] = caption_counts.get(c, 0) + 1
    print()
    print('Caption distribution:')
    for cap, count in sorted(caption_counts.items(), key=lambda x: -x[1]):
        print(f'  {cap:<30} {count:>5}')


if __name__ == '__main__':
    main()
