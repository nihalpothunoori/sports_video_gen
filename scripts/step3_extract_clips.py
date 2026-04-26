#!/usr/bin/env python3
"""
STEP 3: Dry-run clip extraction for match 128058 (smallest match).

For each BAS event:
  - Seek to (event_time - 2.5s) in the correct half's panorama
  - Extract 5s @ 832x480 / 16fps / h264
  - Slice GSR tracking window
  - Write clip.mp4, caption.txt, tracking.json, bas_event.json,
    metadata.json into dataset_wan/{match}/{label}/{event_id}/
  - mapx.npy / mapy.npy / homography.npy stored once at match level
    with relative symlinks from each event dir

Run:  source venv/bin/activate && python3 scripts/step3_extract_clips.py
"""

import json, os, sys, shutil, subprocess, time, logging
from decimal import Decimal
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import ijson
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path('/home/nihal/Documents/cachehacks/soccer_track_data')
OUTPUT_ROOT = Path('/home/nihal/Documents/cachehacks/dataset_wan')
LOG_DIR     = Path('/home/nihal/Documents/cachehacks/logs')
MATCH_ID    = '128058'

CLIP_DURATION  = 5.0    # seconds
HALF_MARGIN    = 2.5    # reject events within this many seconds of half boundary
VIDEO_FPS      = 25
CLIP_FPS       = 16
CLIP_W, CLIP_H = 832, 480
N_WORKERS      = 4      # parallel ffmpeg processes

HOMOGRAPHY_FILES = ['mapx.npy', 'mapy.npy', 'homography.npy']  # stored at match level


class DecimalEncoder(json.JSONEncoder):
    """ijson yields Decimal for all JSON numbers; convert to float for output."""
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'step3_{MATCH_ID}.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def disk_free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def get_video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fc / fps if fps > 0 else 0.0


def sanitize_label(label: str) -> str:
    return label.replace(' ', '_').replace('/', '_')


def action_to_half_and_seek(action: dict, half2_offset_ms: int) -> tuple[str, float]:
    """Return (half_key='1'|'2', seek_time_s_in_half_video)."""
    gt      = action['gameTime']
    pos_ms  = int(action['position'])
    if gt.startswith('1 - '):
        return '1', pos_ms / 1000.0
    # half 2 or extra-time (90+ min with no "2 - " prefix)
    return '2', (pos_ms - half2_offset_ms) / 1000.0


def extract_clip(video_path: Path, clip_start: float, output_path: Path) -> bool:
    """
    Use keyframe-level pre-seek then accurate duration trim.
    Returns True if output file exists and is non-empty.
    """
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{clip_start:.3f}',
        '-i', str(video_path),
        '-t', f'{CLIP_DURATION:.3f}',
        '-vf', f'scale={CLIP_W}:{CLIP_H}',
        '-r', str(CLIP_FPS),
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'fast',
        '-an',
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        return False
    return (result.returncode == 0
            and output_path.exists()
            and output_path.stat().st_size > 0)


def stream_gsr_for_half(
    gsr_path: Path,
    frame_to_events: dict[int, list[str]],
) -> dict[str, dict]:
    """
    Stream a 2GB+ GSR JSON in two passes (images then annotations),
    collecting only records whose image_id falls in the clip windows.

    Returns: {event_id: {'images': [...], 'annotations': [...]}}
    """
    needed_ids: set[str] = {f'3{fn:06d}' for fn in frame_to_events}
    tracking: dict[str, dict] = {}  # built lazily

    def _ensure(eid: str) -> dict:
        if eid not in tracking:
            tracking[eid] = {'images': [], 'annotations': []}
        return tracking[eid]

    log.info(f'  GSR pass 1 (images): {gsr_path.name}  ({len(needed_ids):,} frames of interest)')
    t0 = time.time()
    with open(gsr_path, 'rb') as f:
        for img in ijson.items(f, 'images.item'):
            iid = img['image_id']
            if iid in needed_ids:
                fn = int(iid[1:])   # "3000042" → 42
                for eid in frame_to_events.get(fn, []):
                    _ensure(eid)['images'].append(img)
    log.info(f'  images pass done in {time.time()-t0:.1f}s')

    log.info(f'  GSR pass 2 (annotations): {gsr_path.name}')
    t0 = time.time()
    with open(gsr_path, 'rb') as f:
        for ann in ijson.items(f, 'annotations.item'):
            iid = ann['image_id']
            if iid in needed_ids:
                fn = int(iid[1:])
                for eid in frame_to_events.get(fn, []):
                    _ensure(eid)['annotations'].append(ann)
    log.info(f'  annotations pass done in {time.time()-t0:.1f}s')

    return tracking


def write_event_files(
    e: dict,
    out_dir: Path,
    tracking_data: dict,
    match_out_dir: Path,
) -> bool:
    """Write all sidecar files for one event. Returns True on success."""
    eid = e['event_id']

    # caption placeholder
    (out_dir / 'caption.txt').write_text('')

    # tracking window (use DecimalEncoder because ijson yields Decimal numbers)
    (out_dir / 'tracking.json').write_text(
        json.dumps(tracking_data.get(eid, {'images': [], 'annotations': []}),
                   cls=DecimalEncoder, ensure_ascii=False)
    )

    # raw BAS action
    (out_dir / 'bas_event.json').write_text(
        json.dumps(e['action'], ensure_ascii=False)
    )

    # metadata
    (out_dir / 'metadata.json').write_text(json.dumps({
        'match_id':      MATCH_ID,
        'event_id':      eid,
        'event_class':   e['action']['label'],
        'half':          e['half'],
        'seek_time_s':   e['seek_s'],
        'clip_start_s':  e['clip_start'],
        'team':          e['action'].get('team', ''),
        'player_id':     e['action'].get('player_id', ''),
        'global_bas_idx': e['global_idx'],
    }, ensure_ascii=False, indent=2))

    # relative symlinks for shared homography arrays
    # event dir depth:  dataset_wan/{match}/{label}/{event_id}/  → 3 levels up
    for fname in HOMOGRAPHY_FILES:
        src = match_out_dir / fname
        if not src.exists():
            continue
        dst = out_dir / fname
        if not dst.exists():
            # relative path from event dir to match dir (event_id/label/match_id)
            rel = Path('../..') / fname
            try:
                dst.symlink_to(rel)
            except OSError:
                pass   # already exists or unsupported

    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info(f'=== STEP 3: clip extraction for match {MATCH_ID} ===')

    # Disk guard
    free_gb = disk_free_gb(OUTPUT_ROOT)
    if free_gb < 100:
        sys.exit(f'STOP: only {free_gb:.1f}GB free (need ≥100GB)')
    log.info(f'Disk free: {free_gb:.1f}GB  ✓')

    # Load BAS events
    bas_path = DATA_ROOT / 'bas' / MATCH_ID / f'{MATCH_ID}_12_class_events.json'
    with open(bas_path) as f:
        bas_data = json.load(f)
    actions = bas_data['actions']
    log.info(f'BAS events loaded: {len(actions)}')

    # Video durations
    half_dur: dict[str, float] = {}
    for half_key, label in [('1', '1st'), ('2', '2nd')]:
        vp = DATA_ROOT / 'videos' / MATCH_ID / f'{MATCH_ID}_panorama_{label}_half.mp4'
        d = get_video_duration(vp)
        half_dur[half_key] = d
        log.info(f'  Half {half_key} duration: {d:.1f}s ({d/60:.2f} min)')

    # Half-2 game-time offset (min position_ms among all half-2 events)
    h2_events = [a for a in actions if a['gameTime'].startswith('2 - ')]
    half2_offset_ms = min(int(a['position']) for a in h2_events) if h2_events else 0
    log.info(f'Half-2 offset: {half2_offset_ms}ms = {half2_offset_ms/1000:.1f}s')

    # Build valid event list (reject boundary events)
    valid_events: list[dict] = []
    skipped_boundary = 0
    for idx, action in enumerate(actions):
        half, seek_s = action_to_half_and_seek(action, half2_offset_ms)
        clip_start = seek_s - HALF_MARGIN
        clip_end   = seek_s + HALF_MARGIN
        if clip_start < 0 or clip_end > half_dur[half]:
            skipped_boundary += 1
            continue
        event_id = f'{MATCH_ID}_h{half}_{idx:05d}'
        label    = sanitize_label(action['label'])
        valid_events.append({
            'event_id':   event_id,
            'half':       half,
            'seek_s':     seek_s,
            'clip_start': clip_start,
            'label':      label,
            'action':     action,
            'global_idx': idx,
        })
    log.info(f'Valid events: {len(valid_events)}, skipped (boundary): {skipped_boundary}')

    # Set up match-level output dir and copy homography files once
    match_out_dir = OUTPUT_ROOT / MATCH_ID
    match_out_dir.mkdir(parents=True, exist_ok=True)
    for fname in HOMOGRAPHY_FILES:
        raw_src = DATA_ROOT / 'raw' / MATCH_ID / f'{MATCH_ID}_{fname}'
        dst     = match_out_dir / fname
        if raw_src.exists() and not dst.exists():
            shutil.copy2(raw_src, dst)
            log.info(f'Copied {fname} to match dir ({raw_src.stat().st_size/1e6:.1f}MB)')

    # Process each half: stream GSR, then extract clips in parallel
    total_done = 0
    total_skip = 0  # already extracted (resume)
    total_fail = 0

    events_by_half: dict[str, list[dict]] = defaultdict(list)
    for e in valid_events:
        events_by_half[e['half']].append(e)

    for half_key in ['1', '2']:
        half_events = events_by_half.get(half_key, [])
        if not half_events:
            continue
        half_label = '1st' if half_key == '1' else '2nd'
        video_path = DATA_ROOT / 'videos' / MATCH_ID / f'{MATCH_ID}_panorama_{half_label}_half.mp4'
        gsr_path   = DATA_ROOT / 'gsr'    / MATCH_ID / f'{MATCH_ID}_{half_label}.json'

        log.info(f'\n── Half {half_key}: {len(half_events)} events ──')

        # Build frame→[event_id] map for clip windows
        clip_half_frames = int(HALF_MARGIN * VIDEO_FPS) + 1  # 63 frames each side
        frame_to_events: dict[int, list[str]] = defaultdict(list)
        for e in half_events:
            center = int(e['seek_s'] * VIDEO_FPS) + 1
            for fn in range(max(1, center - clip_half_frames),
                            center + clip_half_frames + 2):
                frame_to_events[fn].append(e['event_id'])

        # Categorise events: fully done / clip-only (sidecar missing) / pending
        fully_done, sidecar_only, pending = [], [], []
        for e in half_events:
            out_dir  = OUTPUT_ROOT / MATCH_ID / e['label'] / e['event_id']
            clip_ok  = (out_dir / 'clip.mp4').exists() and (out_dir / 'clip.mp4').stat().st_size > 0
            meta_ok  = (out_dir / 'metadata.json').exists()
            if clip_ok and meta_ok:
                fully_done.append(e)
            elif clip_ok and not meta_ok:
                sidecar_only.append(e)  # clip exists but sidecars missing
            else:
                pending.append(e)

        if fully_done:
            log.info(f'  Fully done: {len(fully_done)}, sidecar-only: {len(sidecar_only)}, pending: {len(pending)}')
            total_done += len(fully_done)
            total_skip += len(fully_done)
            # Remove fully-done events from GSR frame map
            done_ids = {e['event_id'] for e in fully_done}
            for fn in list(frame_to_events.keys()):
                frame_to_events[fn] = [eid for eid in frame_to_events[fn]
                                        if eid not in done_ids]
                if not frame_to_events[fn]:
                    del frame_to_events[fn]

        needs_gsr = pending + sidecar_only
        if not needs_gsr:
            log.info('  All events complete, skipping GSR stream.')
            continue

        # Stream GSR for events that need sidecar files
        event_tracking = stream_gsr_for_half(gsr_path, frame_to_events)

        # Write sidecars for clip-only events (no re-extraction needed)
        if sidecar_only:
            log.info(f'  Writing missing sidecars for {len(sidecar_only)} events...')
            for e in sidecar_only:
                out_dir = OUTPUT_ROOT / MATCH_ID / e['label'] / e['event_id']
                write_event_files(e, out_dir, event_tracking, match_out_dir)
                total_done += 1

        # Extract clips in parallel for fully pending events
        def process_event(e: dict) -> tuple[str, bool]:
            eid     = e['event_id']
            out_dir = OUTPUT_ROOT / MATCH_ID / e['label'] / eid
            out_dir.mkdir(parents=True, exist_ok=True)
            clip_path = out_dir / 'clip.mp4'
            ok = extract_clip(video_path, e['clip_start'], clip_path)
            if ok:
                write_event_files(e, out_dir, event_tracking, match_out_dir)
            return eid, ok

        if not pending:
            log.info('  No clips to extract (all sidecar-only repairs done).')
            continue

        log.info(f'  Extracting {len(pending)} clips with {N_WORKERS} workers...')
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(process_event, e): e for e in pending}
            for fut in tqdm(as_completed(futures), total=len(pending),
                            desc=f'Half {half_key} clips'):
                eid, ok = fut.result()
                if ok:
                    total_done += 1
                else:
                    total_fail += 1
                    log.warning(f'  FAILED: {eid}')

    # Summary
    log.info('\n=== STEP 3 Summary ===')
    log.info(f'Total extracted:    {total_done}')
    log.info(f'  Already done:     {total_skip}')
    log.info(f'Failed:             {total_fail}')
    log.info(f'Boundary-skipped:   {skipped_boundary}')

    # Validate: check a few random clips exist and are non-empty
    import random
    samples = random.sample(valid_events, min(5, len(valid_events)))
    log.info('\nSpot-check (5 random clips):')
    for e in samples:
        p = OUTPUT_ROOT / MATCH_ID / e['label'] / e['event_id'] / 'clip.mp4'
        sz = p.stat().st_size / 1e6 if p.exists() else 0.0
        status = 'OK' if sz > 0 else 'MISSING'
        log.info(f'  [{status}] {p.name} in {e["label"]}/  ({sz:.1f}MB)')

    # Dataset size on disk
    total_bytes = sum(
        f.stat().st_size
        for f in (OUTPUT_ROOT / MATCH_ID).rglob('clip.mp4')
        if f.is_file()
    )
    log.info(f'\nTotal clip storage: {total_bytes/1e9:.2f}GB')
    log.info(f'Output: {OUTPUT_ROOT / MATCH_ID}')
    log.info('\nSTEP 3 complete. Awaiting approval before STEP 4.')


if __name__ == '__main__':
    main()
