#!/usr/bin/env python3
"""
STEP 4: Scale clip extraction to all 10 SoccerTrack v2 matches.

Processes each match sequentially (two GSR passes per half).
Match 128058 (already done) is handled via resume logic — clips and
sidecar files already on disk are detected and skipped.

Run:  source venv/bin/activate && python3 scripts/step4_extract_all.py
"""

import json, os, sys, shutil, subprocess, time, logging
from decimal import Decimal
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import ijson
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path('/home/nihal/Documents/cachehacks/soccer_track_data')
OUTPUT_ROOT = Path('/home/nihal/Documents/cachehacks/dataset_wan')
LOG_DIR     = Path('/home/nihal/Documents/cachehacks/logs')

ALL_MATCHES = [
    '117092', '117093', '118575', '118576', '118577',
    '118578', '128057', '128058', '132831', '132877',
]

CLIP_DURATION  = 5.0
HALF_MARGIN    = 2.5
VIDEO_FPS      = 25
CLIP_FPS       = 16
CLIP_W, CLIP_H = 832, 480
N_WORKERS      = 4

HOMOGRAPHY_FILES = ['mapx.npy', 'mapy.npy', 'homography.npy']

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'step4_all_matches.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────────────

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


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
    gt     = action['gameTime']
    pos_ms = int(action['position'])
    if gt.startswith('1 - '):
        return '1', pos_ms / 1000.0
    return '2', (pos_ms - half2_offset_ms) / 1000.0


def extract_clip(video_path: Path, clip_start: float, output_path: Path) -> bool:
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{clip_start:.3f}',
        '-i', str(video_path),
        '-t', f'{CLIP_DURATION:.3f}',
        '-vf', f'scale={CLIP_W}:{CLIP_H}',
        '-r', str(CLIP_FPS),
        '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
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


def write_event_files(
    e: dict, out_dir: Path, tracking_data: dict, match_out_dir: Path
) -> None:
    eid = e['event_id']
    (out_dir / 'caption.txt').write_text('')
    (out_dir / 'tracking.json').write_text(
        json.dumps(tracking_data.get(eid, {'images': [], 'annotations': []}),
                   cls=DecimalEncoder, ensure_ascii=False)
    )
    (out_dir / 'bas_event.json').write_text(
        json.dumps(e['action'], ensure_ascii=False)
    )
    (out_dir / 'metadata.json').write_text(json.dumps({
        'match_id':       e['match_id'],
        'event_id':       eid,
        'event_class':    e['action']['label'],
        'half':           e['half'],
        'seek_time_s':    e['seek_s'],
        'clip_start_s':   e['clip_start'],
        'team':           e['action'].get('team', ''),
        'player_id':      e['action'].get('player_id', ''),
        'global_bas_idx': e['global_idx'],
    }, ensure_ascii=False, indent=2))
    # Relative symlinks to match-level homography files
    # Depth: dataset_wan/{match}/{label}/{event_id}/  => ../../{file}
    for fname in HOMOGRAPHY_FILES:
        src = match_out_dir / fname
        if not src.exists():
            continue
        dst = out_dir / fname
        if not dst.exists():
            try:
                dst.symlink_to(Path('../..') / fname)
            except OSError:
                pass


def stream_gsr_for_half(
    gsr_path: Path, frame_to_events: dict[int, list[str]]
) -> dict[str, dict]:
    needed_ids: set[str] = {f'3{fn:06d}' for fn in frame_to_events}
    tracking: dict[str, dict] = {}

    def _ensure(eid: str) -> dict:
        if eid not in tracking:
            tracking[eid] = {'images': [], 'annotations': []}
        return tracking[eid]

    log.info(f'    GSR pass 1 (images): {gsr_path.name}  ({len(needed_ids):,} frames)')
    t0 = time.time()
    with open(gsr_path, 'rb') as f:
        for img in ijson.items(f, 'images.item'):
            iid = img['image_id']
            if iid in needed_ids:
                fn = int(iid[1:])
                for eid in frame_to_events.get(fn, []):
                    _ensure(eid)['images'].append(img)
    log.info(f'    images done in {time.time()-t0:.1f}s')

    log.info(f'    GSR pass 2 (annotations): {gsr_path.name}')
    t0 = time.time()
    with open(gsr_path, 'rb') as f:
        for ann in ijson.items(f, 'annotations.item'):
            iid = ann['image_id']
            if iid in needed_ids:
                fn = int(iid[1:])
                for eid in frame_to_events.get(fn, []):
                    _ensure(eid)['annotations'].append(ann)
    log.info(f'    annotations done in {time.time()-t0:.1f}s')

    return tracking


# ── Per-match processing ──────────────────────────────────────────────────────

def process_match(match_id: str) -> dict:
    """
    Extract clips for one match. Returns a stats dict.
    """
    log.info(f'\n{"="*60}')
    log.info(f'Match {match_id}')
    log.info(f'{"="*60}')

    stats = {
        'match_id': match_id,
        'total_bas': 0, 'valid': 0, 'boundary_skip': 0,
        'done': 0, 'skipped_resume': 0, 'failed': 0,
        'size_gb': 0.0,
    }

    # ── BAS ──────────────────────────────────────────────────────────────────
    bas_path = DATA_ROOT / 'bas' / match_id / f'{match_id}_12_class_events.json'
    with open(bas_path) as f:
        bas_data = json.load(f)
    actions = bas_data['actions']
    stats['total_bas'] = len(actions)
    log.info(f'  BAS events: {len(actions)}')

    # ── Video durations ───────────────────────────────────────────────────────
    half_dur: dict[str, float] = {}
    for half_key, label in [('1', '1st'), ('2', '2nd')]:
        vp = DATA_ROOT / 'videos' / match_id / f'{match_id}_panorama_{label}_half.mp4'
        d = get_video_duration(vp)
        half_dur[half_key] = d
        log.info(f'  Half {half_key}: {d:.1f}s ({d/60:.2f} min)')

    # ── Half-2 offset ─────────────────────────────────────────────────────────
    h2_evts = [a for a in actions if a['gameTime'].startswith('2 - ')]
    half2_offset_ms = min(int(a['position']) for a in h2_evts) if h2_evts else 0

    # ── Build valid events ────────────────────────────────────────────────────
    valid_events: list[dict] = []
    for idx, action in enumerate(actions):
        half, seek_s = action_to_half_and_seek(action, half2_offset_ms)
        clip_start = seek_s - HALF_MARGIN
        clip_end   = seek_s + HALF_MARGIN
        if clip_start < 0 or clip_end > half_dur[half]:
            stats['boundary_skip'] += 1
            continue
        valid_events.append({
            'match_id':   match_id,
            'event_id':   f'{match_id}_h{half}_{idx:05d}',
            'half':       half,
            'seek_s':     seek_s,
            'clip_start': clip_start,
            'label':      sanitize_label(action['label']),
            'action':     action,
            'global_idx': idx,
        })
    stats['valid'] = len(valid_events)
    log.info(f'  Valid: {len(valid_events)}, boundary-skipped: {stats["boundary_skip"]}')

    # ── Match-level output dir & homography files ─────────────────────────────
    match_out_dir = OUTPUT_ROOT / match_id
    match_out_dir.mkdir(parents=True, exist_ok=True)
    for fname in HOMOGRAPHY_FILES:
        raw_src = DATA_ROOT / 'raw' / match_id / f'{match_id}_{fname}'
        dst     = match_out_dir / fname
        if raw_src.exists() and not dst.exists():
            shutil.copy2(raw_src, dst)

    # ── Process by half ───────────────────────────────────────────────────────
    events_by_half: dict[str, list[dict]] = defaultdict(list)
    for e in valid_events:
        events_by_half[e['half']].append(e)

    for half_key in ['1', '2']:
        half_events = events_by_half.get(half_key, [])
        if not half_events:
            continue
        half_label = '1st' if half_key == '1' else '2nd'
        video_path = DATA_ROOT / 'videos' / match_id / f'{match_id}_panorama_{half_label}_half.mp4'
        gsr_path   = DATA_ROOT / 'gsr'    / match_id / f'{match_id}_{half_label}.json'

        log.info(f'  Half {half_key}: {len(half_events)} events')

        # Categorise: fully done / sidecar-only (clip exists, no metadata) / pending
        fully_done, sidecar_only, pending = [], [], []
        for e in half_events:
            out_dir  = OUTPUT_ROOT / match_id / e['label'] / e['event_id']
            clip_ok  = (out_dir / 'clip.mp4').exists() and (out_dir / 'clip.mp4').stat().st_size > 0
            meta_ok  = (out_dir / 'metadata.json').exists()
            if clip_ok and meta_ok:
                fully_done.append(e)
            elif clip_ok and not meta_ok:
                sidecar_only.append(e)
            else:
                pending.append(e)

        stats['done']          += len(fully_done)
        stats['skipped_resume'] += len(fully_done)
        if fully_done:
            log.info(f'    Fully done: {len(fully_done)}, sidecar-only: {len(sidecar_only)}, pending: {len(pending)}')

        needs_gsr = pending + sidecar_only
        if not needs_gsr:
            log.info(f'    All {len(fully_done)} events already complete.')
            continue

        # Build frame→events map (only for events that need GSR data)
        clip_half_frames = int(HALF_MARGIN * VIDEO_FPS) + 1
        frame_to_events: dict[int, list[str]] = defaultdict(list)
        for e in needs_gsr:
            center = int(e['seek_s'] * VIDEO_FPS) + 1
            for fn in range(max(1, center - clip_half_frames),
                            center + clip_half_frames + 2):
                frame_to_events[fn].append(e['event_id'])

        event_tracking = stream_gsr_for_half(gsr_path, frame_to_events)

        # Repair sidecar-only events first (no clip re-extraction)
        for e in sidecar_only:
            out_dir = OUTPUT_ROOT / match_id / e['label'] / e['event_id']
            write_event_files(e, out_dir, event_tracking, match_out_dir)
            stats['done'] += 1

        # Extract + write pending events in parallel
        def process_event(e: dict) -> tuple[str, bool]:
            eid     = e['event_id']
            out_dir = OUTPUT_ROOT / match_id / e['label'] / eid
            out_dir.mkdir(parents=True, exist_ok=True)
            clip_path = out_dir / 'clip.mp4'
            ok = extract_clip(video_path, e['clip_start'], clip_path)
            if ok:
                write_event_files(e, out_dir, event_tracking, match_out_dir)
            return eid, ok

        if pending:
            log.info(f'    Extracting {len(pending)} clips ({N_WORKERS} workers)...')
            with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                futures = {pool.submit(process_event, e): e for e in pending}
                for fut in tqdm(as_completed(futures), total=len(pending),
                                desc=f'  {match_id} half {half_key}',
                                leave=False):
                    eid, ok = fut.result()
                    if ok:
                        stats['done'] += 1
                    else:
                        stats['failed'] += 1
                        log.warning(f'    FAILED: {eid}')

    # ── Per-match size ────────────────────────────────────────────────────────
    stats['size_gb'] = sum(
        f.stat().st_size for f in (OUTPUT_ROOT / match_id).rglob('clip.mp4')
        if f.is_file()
    ) / 1e9

    log.info(f'  Match {match_id} done — clips: {stats["done"]}, '
             f'failed: {stats["failed"]}, size: {stats["size_gb"]:.2f}GB')
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info('=== STEP 4: Extract clips for all 10 matches ===')

    free_gb = disk_free_gb(OUTPUT_ROOT)
    if free_gb < 100:
        sys.exit(f'STOP: only {free_gb:.1f}GB free (need ≥100GB)')
    log.info(f'Disk free: {free_gb:.1f}GB  ✓')
    log.info(f'Matches: {ALL_MATCHES}')

    all_stats = []
    t_start = time.time()

    for match_id in ALL_MATCHES:
        free_gb = disk_free_gb(OUTPUT_ROOT)
        if free_gb < 100:
            log.error(f'STOP: disk dropped to {free_gb:.1f}GB before match {match_id}')
            break
        stats = process_match(match_id)
        all_stats.append(stats)

    elapsed = time.time() - t_start

    # ── Aggregate summary ─────────────────────────────────────────────────────
    log.info('\n' + '='*60)
    log.info('STEP 4 — Aggregate Summary')
    log.info('='*60)
    log.info(f'{"Match":<10} {"BAS":>6} {"Valid":>6} {"Done":>6} {"Skip":>6} {"Fail":>6} {"GB":>6}')
    log.info('-'*55)
    total_done = total_fail = total_gb = 0
    for s in all_stats:
        log.info(f'{s["match_id"]:<10} {s["total_bas"]:>6} {s["valid"]:>6} '
                 f'{s["done"]:>6} {s["skipped_resume"]:>6} {s["failed"]:>6} '
                 f'{s["size_gb"]:>6.2f}')
        total_done += s['done']
        total_fail += s['failed']
        total_gb   += s['size_gb']
    log.info('-'*55)
    log.info(f'{"TOTAL":<10} {sum(s["total_bas"] for s in all_stats):>6} '
             f'{sum(s["valid"] for s in all_stats):>6} '
             f'{total_done:>6} {sum(s["skipped_resume"] for s in all_stats):>6} '
             f'{total_fail:>6} {total_gb:>6.2f}')
    log.info(f'\nWall-clock time: {elapsed/60:.1f} min')
    log.info(f'Disk free remaining: {disk_free_gb(OUTPUT_ROOT):.1f}GB')
    log.info('\nSTEP 4 complete. Awaiting approval before STEP 5.')


if __name__ == '__main__':
    main()
