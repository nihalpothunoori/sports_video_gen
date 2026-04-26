#!/usr/bin/env python3
"""
Read-only validation of a single SoccerTrack v2 match.
Usage: python3 validate_match.py <match_id>
"""
import sys, os, json, struct
import numpy as np
import cv2

DATA_ROOT = "/home/nihal/Documents/cachehacks/soccer_track_data"

def check(label, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}" + (f": {detail}" if detail else ""))
    return passed

def validate_match(match_id):
    m = str(match_id)
    print(f"\n{'='*60}")
    print(f"Validating match {m}")
    print(f"{'='*60}")
    results = {}

    # ── Video ──────────────────────────────────────────────────────
    print("\n[A] Video files")
    video_base = os.path.join(DATA_ROOT, "videos", m)
    halves = {}
    for half, suffix in [("1st", f"{m}_panorama_1st_half.mp4"),
                          ("2nd", f"{m}_panorama_2nd_half.mp4")]:
        path = os.path.join(video_base, suffix)
        if not os.path.exists(path):
            check(f"{half} exists", False, path)
            halves[half] = None
            continue
        cap = cv2.VideoCapture(path)
        opened = cap.isOpened()
        check(f"{half} opens", opened, path)
        if not opened:
            halves[half] = None
            cap.release()
            continue
        fps   = cap.get(cv2.CAP_PROP_FPS)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        dur   = total / fps if fps > 0 else 0
        ret, frame = cap.read()
        has_frame = ret and frame is not None and frame.size > 0
        check(f"{half} readable frame", has_frame,
              f"shape={frame.shape if has_frame else 'N/A'}")
        check(f"{half} duration plausible", 30*60 < dur < 70*60,
              f"{dur:.1f}s ({dur/60:.1f} min) @ {fps:.2f} fps, {int(total)} frames")
        cap.release()
        halves[half] = {"fps": fps, "frames": int(total), "dur": dur}
        results[f"video_{half}"] = halves[half]

    # ── BAS ────────────────────────────────────────────────────────
    print("\n[B] BAS events")
    bas_path = os.path.join(DATA_ROOT, "bas", m, f"{m}_12_class_events.json")
    bas_ok = os.path.exists(bas_path) and os.path.getsize(bas_path) > 0
    check("BAS file exists and non-empty", bas_ok, bas_path)
    events = []
    if bas_ok:
        with open(bas_path) as f:
            bas_data = json.load(f)
        # Detect format: list of events or dict with nested structure
        if isinstance(bas_data, list):
            events = bas_data
        elif isinstance(bas_data, dict):
            # Could be {"annotations": [...]} or {"1st": [...], "2nd": [...]}
            for key in ("annotations", "events", "1st_half", "2nd_half", "1st", "2nd"):
                if key in bas_data:
                    v = bas_data[key]
                    if isinstance(v, list):
                        events.extend(v)
        check("BAS parses as JSON", True, f"{len(events)} events found")
        # Print top-level keys if dict
        if isinstance(bas_data, dict):
            print(f"    Keys: {list(bas_data.keys())[:10]}")
        # Sample structure of first event
        if events:
            print(f"    First event sample: {json.dumps(events[0], indent=2)[:300]}")

        # Check event classes
        classes = set()
        for ev in events:
            if isinstance(ev, dict):
                for k in ("label", "event_class", "class", "type", "action"):
                    if k in ev:
                        classes.add(str(ev[k]))
                        break
        check("Has multiple event classes", len(classes) >= 1,
              f"{len(classes)} classes: {sorted(classes)[:12]}")
        results["bas_events"] = len(events)
        results["bas_classes"] = sorted(classes)

    # ── GSR (stream first 50KB to avoid loading 2.6GB) ─────────────
    print("\n[C] GSR tracking data")
    for half_tag in ("1st", "2nd"):
        gsr_path = os.path.join(DATA_ROOT, "gsr", m, f"{m}_{half_tag}.json")
        gsr_exists = os.path.exists(gsr_path) and os.path.getsize(gsr_path) > 0
        check(f"GSR {half_tag} exists", gsr_exists,
              f"{os.path.getsize(gsr_path)//1024//1024}MB" if gsr_exists else gsr_path)
        if gsr_exists:
            # Stream first ~200KB to inspect structure without loading all 2.6GB
            with open(gsr_path, "rb") as f:
                head = f.read(200 * 1024).decode("utf-8", errors="replace")
            # Find first complete JSON object (rough)
            brace_depth = 0
            in_str = False
            esc = False
            cut = len(head)
            for i, ch in enumerate(head):
                if esc:
                    esc = False
                    continue
                if ch == "\\" and in_str:
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch in ('{', '['):
                    brace_depth += 1
                elif ch in ('}', ']'):
                    brace_depth -= 1
                    if brace_depth == 0:
                        cut = i + 1
                        break
            try:
                sample = json.loads(head[:cut])
                if isinstance(sample, dict):
                    top_keys = list(sample.keys())[:8]
                elif isinstance(sample, list) and sample:
                    top_keys = list(sample[0].keys())[:8] if isinstance(sample[0], dict) else []
                else:
                    top_keys = []
                check(f"GSR {half_tag} partial parse OK", True, f"top keys: {top_keys}")
                # Look for pitch coords (x,y or pitch_x,pitch_y or bbox_pitch etc.)
                sample_str = head[:2000]
                coord_hints = [k for k in ("pitch_x","pitch_y","x_pitch","y_pitch",
                                            "x","y","bbox","position")
                               if k in sample_str]
                check(f"GSR {half_tag} coord fields present", len(coord_hints) > 0,
                      f"hints: {coord_hints}")
            except json.JSONDecodeError as e:
                check(f"GSR {half_tag} partial parse OK", False, str(e)[:80])

    # ── Homography ─────────────────────────────────────────────────
    print("\n[D] Homography maps")
    raw_dir = os.path.join(DATA_ROOT, "raw", m)
    mapx_path = os.path.join(raw_dir, f"{m}_mapx.npy")
    mapy_path = os.path.join(raw_dir, f"{m}_mapy.npy")
    mapx_ok = os.path.exists(mapx_path)
    mapy_ok = os.path.exists(mapy_path)
    check("mapx.npy exists", mapx_ok, mapx_path)
    check("mapy.npy exists", mapy_ok, mapy_path)
    if mapx_ok and mapy_ok:
        mapx = np.load(mapx_path)
        mapy = np.load(mapy_path)
        shapes_match = mapx.shape == mapy.shape
        check("mapx/mapy shapes match", shapes_match,
              f"mapx={mapx.shape}, mapy={mapy.shape}")
        check("mapx non-trivial values", not np.all(mapx == 0),
              f"min={mapx.min():.1f}, max={mapx.max():.1f}")
        results["mapx_shape"] = list(mapx.shape)

    # ── Print 3 sample BAS events ──────────────────────────────────
    if events:
        print(f"\n[SAMPLE BAS EVENTS — 3 of {len(events)}]")
        for i, ev in enumerate(events[:3]):
            print(f"  Event {i+1}: {json.dumps(ev, indent=4)[:400]}")

    print(f"\n{'='*60}\nDone.\n")
    return results

if __name__ == "__main__":
    mid = sys.argv[1] if len(sys.argv) > 1 else "128057"
    validate_match(mid)
