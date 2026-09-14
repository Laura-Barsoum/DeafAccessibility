"""
eval_yolo_latency.py — YOLO hazard-detection latency on the CPU path.

Why this script
---------------
Hazards are SAFETY-CRITICAL. The dissertation needs to demonstrate the
hazard detector meets the < 500 ms safety latency budget on the same
CPU the user runs the assistant on. mAP requires the full COCO val set
which is too large for a quick run, so this script focuses on the
latency dimension and produces a histogram + p50/p95/p99.

If you also want mAP, run `ultralytics`'s built-in `yolo val` against
COCO-val — that's the standard published benchmark we'd compare to.

Test set
--------
Uses ALL frames found in `backend/data/personalizer_eval/` (or any folder
passed via --frames-dir) so you can produce numbers without downloading
any extra data. For a stronger benchmark, point --frames-dir at a folder
of real photographs (e.g. ~100 COCO val images).

Usage
-----
    python scripts/eval_yolo_latency.py
    python scripts/eval_yolo_latency.py --frames-dir /path/to/coco_val_subset --n 100
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "yolo_eval_results.json"


def _frame_to_b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=str, default=None,
                    help="folder of JPEG/PNG frames (default: search for any images "
                         "under data/)")
    ap.add_argument("--n", type=int, default=50,
                    help="max number of frames to time")
    args = ap.parse_args()

    # ── Collect frames ─────────────────────────────────────────────
    if args.frames_dir:
        search_dir = Path(args.frames_dir)
    else:
        search_dir = ROOT / "data"
    frames = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        frames.extend(search_dir.rglob(ext))
    frames = sorted(set(frames))[: args.n]
    if not frames:
        print(f"No image frames found in {search_dir}. Provide --frames-dir.")
        with open(OUT_PATH, "w") as f:
            json.dump({"n_frames": 0, "latencies_ms": []}, f, indent=2)
        return

    # ── Load detector via the same import path the server uses ────
    from modules.hazard_detector import HazardDetector
    det = HazardDetector()

    # Warm-up
    print(f"Warming up YOLO on first frame...")
    det.detect_to_events([_frame_to_b64(frames[0])])

    print(f"Timing {len(frames)} frames\n")
    latencies: List[float] = []
    n_detections = 0
    for p in frames:
        b64 = _frame_to_b64(p)
        t0 = time.perf_counter()
        events, hazards = det.detect_to_events([b64])
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        n_detections += len(hazards)

    sorted_lats = sorted(latencies)
    p50 = sorted_lats[len(sorted_lats) // 2]
    p95 = sorted_lats[int(len(sorted_lats) * 0.95)]
    p99 = sorted_lats[min(len(sorted_lats) - 1, int(len(sorted_lats) * 0.99))]

    out = {
        "n_frames":             len(latencies),
        "mean_latency_ms":      round(statistics.mean(latencies), 1),
        "median_latency_ms":    round(p50, 1),
        "p95_latency_ms":       round(p95, 1),
        "p99_latency_ms":       round(p99, 1),
        "safety_budget_ms":     500,
        "frames_within_budget": sum(1 for l in latencies if l <= 500),
        "total_hazards":        n_detections,
        "latencies_ms":         [round(l, 1) for l in latencies],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Mean:   {out['mean_latency_ms']} ms")
    print(f"Median: {out['median_latency_ms']} ms")
    print(f"p95:    {out['p95_latency_ms']} ms")
    print(f"p99:    {out['p99_latency_ms']} ms")
    print(f"Within 500 ms budget: {out['frames_within_budget']}/{out['n_frames']}")
    print(f"\n✓ Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
