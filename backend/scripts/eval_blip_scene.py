"""
eval_blip_scene.py — Qualitative BLIP scene-caption spot-check.

Why this script
---------------
BLIP is a generative model — its output is a free-form caption, so the
classical accuracy metric doesn't apply. The standard published metric
is CIDEr on COCO-captions, but running that needs the full annotation
set + pycocoevalcap (heavy install).

For the dissertation we report a **qualitative spot-check + latency
distribution** on a small fixed set of representative frames. The
output is a JSON table you can drop into the evaluation chapter.

Folder layout
-------------
    backend/data/scene_eval/
        kitchen.jpg
        living_room.jpg
        ...

Usage
-----
    python scripts/eval_blip_scene.py
"""
from __future__ import annotations

import base64
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "scene_eval"
OUT_PATH = ROOT / "blip_eval_results.json"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.png")))
    if not images:
        print(f"No images in {DATA_DIR} — drop 5-10 representative photos there.")
        with open(OUT_PATH, "w") as f:
            json.dump({"n_images": 0, "captions": []}, f, indent=2)
        return

    from modules.scene import SceneCaptioner
    cap = SceneCaptioner()

    print(f"Captioning {len(images)} images via BLIP\n")
    results: List[dict] = []
    latencies: List[float] = []
    for p in images:
        b64 = base64.b64encode(p.read_bytes()).decode()
        t0 = time.perf_counter()
        caption = cap.describe_jpeg(b64)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        results.append({
            "image": p.name,
            "caption": caption,
            "latency_ms": round(lat, 0),
        })
        print(f"  {p.name:30s} {lat:5.0f} ms")
        print(f"    → {caption}\n")

    out = {
        "n_images":          len(results),
        "mean_latency_ms":   round(statistics.mean(latencies), 0) if latencies else 0,
        "median_latency_ms": round(statistics.median(latencies), 0) if latencies else 0,
        "captions":          results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Mean latency: {out['mean_latency_ms']} ms")
    print(f"✓ Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
