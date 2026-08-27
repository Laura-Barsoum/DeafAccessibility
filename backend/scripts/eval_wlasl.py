"""
eval_wlasl.py — Evaluate the sign-language pipeline against WLASL.

This populates the assignment brief's "evidence of testing several models"
requirement with HARD NUMBERS against the academic WLASL benchmark.

Pipeline
--------
1. Read the WLASL v0.3 JSON manifest from `data/WLASL-master/`.
2. For each gloss in the top-N vocab list, find associated video samples.
3. Extract ~30 frames per video via ffmpeg (uniformly spaced).
4. Run the full sign pipeline (TGCN + MediaPipe + letter + geometric)
   on each clip.
5. Report top-1 / top-3 / top-5 accuracy per gloss + overall.

Usage
-----
    python scripts/eval_wlasl.py --top-n 100 --videos-per-class 3
    # → writes backend/wlasl_eval_results.json

Requires the WLASL-master dataset under backend/data/WLASL-master/ and
ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.sign_language import SignLanguageRecognizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WLASL_DIR = ROOT / "data" / "WLASL-master"
MANIFEST = WLASL_DIR / "start_kit" / "WLASL_v0.3.json"
VIDEOS_DIR = WLASL_DIR / "videos"
OUT_PATH = ROOT / "wlasl_eval_results.json"


def _extract_frames(video_path: Path, n_frames: int = 30) -> List[str]:
    """Extract `n_frames` JPEG frames from a video, return them as base64 strings."""
    out_frames: List[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        out_pattern = Path(tmp) / "frame_%04d.jpg"
        # Use ffmpeg to extract uniformly-spaced frames
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vf", f"fps={n_frames}/{_get_duration(video_path):.4f}",
            "-q:v", "3",
            str(out_pattern),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        except Exception as e:
            print(f"   ffmpeg failed on {video_path.name}: {e}")
            return []
        files = sorted(Path(tmp).glob("frame_*.jpg"))
        for f in files[:n_frames]:
            with open(f, "rb") as fp:
                out_frames.append(base64.b64encode(fp.read()).decode())
    return out_frames


def _get_duration(video_path: Path) -> float:
    """Use ffprobe to read the duration of a video in seconds."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return max(0.5, float(res.stdout.strip()))
    except Exception:
        return 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=100,
                        help="evaluate against the top-N WLASL glosses")
    parser.add_argument("--videos-per-class", type=int, default=2,
                        help="how many videos to sample per gloss")
    parser.add_argument("--frames-per-video", type=int, default=30)
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(f"❌ WLASL manifest not found: {MANIFEST}")
        print("   Expected at data/WLASL-master/start_kit/WLASL_v0.3.json")
        sys.exit(1)

    with open(MANIFEST) as f:
        manifest = json.load(f)

    # Pick the top-N glosses (sorted by number of instances available)
    glosses_by_count = sorted(
        manifest, key=lambda g: -len(g.get("instances", []))
    )
    chosen = glosses_by_count[: args.top_n]
    print(f"✓ Evaluating against top {len(chosen)} WLASL glosses\n")

    slr = SignLanguageRecognizer()
    slr._ensure_gesture_recognizer()
    slr._ensure_tgcn()

    per_class_n = defaultdict(int)
    top1_correct = defaultdict(int)
    top3_correct = defaultdict(int)
    top5_correct = defaultdict(int)
    latencies: List[float] = []
    misclassifications: List[Dict] = []

    t_start = time.time()

    for gi, gloss_entry in enumerate(chosen, 1):
        gt_gloss = gloss_entry["gloss"].lower()
        instances = gloss_entry.get("instances", [])[: args.videos_per_class]
        if not instances:
            continue
        print(f"[{gi}/{len(chosen)}] {gt_gloss} — {len(instances)} videos")

        for inst in instances:
            video_id = inst.get("video_id", "")
            video_path = VIDEOS_DIR / f"{video_id}.mp4"
            if not video_path.exists():
                # Try alternative extensions
                for ext in (".mkv", ".webm", ".avi"):
                    alt = VIDEOS_DIR / f"{video_id}{ext}"
                    if alt.exists():
                        video_path = alt
                        break
                else:
                    continue

            frames_b64 = _extract_frames(video_path, n_frames=args.frames_per_video)
            if not frames_b64:
                continue

            t0 = time.perf_counter()
            sequence, diag = slr.classify_all_frames_combined(
                frames_b64, window_size=8, stride=3,
            )
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)
            per_class_n[gt_gloss] += 1

            predicted_labels = [s["label"].lower() for s in sequence]
            if gt_gloss in predicted_labels[:1]:
                top1_correct[gt_gloss] += 1
            if gt_gloss in predicted_labels[:3]:
                top3_correct[gt_gloss] += 1
            if gt_gloss in predicted_labels[:5]:
                top5_correct[gt_gloss] += 1
            else:
                misclassifications.append({
                    "gloss": gt_gloss,
                    "video": video_path.name,
                    "predicted_top5": predicted_labels[:5],
                })

    # ── Aggregate ────────────────────────────────────────────────────────
    n_total = sum(per_class_n.values())
    n_top1  = sum(top1_correct.values())
    n_top3  = sum(top3_correct.values())
    n_top5  = sum(top5_correct.values())
    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"WLASL Evaluation — {len(per_class_n)} glosses, {n_total} clips")
    print(f"{'='*60}")
    print(f"Top-1 accuracy:  {n_top1}/{n_total} ({100*n_top1/max(1,n_total):.1f}%)")
    print(f"Top-3 accuracy:  {n_top3}/{n_total} ({100*n_top3/max(1,n_total):.1f}%)")
    print(f"Top-5 accuracy:  {n_top5}/{n_total} ({100*n_top5/max(1,n_total):.1f}%)")
    print(f"Mean latency:    {sum(latencies)/max(1,len(latencies)):.0f} ms/clip")
    print(f"Total time:      {elapsed/60:.1f} min")

    out = {
        "n_glosses":   len(per_class_n),
        "n_clips":     n_total,
        "top1_correct": n_top1,
        "top3_correct": n_top3,
        "top5_correct": n_top5,
        "top1_accuracy": n_top1 / max(1, n_total),
        "top3_accuracy": n_top3 / max(1, n_total),
        "top5_accuracy": n_top5 / max(1, n_total),
        "mean_latency_ms": sum(latencies) / max(1, len(latencies)),
        "p95_latency_ms":  sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
        "per_class": {
            g: {
                "n": per_class_n[g],
                "top1": top1_correct[g],
                "top3": top3_correct[g],
                "top5": top5_correct[g],
            } for g in per_class_n
        },
        "misclassifications": misclassifications[:40],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
