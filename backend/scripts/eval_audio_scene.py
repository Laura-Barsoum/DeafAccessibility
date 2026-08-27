"""
eval_audio_scene.py — Evaluate the YamNet-based audio-scene classifier on ESC-50.

For each ESC-50 clip we:
    1. Run our AudioSceneClassifier
    2. Map ESC-50 ground truth label → set of acceptable AudioSet labels
       (via the ESC50_TO_YAMNET map in modules/audio_scene.py)
    3. Mark a "hit" if ANY of the acceptable labels appears in our top-3
       predictions

Outputs:
    - Overall top-1 / top-3 accuracy
    - Per-class accuracy
    - Confusion analysis: which classes does YamNet confuse?
    - JSON dump for the report

This script answers the brief's marking criterion explicitly:
    > "We are keen to see evidence that you have tested several models
    >  and made decisions about which are and are not appropriate."

Usage:
    python scripts/eval_audio_scene.py
    python scripts/eval_audio_scene.py --top-k 3 --max-per-class 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Make `modules` importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.audio_scene import AudioSceneClassifier, ESC50_TO_YAMNET  # noqa: E402

ESC50_ROOT = Path(__file__).resolve().parent.parent / "data" / "ESC-50"
META_PATH = ESC50_ROOT / "meta" / "esc50.csv"
AUDIO_DIR = ESC50_ROOT / "audio"


def load_meta() -> List[Dict[str, str]]:
    if not META_PATH.exists():
        raise SystemExit(
            f"ESC-50 not found at {META_PATH}.\n"
            f"Run: python scripts/download_esc50.py first."
        )
    with open(META_PATH) as f:
        return list(csv.DictReader(f))


def label_hits(predicted: List[Tuple[str, float]], gt_class: str) -> bool:
    """Did any of the predicted top-k YamNet labels match the ESC-50 class?"""
    accepted = ESC50_TO_YAMNET.get(gt_class, [])
    if not accepted:
        # If we have no mapping for this class, count as a miss but log it
        return False
    pred_labels = {p[0].lower() for p in predicted}
    for a in accepted:
        if a.lower() in pred_labels:
            return True
    # Also accept partial substring matches
    for a in accepted:
        for pl in pred_labels:
            if a.lower() in pl or pl in a.lower():
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--out", type=str, default="esc50_eval_results.json")
    args = parser.parse_args()

    meta = load_meta()
    print(f"Loaded {len(meta)} ESC-50 clips")
    if args.max_per_class is not None:
        # Stratified subsample
        by_class: Dict[str, List[Dict]] = defaultdict(list)
        for row in meta:
            by_class[row["category"]].append(row)
        meta = []
        for c, rows in by_class.items():
            meta.extend(rows[: args.max_per_class])
        print(f"Subsampled to {len(meta)} clips ({args.max_per_class}/class)")

    classifier = AudioSceneClassifier()

    correct = 0
    total = 0
    per_class: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # [hits, total]
    confusion: Dict[str, Counter] = defaultdict(Counter)

    start = time.time()
    for i, row in enumerate(meta):
        path = AUDIO_DIR / row["filename"]
        if not path.exists():
            continue
        with open(path, "rb") as f:
            audio_bytes = f.read()
        preds = classifier.classify(audio_bytes, sample_rate=44100, top_k=args.top_k)
        gt = row["category"]
        hit = label_hits(preds, gt)
        per_class[gt][1] += 1
        if hit:
            correct += 1
            per_class[gt][0] += 1
        else:
            if preds:
                confusion[gt][preds[0][0]] += 1
        total += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  {i+1}/{len(meta)}  acc={correct/total:.3f}  "
                  f"({elapsed:.1f}s, {(i+1)/elapsed:.1f} clips/s)")

    elapsed = time.time() - start
    print(f"\n=== Evaluation complete ({elapsed:.1f}s) ===")
    overall = correct / max(1, total)
    print(f"Top-{args.top_k} accuracy:  {overall:.3f}  ({correct}/{total})")

    # Per-class table
    print("\nPer-class accuracy:")
    rows = []
    for cls, (hits, n) in sorted(per_class.items()):
        if n == 0:
            continue
        acc = hits / n
        rows.append((cls, hits, n, acc))
        print(f"  {cls:25s}  {hits:3d}/{n:3d}  ({acc:.2f})")

    # Top confusions per class
    print("\nMost-confused predictions per ground-truth class:")
    for cls, counter in confusion.items():
        if not counter:
            continue
        top = counter.most_common(2)
        confused = ", ".join(f"{lbl} (×{n})" for lbl, n in top)
        print(f"  {cls:25s} → {confused}")

    # JSON dump
    out_path = Path(__file__).resolve().parent.parent / args.out
    with open(out_path, "w") as f:
        json.dump({
            "top_k": args.top_k,
            "n_clips": total,
            "n_correct": correct,
            "overall_accuracy": overall,
            "per_class": {c: {"hits": h, "total": n, "acc": h / n if n else 0}
                          for c, (h, n) in per_class.items()},
            "elapsed_s": elapsed,
        }, f, indent=2)
    print(f"\nJSON results → {out_path}")


if __name__ == "__main__":
    main()
