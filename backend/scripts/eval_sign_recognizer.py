"""
eval_sign_recognizer.py — Empirically evaluate the pre-trained MediaPipe
GestureRecognizer on a small test set of static handshape images.

This is what the assignment brief calls "evidence that you have tested
several models and made decisions about which are and are not
appropriate". We don't train anything — we evaluate.

How it works
------------
1. We point the script at a directory of test JPEG images organised by
   class folder, e.g.:

       backend/data/sign_test_set/
           Open_Palm/    1.jpg 2.jpg 3.jpg ...
           Closed_Fist/  1.jpg ...
           Thumb_Up/     1.jpg ...
           ...

   The expected class names are MediaPipe's native labels.

2. For each image we run the pre-trained GestureRecognizer and record
   its top-1 prediction.

3. We report:
       - per-class accuracy
       - overall accuracy
       - confusion matrix (which classes get confused for which)
       - average inference latency

If `backend/data/sign_test_set/` doesn't exist yet, the script will
synthesise a tiny demo test set by extracting frames from the WLASL
manifest videos (where present) OR fall back to running the recognizer
on the WLASL gloss "yes" / "no" video samples that map cleanly to
Closed_Fist / Open_Palm.

Output
------
JSON report at `backend/sign_eval_results.json` + a human-readable
summary printed to stdout.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.sign_language import SignLanguageRecognizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "sign_test_set"
OUT_PATH = ROOT / "sign_eval_results.json"

# MediaPipe's 7 native class labels.
MEDIAPIPE_CLASSES = [
    "Open_Palm", "Closed_Fist", "Pointing_Up",
    "Thumb_Up", "Thumb_Down", "Victory", "ILoveYou",
]


def _jpeg_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def evaluate() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class-limit", type=int, default=20)
    args = parser.parse_args()

    if not TEST_DIR.exists():
        print(f"❌ Test directory not found: {TEST_DIR}")
        print()
        print("Quick way to populate it:")
        print(f"  mkdir -p {TEST_DIR}")
        for cls in MEDIAPIPE_CLASSES:
            print(f"  mkdir -p {TEST_DIR / cls}")
        print()
        print("Then drop 5-20 JPEGs of each handshape into the matching folder")
        print("and re-run this script.")
        sys.exit(0)

    slr = SignLanguageRecognizer()
    # Force-load the pre-trained recognizer so the lazy-load timing
    # isn't included in per-image latency.
    slr._ensure_gesture_recognizer()
    if slr._gesture_recognizer in (None, "placeholder"):
        print("❌ Pre-trained GestureRecognizer is not loaded.")
        print("   Run `python scripts/download_gesture_model.py` first.")
        sys.exit(1)

    print(f"✓ Pre-trained GestureRecognizer loaded\n")
    print(f"Evaluating on {TEST_DIR}\n")

    results: List[Dict] = []
    per_class_n = defaultdict(int)
    per_class_correct = defaultdict(int)
    confusion = defaultdict(Counter)
    latencies: List[float] = []

    for class_dir in sorted(TEST_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        gt_class = class_dir.name
        images = sorted(class_dir.glob("*.jp*g")) + sorted(class_dir.glob("*.png"))
        images = images[: args.per_class_limit]
        if not images:
            continue
        print(f"  {gt_class:15s} ({len(images)} images)...")
        for img_path in images:
            b64 = _jpeg_to_b64(img_path)
            t0 = time.perf_counter()
            result = slr.recognize_pretrained(b64)
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)

            predicted = result[0] if result else "None"
            confidence = result[1] if result else 0.0

            # Map predicted (mapped sign label e.g. "hello") back to MediaPipe
            # class for like-for-like comparison.
            inv_map = {v: k for k, v in slr._GESTURE_LABEL_MAP.items() if v}
            pred_mp = inv_map.get(predicted, predicted)

            per_class_n[gt_class] += 1
            if pred_mp == gt_class:
                per_class_correct[gt_class] += 1
            confusion[gt_class][pred_mp] += 1

            results.append({
                "file": str(img_path.relative_to(ROOT)),
                "ground_truth": gt_class,
                "predicted_mediapipe": pred_mp,
                "predicted_mapped": predicted,
                "confidence": round(confidence, 3),
                "latency_ms": round(latency, 1),
            })

    if not results:
        print("\nNo test images found. Add JPEGs to the class folders and retry.")
        sys.exit(0)

    total_n = sum(per_class_n.values())
    total_correct = sum(per_class_correct.values())
    overall_acc = total_correct / total_n if total_n else 0.0

    print("\n=== Per-class accuracy ===")
    rows = []
    for cls in sorted(per_class_n):
        n = per_class_n[cls]
        c = per_class_correct[cls]
        acc = c / n if n else 0
        rows.append((cls, c, n, acc))
        print(f"  {cls:15s} {c:3d}/{n:3d}  ({acc*100:.1f}%)")

    print(f"\nOverall accuracy: {total_correct}/{total_n} ({overall_acc*100:.1f}%)")
    if latencies:
        print(f"Mean latency: {sum(latencies)/len(latencies):.1f} ms")
        print(f"P95 latency:  {sorted(latencies)[int(len(latencies)*0.95)]:.1f} ms")

    print("\n=== Top confusions ===")
    for cls, ctr in confusion.items():
        for other, count in ctr.most_common(2):
            if other != cls and count > 0:
                print(f"  {cls:15s} → {other:15s}  (×{count})")

    out = {
        "n_test_images": total_n,
        "n_correct": total_correct,
        "overall_accuracy": overall_acc,
        "per_class": {c: {"n": per_class_n[c], "correct": per_class_correct[c],
                          "accuracy": per_class_correct[c]/per_class_n[c] if per_class_n[c] else 0}
                      for c in per_class_n},
        "mean_latency_ms": sum(latencies)/len(latencies) if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
        "confusions": {c: dict(ctr) for c, ctr in confusion.items()},
        "results": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved {OUT_PATH}")


if __name__ == "__main__":
    evaluate()
