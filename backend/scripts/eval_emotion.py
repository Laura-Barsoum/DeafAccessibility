"""
eval_emotion.py — Emotion classifier accuracy on per-label frames.

Why this script
---------------
DeepFace's 7-class Ekman classifier is one of the orchestrated models.
We test it on a small per-label sample to confirm it generalises to
unseen faces under our pipeline's conditions (webcam-quality JPEGs,
imperfect cropping).

Folder layout it expects
------------------------
    backend/data/emotion_eval/
        happy/      *.jpg
        sad/        *.jpg
        angry/      *.jpg
        fear/       *.jpg
        surprise/   *.jpg
        disgust/    *.jpg
        neutral/    *.jpg

Metrics
-------
Per-label: precision, recall, F1.
Overall: macro-averaged accuracy + per-class confusion summary.

Usage
-----
    python scripts/eval_emotion.py
"""
from __future__ import annotations

import base64
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "emotion_eval"
OUT_PATH = ROOT / "emotion_eval_results.json"

EKMAN_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    label_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if not label_dirs:
        print(f"No per-label folders in {DATA_DIR}")
        print("Create one sub-folder per emotion label with .jpg samples.")
        with open(OUT_PATH, "w") as f:
            json.dump({"n_images": 0}, f, indent=2)
        return

    from modules.emotion import VisualEmotionAnalyzer
    analyzer = VisualEmotionAnalyzer()

    correct = 0
    total = 0
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    confusion: Dict[str, Dict[str, int]] = {l: defaultdict(int) for l in EKMAN_LABELS}

    print("Per-image results:")
    for d in label_dirs:
        gt = d.name.lower()
        for p in sorted(list(d.glob("*.jpg")) + list(d.glob("*.png"))):
            b64 = base64.b64encode(p.read_bytes()).decode()
            try:
                res = analyzer.analyse_jpeg(b64)
            except Exception as e:
                print(f"  ⚠ {p.name}: {e}")
                continue
            pred = (res.get("label") or "neutral").lower()
            total += 1
            confusion.setdefault(gt, defaultdict(int))[pred] += 1
            if pred == gt:
                correct += 1
                tp[gt] += 1
            else:
                fp[pred] += 1
                fn[gt] += 1
            print(f"  {p.name:30s} gt={gt:8s}  pred={pred:8s}  {'✓' if pred == gt else '✗'}")

    n_labels = len(label_dirs)
    summary_per_label = {}
    for label in [d.name.lower() for d in label_dirs]:
        tps, fps, fns = tp[label], fp[label], fn[label]
        precision = tps / max(1, tps + fps)
        recall    = tps / max(1, tps + fns)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        summary_per_label[label] = {
            "tp": tps, "fp": fps, "fn": fns,
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
        }

    out = {
        "n_images":         total,
        "n_labels":         n_labels,
        "accuracy":         round(correct / max(1, total), 3),
        "per_label":        summary_per_label,
        "confusion_matrix": {gt: dict(row) for gt, row in confusion.items()},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nOverall accuracy: {out['accuracy']:.1%}  ({correct}/{total})")
    print(f"Per-label F1:")
    for label, s in summary_per_label.items():
        print(f"  {label:10s} F1={s['f1']:.2f}")
    print(f"\n✓ Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
