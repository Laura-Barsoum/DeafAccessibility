"""
eval_personalizer.py — Precision/recall on held-out custom-sound clips.

Why this script
---------------
The personalizer is the most distinctive feature of this project. The
dissertation needs hard numbers showing the few-shot cosine-match
recogniser works on held-out clips it didn't see during enrolment.

Folder layout it expects
------------------------
    backend/data/personalizer_eval/
        my_doorbell/
            enrol_01.wav      # used to enrol
            enrol_02.wav
            test_01.wav       # held-out positive examples
            test_02.wav
        baby_cry/
            ...

Naming convention:
    enrol_*.wav  → used to build the embedding
    test_*.wav   → held-out positives
    distractor_<other_label>/  → used as held-out NEGATIVES for every
                                 other enrolled label.

Metrics
-------
Per-label: precision, recall, F1 at the default threshold (0.78).
Overall: macro-averaged P / R / F1, plus a confusion summary.

Usage
-----
    python scripts/eval_personalizer.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.personalizer import Personalizer  # noqa: E402
from modules.events import Priority             # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "personalizer_eval"
OUT_PATH = ROOT / "personalizer_eval_results.json"
THRESHOLD = 0.78


def _wav_bytes(p: Path) -> bytes:
    return p.read_bytes()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Discover labels = sub-directory names
    label_dirs = [d for d in DATA_DIR.iterdir()
                  if d.is_dir() and not d.name.startswith(".")]
    if not label_dirs:
        print(f"No label folders found in {DATA_DIR}")
        print("Create one sub-folder per sound, with `enrol_*.wav` + `test_*.wav` clips.")
        with open(OUT_PATH, "w") as f:
            json.dump({"n_labels": 0, "samples": []}, f, indent=2)
        return

    # ── Fresh in-memory personalizer; do NOT touch the live profile.
    p = Personalizer()
    # Wipe any prior state so the eval is clean.
    p.profile = {"sounds": {}}

    # Enrol every label
    for d in label_dirs:
        clips = sorted(d.glob("enrol_*.wav")) + sorted(d.glob("enrol_*.mp3"))
        if not clips:
            print(f"  ⚠ {d.name}: no enrol_* clips — skipping")
            continue
        audio_list = [_wav_bytes(c) for c in clips]
        r = p.enrol(d.name, Priority.IMPORTANT, audio_list)
        print(f"  ✓ enrolled {d.name} ({r.get('n_examples', 0)} examples)")

    # Build the held-out test set: all test_*.wav per label
    test_pool: List[Dict] = []
    for d in label_dirs:
        for t in sorted(d.glob("test_*.wav")) + sorted(d.glob("test_*.mp3")):
            test_pool.append({"true_label": d.name, "path": t})
    if not test_pool:
        print("No test_*.wav clips found — cannot compute precision/recall.")
        return

    print(f"\nEvaluating {len(test_pool)} held-out clips...\n")

    # Per-label counters
    tp = {d.name: 0 for d in label_dirs}
    fp = {d.name: 0 for d in label_dirs}
    fn = {d.name: 0 for d in label_dirs}

    per_clip: List[Dict] = []
    for t in test_pool:
        true_label = t["true_label"]
        events = p.match(_wav_bytes(t["path"]), threshold=THRESHOLD)
        predicted = [e.label.replace(" (personal)", "") for e in events]
        is_correct = true_label in predicted
        if is_correct:
            tp[true_label] += 1
        else:
            fn[true_label] += 1
        for pred in predicted:
            if pred != true_label and pred in fp:
                fp[pred] += 1
        per_clip.append({
            "path": str(t["path"].relative_to(DATA_DIR)),
            "true_label": true_label,
            "predicted": predicted,
            "correct": is_correct,
        })

    # Aggregate
    summary_per_label = {}
    P_sum, R_sum, F_sum, n_labels = 0.0, 0.0, 0.0, 0
    for label in tp:
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
        P_sum += precision; R_sum += recall; F_sum += f1; n_labels += 1

    out = {
        "n_labels": n_labels,
        "n_test_clips": len(test_pool),
        "threshold": THRESHOLD,
        "macro_precision": round(P_sum / max(1, n_labels), 3),
        "macro_recall":    round(R_sum / max(1, n_labels), 3),
        "macro_f1":        round(F_sum / max(1, n_labels), 3),
        "per_label": summary_per_label,
        "per_clip":  per_clip,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 60)
    print(f"Macro precision: {out['macro_precision']:.3f}  (target ≥ 0.85)")
    print(f"Macro recall:    {out['macro_recall']:.3f}  (target ≥ 0.80)")
    print(f"Macro F1:        {out['macro_f1']:.3f}")
    print(f"\nPer-label:")
    for label, s in summary_per_label.items():
        print(f"  {label:25s} P={s['precision']:.2f} R={s['recall']:.2f} F1={s['f1']:.2f}")
    print(f"\n✓ Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
