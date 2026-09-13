"""
run_eval_matrix.py — Run every evaluation script and produce one summary.

Why this script
---------------
For the dissertation we want a single command that runs every per-model
benchmark and emits a unified `docs/EVAL_MATRIX.json` + a printable
markdown summary. Easier than chasing seven separate JSON files.

What it runs (in order)
-----------------------
    1. eval_audio_scene.py    (ESC-50 — existing)
    2. eval_wlasl.py          (WLASL — existing)
    3. eval_sign_recognizer.py (custom sign set — existing)
    4. eval_whisper.py        (LibriSpeech WER — new)
    5. eval_blip_scene.py     (BLIP qualitative — new)
    6. eval_emotion.py        (DeepFace per-label — new)
    7. eval_yolo_latency.py   (YOLO p95 — new)
    8. eval_personalizer.py   (Personalizer P/R/F1 — new)

Any script that can't find its data writes an empty result and is
marked SKIPPED in the matrix — no crash.

Usage
-----
    python scripts/run_eval_matrix.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT.parent / "docs"
OUT_JSON = DOCS / "EVAL_MATRIX.json"
OUT_MD = DOCS / "EVAL_MATRIX.md"

# (script_name, result_json_filename, friendly_name)
SCRIPTS = [
    ("eval_audio_scene.py",     "esc50_eval_results.json",       "ESC-50 (AST)"),
    ("eval_wlasl.py",           "wlasl_eval_results.json",       "WLASL-100 (TGCN+geom)"),
    ("eval_sign_recognizer.py", "sign_eval_results.json",        "Custom sign set"),
    ("eval_whisper.py",         "whisper_eval_results.json",     "LibriSpeech WER (Whisper)"),
    ("eval_blip_scene.py",      "blip_eval_results.json",        "BLIP scene captions"),
    ("eval_emotion.py",         "emotion_eval_results.json",     "DeepFace 7-class"),
    ("eval_yolo_latency.py",    "yolo_eval_results.json",        "YOLO hazard latency"),
    ("eval_personalizer.py",    "personalizer_eval_results.json","Personalizer P/R/F1"),
]


def _run(script: str) -> Dict[str, Any]:
    t0 = time.time()
    try:
        p = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)],
            capture_output=True, text=True, cwd=str(ROOT), timeout=900,
        )
        return {
            "ok": p.returncode == 0,
            "elapsed_s": round(time.time() - t0, 1),
            "stdout_tail": "\n".join(p.stdout.splitlines()[-10:]),
            "stderr_tail": "\n".join(p.stderr.splitlines()[-5:]),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "elapsed_s": 900, "stdout_tail": "", "stderr_tail": "timeout"}
    except Exception as e:
        return {"ok": False, "elapsed_s": round(time.time() - t0, 1),
                "stdout_tail": "", "stderr_tail": str(e)[:200]}


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    matrix: Dict[str, Any] = {"runs": {}, "results": {}}

    for script, result_file, friendly in SCRIPTS:
        if not (ROOT / "scripts" / script).exists():
            matrix["runs"][script] = {"ok": False, "reason": "script not found"}
            continue
        print(f"▶ {friendly}  ({script})")
        run = _run(script)
        matrix["runs"][script] = run
        # Load the result JSON if it was produced
        rp = ROOT / result_file
        if rp.exists():
            try:
                matrix["results"][friendly] = json.loads(rp.read_text())
            except Exception:
                matrix["results"][friendly] = {"error": "json parse failed"}
        else:
            matrix["results"][friendly] = {"status": "SKIPPED — no data / no output"}
        print(f"   ok={run['ok']}  {run['elapsed_s']}s\n")

    OUT_JSON.write_text(json.dumps(matrix, indent=2))

    # ── Pretty markdown summary ────────────────────────────────────
    lines = ["# Evaluation matrix (auto-generated)\n",
             "Run via `python backend/scripts/run_eval_matrix.py`. "
             "This is the development harness: \"Script ran\" means the "
             "evaluation script finished, not that the target was met. "
             "Empty cells mean the script couldn't find its dataset; "
             "drop the data in and re-run. The results quoted in the final "
             "report come from `backend/scripts/report_experiments/` and are "
             "stored in `backend/eval_results/`.\n"]
    lines.append("| Component | Headline metric | Value | Target | Script ran |")
    lines.append("|---|---|---|---|---|")
    for script, _, friendly in SCRIPTS:
        r = matrix["results"].get(friendly, {})
        run = matrix["runs"].get(script, {})

        # Map each result shape to one headline number. Order matters —
        # check the most specific keys first.
        if "wer_mean" in r and r["wer_mean"] is not None:
            metric, value, target = "WER", f"{r['wer_mean']*100:.1f} %", "≤ 15 %"
        elif "top1_accuracy" in r:
            metric, value, target = "Top-1", f"{r['top1_accuracy']*100:.1f} %", "≥ 30 %"
        elif "overall_accuracy" in r:
            metric = f"Top-{r.get('top_k', 1)} acc"
            value = f"{r['overall_accuracy']*100:.1f} %"
            target = "≥ 60 %" if r.get('top_k', 1) >= 3 else "≥ 45 %"
        elif "macro_f1" in r:
            metric, value, target = "Macro F1", f"{r['macro_f1']:.3f}", "≥ 0.80"
        elif "accuracy" in r:
            metric, value, target = "Accuracy", f"{r['accuracy']*100:.1f} %", "≥ 50 %"
        elif "p95_latency_ms" in r:
            metric, value, target = "p95 latency", f"{r['p95_latency_ms']} ms", "≤ 500 ms"
        elif "mean_latency_ms" in r and r.get("n_images", 1) > 0:
            metric, value, target = "Mean latency", f"{r['mean_latency_ms']} ms", "≤ 2 000 ms"
        elif r.get("n_clips") == 0 or r.get("n_images") == 0 or r.get("n_labels") == 0:
            metric, value, target = "—", "(awaiting data)", "—"
        elif "status" in r:
            metric, value, target = "—", "(no data)", "—"
        else:
            metric, value, target = "—", "—", "—"

        status_emoji = "✓" if run.get("ok") else ("⏭ SKIPPED" if "status" in r else "❌")
        lines.append(f"| {friendly} | {metric} | {value} | {target} | {status_emoji} |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\n✓ Saved {OUT_JSON.relative_to(ROOT.parent)}")
    print(f"✓ Saved {OUT_MD.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
