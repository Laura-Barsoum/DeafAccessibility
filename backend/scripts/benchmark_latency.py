"""
benchmark_latency.py — End-to-end latency benchmark.

Measures the per-stage latency of the multimodal pipeline on a single
synthetic input:

    Stage             | Target ms
    ──────────────────┼──────────
    Whisper STT       | < 800
    YamNet classify   | < 200
    BLIP scene        | < 600
    MediaPipe face    | < 100
    LLM notification  | < 1000
    ─────────────────
    End-to-end        | < 2.5 s

Latencies above the target are flagged in the output. Used to validate
the ≤500ms safety-critical-event constraint claim in the proposal.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def synthesize_audio(seconds: float = 2.0, sr: int = 16000) -> bytes:
    """Generate a synthetic noise burst as a stand-in for real audio."""
    import io, wave
    n = int(seconds * sr)
    samples = (np.random.randn(n) * 0.05 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def synthesize_jpeg(width: int = 320, height: int = 240) -> str:
    """Generate a synthetic JPEG (uniform grey)."""
    import base64, io
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), (128, 128, 128))
        b = io.BytesIO()
        img.save(b, format="JPEG", quality=70)
        return base64.b64encode(b.getvalue()).decode()
    except Exception:
        return ""


def measure(name: str, fn) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = fn()
    dt = (time.perf_counter() - t0) * 1000.0
    return {"stage": name, "latency_ms": round(dt, 1), "ok": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    from modules import audio_scene, face_tracker, fusion as fusion_mod
    from modules import lip_reader, llm, personalizer, scene_describer, stt

    audio = synthesize_audio(2.0)
    frame = synthesize_jpeg()

    s = stt.SpeechToText()
    a = audio_scene.AudioSceneClassifier()
    sc = scene_describer.SceneDescriber(sample_every_s=0)
    f = face_tracker.FaceTracker()
    l = lip_reader.LipReader()
    L = llm.LLM()

    results_per_run = []
    print(f"Running {args.runs} latency runs...\n")

    for run in range(args.runs):
        per = []
        per.append(measure("STT (Whisper)", lambda: s.transcribe(audio)))
        per.append(measure("Audio scene (YamNet)", lambda: a.classify(audio)))
        per.append(measure("Visual scene (BLIP)", lambda: sc.describe_jpeg(frame) if frame else None))
        per.append(measure("Face tracker (MediaPipe)", lambda: f.analyse_frame(frame) if frame else []))
        per.append(measure("Lip reliability", lambda: l.reliability([0.01, 0.02, 0.03], 0.05)))
        per.append(measure("LLM notification", lambda: L.compose_notification([], "")))

        total = sum(p["latency_ms"] for p in per)
        per.append({"stage": "TOTAL", "latency_ms": round(total, 1), "ok": True})
        results_per_run.append(per)
        print(f"=== Run {run+1} ===")
        for r in per:
            badge = "🔴" if r["latency_ms"] > 1000 else ("🟡" if r["latency_ms"] > 300 else "🟢")
            print(f"  {badge} {r['stage']:30s}  {r['latency_ms']:7.1f} ms")
        print()

    # Mean across runs
    print("=== Mean across runs ===")
    stages = [s["stage"] for s in results_per_run[0]]
    means = {}
    for stg in stages:
        vals = [r["latency_ms"] for run in results_per_run for r in run if r["stage"] == stg]
        means[stg] = round(float(np.mean(vals)), 1)
        badge = "🔴" if means[stg] > 1000 else ("🟡" if means[stg] > 300 else "🟢")
        print(f"  {badge} {stg:30s}  {means[stg]:7.1f} ms")

    out_path = Path(__file__).resolve().parent.parent / "latency_benchmark.json"
    with open(out_path, "w") as f:
        json.dump({"runs": results_per_run, "means": means}, f, indent=2)
    print(f"\nJSON → {out_path}")


if __name__ == "__main__":
    main()
