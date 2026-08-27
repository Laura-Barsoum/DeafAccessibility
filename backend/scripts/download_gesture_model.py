"""
download_gesture_model.py — Download the MediaPipe Gesture Recognizer
pre-trained model (a real CNN trained by Google on ~30k+ hand images).

This is a TRUE pre-trained model — the kind the assignment brief
explicitly asks us to evaluate and orchestrate alongside Whisper,
YamNet, BLIP and YOLO. It is NOT a model we train ourselves.

Model card: https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer
7 built-in gesture classes:
    Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou

Once downloaded the file lives at:
    backend/data/gesture_recognizer.task
and is consumed by modules/sign_language.py at runtime.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
TARGET = Path(__file__).resolve().parent.parent / "data" / "gesture_recognizer.task"


def main() -> None:
    if TARGET.exists():
        size = TARGET.stat().st_size
        print(f"✓ Gesture model already present: {TARGET} ({size:,} bytes)")
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe GestureRecognizer pre-trained model")
    print(f"  from {URL}")
    print(f"  to   {TARGET}")
    r = requests.get(URL, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(TARGET, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="gesture_recognizer.task",
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    size = TARGET.stat().st_size
    print(f"\n✓ Done: {size:,} bytes")


if __name__ == "__main__":
    main()
