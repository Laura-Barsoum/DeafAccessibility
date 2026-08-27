"""
download_esc50.py — Download the ESC-50 environmental sound dataset.

ESC-50 (Piczak 2015):
    - 2,000 environmental audio recordings (5s each, WAV)
    - 50 classes, 40 examples per class
    - 5 official folds for cross-validation
    - Free for research

Download URL: https://github.com/karolpiczak/ESC-50/archive/master.zip

Usage:
    python scripts/download_esc50.py
    # → backend/data/ESC-50/audio/*.wav
    # → backend/data/ESC-50/meta/esc50.csv

Idempotent: skips if already downloaded.
"""
from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
TARGET_ROOT = Path(__file__).resolve().parent.parent / "data" / "ESC-50"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n   → {dest}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024,
        desc="ESC-50",
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def extract(zip_path: Path, target_root: Path) -> None:
    print(f"Extracting → {target_root}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_root.parent)
    # github archive extracts to ESC-50-master; rename to ESC-50
    extracted = target_root.parent / "ESC-50-master"
    if extracted.exists():
        if target_root.exists():
            shutil.rmtree(target_root)
        extracted.rename(target_root)
    print("Done.")


def main() -> None:
    if (TARGET_ROOT / "meta" / "esc50.csv").exists():
        print(f"ESC-50 already present at {TARGET_ROOT}")
        return

    zip_path = TARGET_ROOT.parent / "esc50.zip"
    download(URL, zip_path)
    extract(zip_path, TARGET_ROOT)
    try:
        zip_path.unlink()
    except Exception:
        pass

    # Sanity check
    meta = TARGET_ROOT / "meta" / "esc50.csv"
    audio = TARGET_ROOT / "audio"
    if meta.exists() and audio.exists():
        n = len(list(audio.glob("*.wav")))
        print(f"\n✓ ESC-50 ready: {n} audio files at {audio}")
    else:
        print("✗ Extraction incomplete", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
