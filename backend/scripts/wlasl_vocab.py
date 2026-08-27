"""
wlasl_vocab.py — Build a usable vocabulary list from the WLASL v0.3 manifest.

The full WLASL JSON manifest at
    backend/data/WLASL-master/start_kit/WLASL_v0.3.json
defines 2000 ASL glosses with 21,083 video instances pointing to YouTube
URLs. We don't need to download the videos to USE the vocabulary — we
just need the gloss strings.

This script:
    1. Loads the JSON manifest.
    2. Keeps glosses that have ≥ 5 video instances (more reliable signs).
    3. Filters to a sensible top-N (default 100) by instance count.
    4. Writes the result to backend/data/wlasl_vocab.json — consumed by
       sign_language.py at startup.

Usage:
    python scripts/wlasl_vocab.py
    python scripts/wlasl_vocab.py --top 200
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "WLASL-master" / "start_kit" / "WLASL_v0.3.json"
OUT = ROOT / "data" / "wlasl_vocab.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=100,
                   help="Keep top-N glosses by instance count.")
    p.add_argument("--min-instances", type=int, default=5,
                   help="Drop glosses with fewer than this many videos.")
    args = p.parse_args()

    if not MANIFEST.exists():
        print(f"WLASL manifest not found at {MANIFEST}", file=sys.stderr)
        print("Make sure backend/data/WLASL-master/ exists.", file=sys.stderr)
        sys.exit(1)

    with open(MANIFEST) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} glosses from WLASL")

    # Filter + sort by instance count
    pairs = [(e["gloss"], len(e["instances"])) for e in data]
    pairs = [p for p in pairs if p[1] >= args.min_instances]
    pairs.sort(key=lambda x: -x[1])
    pairs = pairs[: args.top]
    vocab = [g for g, _ in pairs]

    summary = {
        "n_glosses": len(vocab),
        "min_instances": args.min_instances,
        "vocab": vocab,
        "vocab_with_counts": [{"gloss": g, "instances": n} for g, n in pairs],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Wrote {len(vocab)} glosses → {OUT}")
    print(f"  First 10: {vocab[:10]}")


if __name__ == "__main__":
    main()
