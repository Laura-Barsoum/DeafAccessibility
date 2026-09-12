"""
model_cache.py — Persistent, self-healing cache for TF-Hub models.

By default TF-Hub caches downloaded models under the system temp directory.
That location is not durable: the OS may purge its files, and an interrupted
download can leave the same end state, a cache folder whose directory
structure exists but whose model files do not. TF-Hub treats the existing
folder as a cache hit, the load fails, and callers fall back silently.

YamNet was found in exactly this state during evaluation (an empty cache entry
dated 9 August 2026), which meant personal sound recognition had been running
on the weaker spectral fallback embedding without any visible error.

This module moves the cache to backend/data/model_cache/tfhub, which the OS
does not purge, and removes incomplete entries before a load so that a damaged
entry triggers a clean re-download instead of a silent fallback.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger("accessibility.model_cache")

_BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TFHUB_CACHE = _BACKEND_DIR / "data" / "model_cache" / "tfhub"


def _is_complete(entry: Path) -> bool:
    """A TF-Hub SavedModel entry is usable only if its graph file exists."""
    return (entry / "saved_model.pb").exists() or (entry / "saved_model.pbtxt").exists()


def prepare_tfhub_cache() -> str:
    """Point TF-Hub at a persistent cache and delete incomplete entries.

    An explicit TFHUB_CACHE_DIR set by the user is respected. In-progress
    downloads (TF-Hub extracts into '<hash>.<uuid>.tmp' and holds a
    '<hash>.lock' file) are never touched, so concurrent processes are safe.
    Returns the cache path. Safe to call repeatedly.
    """
    path = Path(os.environ.get("TFHUB_CACHE_DIR") or DEFAULT_TFHUB_CACHE)
    path.mkdir(parents=True, exist_ok=True)
    os.environ["TFHUB_CACHE_DIR"] = str(path)
    for entry in path.iterdir():
        if not entry.is_dir() or ".tmp" in entry.name:
            continue
        if not _is_complete(entry):
            log.warning("removing incomplete TF-Hub cache entry %s "
                        "(it would otherwise fail to load and force a silent fallback)",
                        entry.name)
            shutil.rmtree(entry, ignore_errors=True)
    return str(path)
