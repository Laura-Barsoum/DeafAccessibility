"""
diarized_stt.py — Whisper transcription + pyannote diarization fusion.

WhisperX-style output without the WhisperX dependency. We keep the existing
`SpeechToText` (faster-whisper) and `Diarizer` (pyannote 3.1) modules and
merge their outputs by time-overlap, then translate raw speaker ids
(SPEAKER_00, SPEAKER_01, …) into friendly names via a small persistent
mapping at `data/speakers.json`.

Output shape (per tick):
    {
      "text": "Sarah said hi",
      "language": "en",
      "segments_with_speakers": [
          {"start": 0.8, "end": 1.4, "text": "hi", "speaker": "SPEAKER_00",
           "speaker_name": "Sarah"},
          ...
      ],
      "speaker": "SPEAKER_00",          # dominant speaker for this tick
      "speaker_name": "Sarah",          # mapped friendly name, or None
    }

If pyannote fails to load or returns nothing, every segment gets
`speaker = None` and the pipeline continues unchanged.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("accessibility.diarized_stt")

# ── Friendly-name persistence ────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_SPEAKERS_PATH = _DATA_DIR / "speakers.json"
_lock = threading.Lock()


def _load_speakers() -> Dict[str, str]:
    if not _SPEAKERS_PATH.exists():
        return {}
    try:
        with open(_SPEAKERS_PATH) as f:
            d = json.load(f)
        if isinstance(d, dict):
            return {str(k): str(v) for k, v in d.items()}
    except Exception:
        return {}
    return {}


def _save_speakers(d: Dict[str, str]) -> None:
    with _lock:
        with open(_SPEAKERS_PATH, "w") as f:
            json.dump(d, f, indent=2)


class DiarizedSTT:
    """
    Lazy-orchestrator combining SpeechToText + Diarizer. Caches a small
    in-memory map of raw_speaker_id → friendly_name; persisted to
    data/speakers.json.
    """

    def __init__(self) -> None:
        self._stt = None
        self._diarizer = None
        self._speaker_names: Dict[str, str] = _load_speakers()
        # Track which raw IDs we've ever observed (so the UI can prompt
        # the user to name a brand-new speaker).
        self._seen_speakers: set = set(self._speaker_names.keys())

    # ── Lazy loaders ────────────────────────────────────────────────
    def _ensure_stt(self):
        if self._stt is None:
            from .stt import SpeechToText
            self._stt = SpeechToText()
        return self._stt

    def _ensure_diarizer(self):
        if self._diarizer is None:
            from .diarizer import Diarizer
            self._diarizer = Diarizer()
        return self._diarizer

    # ── Friendly-name helpers ───────────────────────────────────────
    def name_speaker(self, raw_id: str, friendly_name: str) -> None:
        """Assign `friendly_name` to a raw diarizer label (e.g.
        SPEAKER_00 → 'Sarah'). Persisted to disk."""
        raw_id = raw_id.strip()
        friendly_name = friendly_name.strip()
        if not raw_id or not friendly_name:
            return
        self._speaker_names[raw_id] = friendly_name
        self._seen_speakers.add(raw_id)
        _save_speakers(self._speaker_names)

    def get_speaker_name(self, raw_id: Optional[str]) -> Optional[str]:
        if not raw_id:
            return None
        return self._speaker_names.get(raw_id)

    def list_speakers(self) -> Dict[str, Any]:
        """All known mappings + every raw id we've ever seen this run."""
        return {
            "mappings": dict(self._speaker_names),
            "seen_unnamed": sorted(
                s for s in self._seen_speakers if s not in self._speaker_names
            ),
        }

    def forget_speaker(self, raw_id: str) -> None:
        self._speaker_names.pop(raw_id, None)
        _save_speakers(self._speaker_names)

    # ── Main entrypoint ─────────────────────────────────────────────
    def transcribe_with_speakers(
        self, audio_bytes: bytes, sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        stt = self._ensure_stt()
        stt_out = stt.transcribe(audio_bytes, sample_rate=sample_rate,
                                 with_timestamps=True)
        text = stt_out.get("text", "")
        segments = stt_out.get("segments", [])
        language = stt_out.get("language", "en")

        # Short-circuit: empty transcript → don't bother diarizing.
        if not text or not segments:
            return {
                "text": text,
                "language": language,
                "segments_with_speakers": [],
                "speaker": None,
                "speaker_name": None,
            }

        # Pyannote diarization — graceful degradation.
        diar: List[Dict[str, Any]] = []
        try:
            diar = self._ensure_diarizer().diarize(audio_bytes, sample_rate)
        except Exception as e:
            log.info("Diarizer raised — falling back to speaker=None (%s)", e)
            diar = []

        merged = self._merge(segments, diar)
        # Pick the dominant speaker for the tick: the one with the most
        # accumulated speech time.
        per_speaker: Dict[str, float] = {}
        for s in merged:
            sp = s.get("speaker")
            if sp:
                per_speaker[sp] = per_speaker.get(sp, 0.0) + max(
                    0.0, s["end"] - s["start"],
                )
        dominant = max(per_speaker, key=per_speaker.get) if per_speaker else None
        if dominant:
            self._seen_speakers.add(dominant)
        for s in merged:
            sp = s.get("speaker")
            if sp:
                self._seen_speakers.add(sp)
            s["speaker_name"] = self.get_speaker_name(sp)

        return {
            "text": text,
            "language": language,
            "segments_with_speakers": merged,
            "speaker": dominant,
            "speaker_name": self.get_speaker_name(dominant),
        }

    # ── Merge logic ─────────────────────────────────────────────────
    @staticmethod
    def _merge(
        segments: List[Dict[str, Any]],
        diar: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        For each Whisper segment, find the diarizer turn with the most
        time-overlap and tag the segment with that speaker id.
        Returns a NEW list — does not mutate input.
        """
        out: List[Dict[str, Any]] = []
        for seg in segments:
            s_start, s_end = float(seg["start"]), float(seg["end"])
            best_speaker = None
            best_overlap = 0.0
            for t in diar:
                t_start, t_end = float(t["start"]), float(t["end"])
                ov = max(0.0, min(s_end, t_end) - max(s_start, t_start))
                if ov > best_overlap:
                    best_overlap = ov
                    best_speaker = t["speaker"]
            out.append({
                "start": s_start,
                "end":   s_end,
                "text":  seg.get("text", ""),
                "speaker": best_speaker,
            })
        return out

    # ── Status ──────────────────────────────────────────────────────
    def status(self) -> Dict[str, Any]:
        return {
            "pyannote_token_set": bool(os.environ.get("PYANNOTE_TOKEN")),
            "mappings_count": len(self._speaker_names),
            "seen_count": len(self._seen_speakers),
        }


# ── Module-level singleton (matches existing pattern in this codebase) ─
_diarized_stt_singleton: Optional[DiarizedSTT] = None


def get_diarized_stt() -> DiarizedSTT:
    global _diarized_stt_singleton
    if _diarized_stt_singleton is None:
        _diarized_stt_singleton = DiarizedSTT()
    return _diarized_stt_singleton
