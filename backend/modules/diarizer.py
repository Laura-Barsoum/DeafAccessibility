"""
diarizer.py — Speaker diarization wrapper.

Two backends:
    1. Pyannote 3.1 (preferred) — requires HuggingFace token, free for research.
    2. Heuristic energy-based fallback that does silence-based segment
       splitting + face-tracker speaker attribution.

The output format mirrors WhisperX's diarized segments so the rest of the
pipeline doesn't care which backend produced them.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("accessibility.diarizer")


class Diarizer:
    def __init__(self) -> None:
        self._pipeline = None
        log.info("Diarizer initialised (lazy load)")

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        token = os.environ.get("PYANNOTE_TOKEN", "")
        if not token:
            log.info("PYANNOTE_TOKEN not set — using heuristic diarization")
            self._pipeline = "heuristic"
            return
        try:
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
            log.info("Pyannote diarization pipeline loaded")
        except Exception as e:
            log.warning("Pyannote unavailable (%s) — heuristic fallback", e)
            self._pipeline = "heuristic"

    def diarize(
        self, audio_bytes: bytes, sample_rate: int = 16000
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of segments:
            [{"start": float, "end": float, "speaker": "SPEAKER_00"}, ...]
        """
        self._ensure_loaded()
        if self._pipeline == "heuristic":
            return self._heuristic_diarize(audio_bytes, sample_rate)

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                path = tmp.name
            try:
                annotation = self._pipeline(path)
                out = []
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    out.append({
                        "start": float(turn.start),
                        "end": float(turn.end),
                        "speaker": str(speaker),
                    })
                return out
            finally:
                try: os.unlink(path)
                except Exception: pass
        except Exception as e:
            log.warning("Pyannote inference failed: %s", e)
            return self._heuristic_diarize(audio_bytes, sample_rate)

    def _heuristic_diarize(
        self, audio_bytes: bytes, sample_rate: int
    ) -> List[Dict[str, Any]]:
        """Energy-based VAD splitter; treats each spoken segment as one
        speaker turn, defaulting to SPEAKER_00 unless the face tracker
        attributes otherwise downstream."""
        try:
            import librosa
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                path = tmp.name
            try:
                y, sr = librosa.load(path, sr=16000, mono=True)
            finally:
                try: os.unlink(path)
                except Exception: pass

            if len(y) < sr // 2:
                return []
            # Frame-level RMS
            frame_len = int(0.04 * sr)
            hop_len = int(0.02 * sr)
            rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
            threshold = max(0.005, float(np.percentile(rms, 30)))
            voiced = rms > threshold

            segments = []
            in_seg = False
            seg_start = 0
            for i, v in enumerate(voiced):
                t = i * hop_len / sr
                if v and not in_seg:
                    seg_start = t
                    in_seg = True
                elif not v and in_seg:
                    if t - seg_start > 0.3:  # min 300ms
                        segments.append({"start": seg_start, "end": t,
                                         "speaker": "SPEAKER_00"})
                    in_seg = False
            if in_seg:
                segments.append({"start": seg_start, "end": len(y) / sr,
                                 "speaker": "SPEAKER_00"})
            return segments
        except Exception as e:
            log.warning("heuristic diarization failed: %s", e)
            return []
