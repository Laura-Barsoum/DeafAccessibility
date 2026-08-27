"""
lip_reader.py — Caption reliability scorer from lip motion.

SCOPE, STATED PLAINLY: this module does NOT perform lip reading. No
audio-visual speech recognition model is loaded or run. It computes a
heuristic confidence score in [0, 1] for the *audio* transcript, based on
how much mouth movement was observed, and flags when the audio channel
looks unreliable (for example, high background noise with no visible
speaker). Any claim of lip-reading capability would be inaccurate.

Rationale for the scope decision: true audio-visual speech recognition
would wrap AV-HuBERT (Shi et al., 2022), which learns a joint
audio-visual representation robust to acoustic noise. That would need a
large checkpoint and a mouth-region extraction pipeline, which is beyond
the scope and hardware budget of this project. The reliability score is
the useful part that is achievable: it tells the fusion layer how much to
trust a transcript, which is what the interface actually consumes.

The lightweight wrapper therefore provides:
    - When a Whisper transcript and a sequence of mouth-region images are
      both available, the lip channel is used to *score the confidence*
      of the audio transcript, and to flag when audio is unreliable
      (background noise high, speaker not visible).
    - In a future iteration, AV-HuBERT or VATLM weights are loaded and
      run end-to-end.

This module's main function is therefore a *reliability score* in [0, 1]
for the current audio transcript, given the lip motion observed.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("accessibility.lip")


class LipReader:
    """
    Lightweight reliability scorer for audio transcripts.

    Inputs:
        - mar_series: list of mouth-aspect-ratio values during the segment
        - audio_rms: RMS energy of the audio segment
        - background_noise: estimated background noise level

    Output:
        reliability ∈ [0, 1]
            1.0  → audio + lip both strong → trust transcript
            0.5  → ambiguous
            0.0  → speaker not visible OR mouth not moving but audio claims speech
    """

    def __init__(self) -> None:
        self._av_hubert = None
        log.info("LipReader initialised (lite scorer; AV-HuBERT lazy-load placeholder)")

    def _ensure_av_hubert(self) -> None:
        # Placeholder for full AV-HuBERT. Loading not yet implemented since the
        # checkpoints require a HuggingFace gated download. See README.
        self._av_hubert = "placeholder"

    def reliability(
        self, mar_series: List[float], audio_rms: float,
        background_noise: float = 0.0,
    ) -> Dict[str, Any]:
        """Score how trustworthy the audio transcript is given lip motion."""
        if not mar_series:
            return {
                "reliability": 0.5,
                "lip_motion": 0.0,
                "audio_strength": float(audio_rms),
                "verdict": "no_face_visible",
            }

        lip_motion = float(np.std(mar_series))
        audio_strength = float(audio_rms)
        snr_estimate = audio_strength / max(background_noise + 1e-6, 0.001)

        # Heuristic: if audio claims strong speech but lips aren't moving,
        # flag low reliability (could be background TV, off-screen speaker).
        if audio_strength > 0.05 and lip_motion < 0.005:
            verdict = "audio_without_visible_speaker"
            reliability = 0.4
        # If lips moving and audio strong → high confidence
        elif lip_motion > 0.02 and audio_strength > 0.02:
            verdict = "audio_visual_aligned"
            reliability = min(1.0, 0.7 + 0.3 * min(snr_estimate / 5.0, 1.0))
        # Lips moving but audio weak → maybe whisper / partial — moderate
        elif lip_motion > 0.02 and audio_strength <= 0.02:
            verdict = "lips_moving_audio_weak"
            reliability = 0.55
        else:
            verdict = "low_activity"
            reliability = 0.5

        return {
            "reliability": round(reliability, 3),
            "lip_motion": round(lip_motion, 4),
            "audio_strength": round(audio_strength, 4),
            "snr_estimate": round(snr_estimate, 3),
            "verdict": verdict,
        }

    def fuse_transcript(
        self, audio_text: str, reliability: float
    ) -> Dict[str, Any]:
        """
        Annotate a Whisper transcript with a reliability flag.
        In the AV-HuBERT version this would *correct* the transcript using
        lip-reading; here we simply tag it.
        """
        if reliability >= 0.75:
            return {"text": audio_text, "confidence": "high"}
        if reliability >= 0.5:
            return {"text": audio_text, "confidence": "medium"}
        return {
            "text": f"{audio_text}  [⚠ low audio-visual confidence]",
            "confidence": "low",
        }
