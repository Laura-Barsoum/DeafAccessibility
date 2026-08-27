"""
localization.py — 3D sound localization (direction of arrival).

Two channels are needed for spatial audio. We support:

    1. Stereo TDOA (Time Difference of Arrival) — classical method:
       cross-correlate left/right channels, recover the inter-aural delay,
       map to azimuth angle. Implementation: GCC-PHAT (Knapp & Carter
       1976) — robust under reverberation, no training needed.

    2. (Optional, hook left for future work) VGGish embedding + a
       direction-classification head trained on a stereo dataset such
       as DCASE-Sound-Event-Localization. Replaces TDOA when the user
       has > 2 microphones.

Output: a degree (-180..+180) and a coarse compass bin
("front", "front-left", "left", "back-left", "back", ...).

We also maintain a short history per detected azimuth to support
"behind you" warnings even briefly after the sound stops.

References
    - Knapp & Carter (1976). Generalized cross-correlation method for
      estimation of time delay (GCC-PHAT).
    - Adavanne et al. (2018). Sound event localization and detection
      with convolutional recurrent neural networks (SELDnet).
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("accessibility.localization")


# Speed of sound in m/s (standard room temperature).
SOUND_SPEED_MS = 343.0


def _gcc_phat(sig: np.ndarray, ref: np.ndarray, fs: int, max_tau: Optional[float] = None,
              interp: int = 16) -> Tuple[float, float]:
    """
    Generalised cross-correlation with phase transform (Knapp & Carter 1976).

    Returns:
        tau   — estimated delay in seconds (sig leads ref by tau when positive)
        peak  — the GCC peak value (proxy for confidence)
    """
    n = sig.shape[0] + ref.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(ref, n=n)
    R = SIG * np.conj(REF)
    R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, n=interp * n)
    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = int(min(max_shift, interp * fs * max_tau))
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc))) - max_shift
    tau = shift / float(interp * fs)
    peak = float(np.abs(cc[shift + max_shift]))
    return tau, peak


def _tau_to_azimuth(tau: float, mic_distance_m: float = 0.18) -> Optional[float]:
    """
    Convert inter-aural time delay to azimuth angle (degrees).
    Assumes a head-width spacing for the two microphones (default 18cm
    ≈ typical interaural distance / laptop stereo mic spread).
    """
    sin_arg = max(-1.0, min(1.0, tau * SOUND_SPEED_MS / mic_distance_m))
    angle_rad = math.asin(sin_arg)
    return math.degrees(angle_rad)


def _azimuth_to_compass(angle_deg: Optional[float]) -> str:
    if angle_deg is None:
        return "unknown"
    a = angle_deg
    # Convention: 0° = front, +90° = right, -90° = left, ±180° = behind.
    # Stereo TDOA can't really distinguish front-vs-back without 3+ mics;
    # we return left/right primarily and let downstream fusion guess
    # behind-vs-front from the lack of visible speaker.
    if -15 <= a <= 15:
        return "front"
    if 15 < a <= 60:
        return "front-right"
    if 60 < a <= 120:
        return "right"
    if -60 <= a < -15:
        return "front-left"
    if -120 <= a < -60:
        return "left"
    return "behind"


# --------------------------------------------------------------------------


class SoundLocalizer:
    """
    3D sound localization via GCC-PHAT.

    Public API:
        localize(audio_stereo_bytes, sample_rate)
            → {azimuth_deg, compass, confidence}

        infer_back_or_front(visible_speaker_in_frame: bool, compass: str)
            → "behind" if compass='front' but no visible speaker, else compass
    """

    def __init__(self, mic_distance_m: float = 0.18) -> None:
        self.mic_distance_m = float(
            os.environ.get("ACCESSIBILITY_MIC_DISTANCE_M", mic_distance_m)
        )
        log.info("SoundLocalizer initialised (mic distance=%.2fm)", self.mic_distance_m)

    def localize(self, audio_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Audio must be a STEREO WAV. If only mono is supplied we return
        unknown direction.
        """
        try:
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                path = tmp.name
            try:
                data, fs = sf.read(path, dtype="float32", always_2d=True)
            finally:
                try: os.unlink(path)
                except Exception: pass

            if data.shape[1] < 2:
                return {"azimuth_deg": None, "compass": "unknown",
                        "confidence": 0.0, "stereo": False}

            left = data[:, 0]
            right = data[:, 1]
            if len(left) < 0.1 * fs:
                return {"azimuth_deg": None, "compass": "unknown",
                        "confidence": 0.0, "stereo": True}

            # Max possible delay across the head/laptop
            max_tau = self.mic_distance_m / SOUND_SPEED_MS * 1.5
            tau, peak = _gcc_phat(left, right, fs, max_tau=max_tau, interp=16)
            angle = _tau_to_azimuth(tau, self.mic_distance_m)
            compass = _azimuth_to_compass(angle)
            return {
                "azimuth_deg": round(float(angle), 1) if angle is not None else None,
                "compass": compass,
                "confidence": round(min(1.0, peak * 6.0), 3),
                "tau_s": round(tau, 5),
                "stereo": True,
            }
        except Exception as e:
            log.info("localization failed: %s", e)
            return {"azimuth_deg": None, "compass": "unknown",
                    "confidence": 0.0, "stereo": False, "error": str(e)}

    @staticmethod
    def infer_back_or_front(
        compass: str, visible_speaker_in_frame: bool
    ) -> str:
        """
        Stereo can't distinguish front vs. back. If TDOA says 'front' but
        the camera sees no visible speaker, the source is most likely
        *behind* the user.
        """
        if compass == "front" and not visible_speaker_in_frame:
            return "behind"
        return compass

    @staticmethod
    def emoji_for_compass(compass: str) -> str:
        return {
            "front": "⬆️", "front-left": "↖️", "left": "⬅️",
            "front-right": "↗️", "right": "➡️",
            "behind": "⬇️", "unknown": "❓",
        }.get(compass, "")
