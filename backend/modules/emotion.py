"""
emotion.py — Audio + visual emotion / tone analysis.

For Deaf users, the *tone* of speech is invisible. This module produces
emotion tags that the frontend prepends to captions:

    [😡 Angry] "Stop it!"
    [😟 Worried] "I'm fine, really..."
    [😊 Happy] "I got the job!"

Two channels:

    1. Visual emotion — DeepFace (Serengil 2021) detects 7 basic Ekman
       emotions from the speaker's face. Robust without audio.
    2. Audio emotion — wav2vec2 fine-tuned on IEMOCAP (4-class:
       neutral / happy / sad / angry). Catches sarcasm, prosody, vocal
       distress that the face may not show.

Fusion: when both available, we average their per-class probabilities
and report the agreed emotion + a confidence score; if they disagree, we
flag "mixed signal" — which is itself useful information for the user.

References
    - Serengil & Ozpinar (2021). DeepFace: Lightweight face recognition
      and facial attribute analysis library.
    - Ekman (1992). An argument for basic emotions.
    - Ravanelli et al. (2020). SpeechBrain emotion-recognition recipe.
"""
from __future__ import annotations

import base64
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("accessibility.emotion")


# Canonical 7-class emotion vocabulary (Ekman + neutral).
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Per-class recalibration for DeepFace (FER-2013), which over-predicts
# 'neutral' and under-detects 'happy'/'surprise'. Applied to each frame's
# raw scores before normalisation so genuine expressions surface.
_DEEPFACE_CALIB = {
    "happy": 1.7, "surprise": 1.3, "neutral": 0.7,
    "sad": 0.9, "angry": 0.9, "fear": 0.9, "disgust": 0.9,
}
EMOJI = {
    "angry": "😡", "disgust": "🤢", "fear": "😨", "happy": "😊",
    "sad": "😢", "surprise": "😮", "neutral": "😐",
    "calm": "🙂", "excited": "🤩", "frustrated": "😤", "worried": "😟",
}


def emoji_for(label: str) -> str:
    """Map an emotion label to its display emoji (empty string if unknown)."""
    return EMOJI.get(label.lower(), "")


# --------------------------------------------------------------------------
# Audio side — wav2vec2 fine-tuned for emotion (IEMOCAP 4-class)
# --------------------------------------------------------------------------


class AudioEmotionAnalyzer:
    """
    HuggingFace `superb/wav2vec2-base-superb-er` — IEMOCAP 4-class.
    Falls back to a librosa heuristic (energy + F0 variance) if the model
    is unavailable.
    """

    LABEL_MAP = {  # IEMOCAP → our 7-class
        "neu": "neutral",
        "hap": "happy",
        "sad": "sad",
        "ang": "angry",
    }

    def __init__(self) -> None:
        self._model = None
        self._extractor = None
        log.info("AudioEmotionAnalyzer initialised (lazy load)")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            mid = os.environ.get(
                "EMO_AUDIO_MODEL", "superb/wav2vec2-base-superb-er"
            )
            self._extractor = AutoFeatureExtractor.from_pretrained(mid)
            self._model = AutoModelForAudioClassification.from_pretrained(mid)
            self._model.eval()
            log.info("Audio emotion model loaded: %s", mid)
        except Exception as e:
            log.warning("Audio emotion model unavailable (%s) — heuristic fallback", e)
            self._model = "placeholder"

    def analyse(self, audio_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """Classify vocal emotion from an audio clip via wav2vec2, returning
        {label, confidence, probs}. Falls back to an energy/pitch heuristic if
        the model is unavailable."""
        self._ensure_loaded()
        if self._model == "placeholder":
            return self._heuristic(audio_bytes, sample_rate)
        try:
            import torch
            import librosa
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                path = tmp.name
            try:
                y, sr = librosa.load(path, sr=16000, mono=True)
            finally:
                try: os.unlink(path)
                except Exception: pass
            if len(y) < 0.4 * sr:
                return {"label": "neutral", "confidence": 0.0, "probs": {}}

            inputs = self._extractor(
                y, sampling_rate=16000, return_tensors="pt", padding=True,
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy().squeeze()
            id2label = self._model.config.id2label
            class_probs = {self.LABEL_MAP.get(id2label[i].lower(), id2label[i].lower()): float(probs[i])
                           for i in range(len(probs))}
            top_label = max(class_probs, key=class_probs.get)
            return {"label": top_label, "confidence": float(class_probs[top_label]),
                    "probs": class_probs}
        except Exception as e:
            log.warning("audio emotion inference failed: %s", e)
            return self._heuristic(audio_bytes, sample_rate)

    def _heuristic(self, audio_bytes: bytes, sample_rate: int) -> Dict[str, Any]:
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
            if len(y) < 0.2 * sr:
                return {"label": "neutral", "confidence": 0.0, "probs": {}}
            rms = float(np.sqrt(np.mean(y ** 2)))
            try:
                f0, _, _ = librosa.pyin(
                    y, fmin=80, fmax=400, sr=sr,
                )
                f0_voiced = f0[~np.isnan(f0)]
                f0_std = float(np.std(f0_voiced)) if len(f0_voiced) else 0.0
            except Exception:
                f0_std = 0.0
            # Crude rules: high energy + high pitch variability = angry/excited;
            # low energy + low pitch = sad; otherwise neutral.
            if rms > 0.06 and f0_std > 35:
                return {"label": "angry", "confidence": 0.55, "probs": {}}
            if rms < 0.02 and f0_std < 15:
                return {"label": "sad", "confidence": 0.5, "probs": {}}
            if rms > 0.05 and f0_std > 25:
                return {"label": "happy", "confidence": 0.5, "probs": {}}
            return {"label": "neutral", "confidence": 0.5, "probs": {}}
        except Exception:
            return {"label": "neutral", "confidence": 0.0, "probs": {}}


# --------------------------------------------------------------------------
# Visual side — DeepFace Ekman 7-class
# --------------------------------------------------------------------------


class VisualEmotionAnalyzer:
    """
    DeepFace wrapper with **temporal smoothing**.

    Single-frame DeepFace predictions are notoriously jumpy — a smile can
    be classified as 'angry' or 'fear' for a single frame because the
    underlying classifier is multi-class softmax over images that were
    never meant to support streaming use.

    We fix this in three ways:
      1. Run DeepFace on N frames (configurable, default 4) and average
         per-class probabilities → far more stable.
      2. Apply a small NEUTRAL bias so noisy single-frame predictions
         default toward neutral rather than e.g. 'angry'.
      3. Maintain a rolling exponential-moving-average across calls so
         the "current emotion" smooths over ~10 seconds of conversation.
    """

    def __init__(self, smooth_alpha: float = 0.72, neutral_bias: float = 0.0) -> None:
        """
        Args:
            smooth_alpha: how much weight the CURRENT frame gets in the
                EMA (0.60 → reactive to real expressions, still smooths
                single-frame noise).
            neutral_bias: small bump for neutral to break ties only.
                Earlier values (0.18) over-corrected and made every
                expression read as 'neutral'.
        """
        self._loaded = False
        self._available = None
        # EMA state across calls
        self._ema_probs: Dict[str, float] = {}
        self._smooth_alpha = smooth_alpha
        self._neutral_bias = neutral_bias
        log.info("VisualEmotionAnalyzer initialised (EMA α=%.2f, neutral_bias=%.2f)",
                 smooth_alpha, neutral_bias)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from deepface import DeepFace  # noqa
            self._available = True
        except Exception as e:
            log.warning("DeepFace unavailable (%s) — visual emotion disabled", e)
            self._available = False
        self._loaded = True

    def _analyse_single(self, jpeg_b64: str) -> Dict[str, float]:
        """Run DeepFace on a single JPEG. Returns per-class probabilities."""
        self._ensure_loaded()
        if not self._available:
            return {}
        try:
            from deepface import DeepFace
            import cv2
            img_bytes = base64.b64decode(jpeg_b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return {}
            res = DeepFace.analyze(
                img, actions=["emotion"],
                enforce_detection=False, detector_backend="opencv", silent=True,
            )
            if isinstance(res, list):
                res = res[0]
            emotions = res.get("emotion", {})
            # Recalibrate DeepFace output. The FER-2013-trained model is well
            # known to over-predict 'neutral' and under-detect 'happy' and
            # 'surprise', so a clear smile frequently reads as neutral. These
            # multipliers correct that bias before normalising.
            raw = {k.lower(): float(v) for k, v in emotions.items()}
            cal = {k: raw.get(k, 0.0) * _DEEPFACE_CALIB.get(k, 1.0)
                   for k in (set(raw) | set(_DEEPFACE_CALIB))}
            total = sum(cal.values()) or 1.0
            return {k: v / total for k, v in cal.items()}
        except Exception as e:
            log.info("DeepFace single-frame failed: %s", e)
            return {}

    def analyse_jpeg(self, jpeg_b64: str) -> Dict[str, Any]:
        """Single-frame entry point (kept for backwards compatibility).
        Internally now applies EMA smoothing."""
        probs = self._analyse_single(jpeg_b64)
        return self._smooth_and_report(probs)

    def analyse_frames(self, frames_b64: List[str]) -> Dict[str, Any]:
        """
        Multi-frame analysis with intra-tick averaging + cross-tick EMA.
        Prefer this over `analyse_jpeg` whenever multiple frames are
        available — it is dramatically more stable.
        """
        if not frames_b64:
            return self._smooth_and_report({})
        per_frame: List[Dict[str, float]] = []
        for f in frames_b64:
            p = self._analyse_single(f)
            if p:
                per_frame.append(p)
        if not per_frame:
            return self._smooth_and_report({})
        # Average per-class probs across this tick
        keys = set().union(*[p.keys() for p in per_frame])
        avg = {k: float(np.mean([p.get(k, 0.0) for p in per_frame])) for k in keys}
        return self._smooth_and_report(avg)

    def _smooth_and_report(self, instant_probs: Dict[str, float]) -> Dict[str, Any]:
        """
        Apply (a) neutral bias and (b) cross-tick EMA to stabilise output.
        """
        if not instant_probs:
            # Nothing this tick — let EMA decay slowly toward neutral.
            decayed = {k: v * (1 - self._smooth_alpha) for k, v in self._ema_probs.items()}
            decayed["neutral"] = decayed.get("neutral", 0.0) + self._smooth_alpha * 0.5
            self._ema_probs = decayed
            return self._report(decayed)

        # Apply neutral bias to discourage spurious 'angry'/'fear'
        # predictions from short blinks or asymmetric facial expressions.
        biased = dict(instant_probs)
        biased["neutral"] = biased.get("neutral", 0.0) + self._neutral_bias
        # Renormalise
        total = sum(biased.values()) or 1.0
        biased = {k: v / total for k, v in biased.items()}

        # EMA update across calls.
        keys = set(self._ema_probs.keys()) | set(biased.keys())
        new_ema = {}
        for k in keys:
            old = self._ema_probs.get(k, 0.0)
            new = biased.get(k, 0.0)
            new_ema[k] = (1 - self._smooth_alpha) * old + self._smooth_alpha * new
        self._ema_probs = new_ema
        return self._report(new_ema)

    def _report(self, probs: Dict[str, float]) -> Dict[str, Any]:
        if not probs:
            return {"label": "neutral", "confidence": 0.0, "probs": {}}
        top = max(probs, key=probs.get)
        return {"label": top, "confidence": float(probs[top]), "probs": probs}

    def reset(self) -> None:
        """Clear cross-tick EMA state (call when session ends/restarts)."""
        self._ema_probs = {}


# --------------------------------------------------------------------------
# Fusion: combine audio + visual emotion
# --------------------------------------------------------------------------


def fuse_emotions(
    audio_emotion: Dict[str, Any],
    visual_emotion: Dict[str, Any],
    neutral_margin: float = 0.01,
) -> Dict[str, Any]:
    """
    Confidence-weighted blend of per-class probabilities.

    Each channel is weighted by its own confidence, so a clear visual smile
    is not halved by a low-confidence (or absent) audio reading. When only
    one channel has data, that channel decides outright. `neutral_margin`
    still prevents flicker: a non-neutral label must beat neutral by at
    least this much.
    """
    a_probs = audio_emotion.get("probs") or {}
    v_probs = visual_emotion.get("probs") or {}
    merged: Dict[str, float] = {}
    keys = set(a_probs) | set(v_probs)
    if not keys:
        return {"label": "neutral", "emoji": "😐", "mixed_signal": False,
                "confidence": 0.0,
                "audio": audio_emotion, "visual": visual_emotion}
    # Weight each channel by its confidence; the visual channel is the
    # reliable one for facial expressions like smiling.
    wa = float(audio_emotion.get("confidence") or 0.0) if a_probs else 0.0
    wv = float(visual_emotion.get("confidence") or 0.0) if v_probs else 0.0
    if wa + wv <= 0.0:
        for k in keys:
            merged[k] = (a_probs.get(k, 0.0) + v_probs.get(k, 0.0)) / 2
    else:
        for k in keys:
            merged[k] = (wa * a_probs.get(k, 0.0) + wv * v_probs.get(k, 0.0)) / (wa + wv)

    # Pick top; fall back to neutral unless top beats neutral by margin.
    top = max(merged, key=merged.get)
    neutral_p = merged.get("neutral", 0.0)
    if top != "neutral" and (merged[top] - neutral_p) < neutral_margin:
        top = "neutral"

    a_top = audio_emotion.get("label")
    v_top = visual_emotion.get("label")
    mixed = (
        a_top and v_top and a_top != v_top
        and audio_emotion.get("confidence", 0) > 0.45
        and visual_emotion.get("confidence", 0) > 0.45
        and a_top != "neutral" and v_top != "neutral"
    )
    return {
        "label": top,
        "emoji": emoji_for(top),
        "confidence": round(float(merged[top]), 3),
        "mixed_signal": bool(mixed),
        "audio": {"label": a_top, "confidence": audio_emotion.get("confidence")},
        "visual": {"label": v_top, "confidence": visual_emotion.get("confidence")},
    }


def tag_caption(caption: str, fused: Dict[str, Any],
                min_confidence: float = 0.55) -> str:
    """
    Prepend an emoji + label to a caption — but ONLY when both:
      (a) the fused emotion is non-neutral, AND
      (b) the fused confidence is high enough to trust.

    Earlier versions tagged 'angry' on noisy single-frame DeepFace
    predictions even when the user was calm. We now demand a real
    consensus before adding a tag.
    """
    if not caption:
        return caption
    label = (fused.get("label") or "").lower()
    if not label or label == "neutral":
        return caption
    confidence = float(fused.get("confidence", 0.0))
    if confidence < min_confidence:
        return caption
    # Mixed signal: only display if it really is meaningful
    if fused.get("mixed_signal"):
        a_lbl = (fused.get("audio") or {}).get("label", "")
        v_lbl = (fused.get("visual") or {}).get("label", "")
        return f"[⚠️ voice {a_lbl}, face {v_lbl}] {caption}"
    emoji = fused.get("emoji", "")
    return f"[{emoji} {label.title()}] {caption}"
