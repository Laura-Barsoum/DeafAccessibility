"""
stt.py — Speech-to-text using faster-whisper.

Same lazy-load pattern as the Cognitive Decline project. Whisper is the
SOTA open speech recognition model (Radford et al. 2023). For the
accessibility use case we prefer the `small` size by default — it gives
~95% of `large` accuracy at ~10x speed on CPU, important when streaming
audio in real time for live captions.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

log = logging.getLogger("accessibility.stt")


# Whisper is famous for hallucinating these stock phrases on near-silent
# or very short audio chunks. We aggressively suppress them to keep the
# captions panel honest. See: https://github.com/openai/whisper/discussions/679
WHISPER_HALLUCINATIONS = {
    "thank you.",
    "thank you for watching.",
    "thanks for watching.",
    "thanks for watching!",
    "thanks.",
    "thanks!",
    "thank you for watching!",
    "you",
    "you.",
    ".",
    "bye.",
    "bye!",
    ".",
    "♪",
    "[music]",
    "[applause]",
    "(applause)",
    "(music)",
    "subtitles by",
    "subscribe to my channel",
    "please subscribe",
}


def collapse_repetition(text: str) -> str:
    """Collapse a phrase repeated 3+ times in a row into a single copy.

    Whisper sometimes loops, e.g. "I'm going to get you a little bit of a
    little bit of a little bit of ...". The decoding guards usually prevent
    this, but this is a cheap safety net. It scans for any block of 1-6 words
    that repeats consecutively and keeps only one copy.
    """
    words = text.split()
    n = len(words)
    if n < 9:
        return text
    out: list = []
    i = 0
    while i < n:
        collapsed = False
        for L in range(1, 7):
            if i + 2 * L > n:
                continue
            block = words[i:i + L]
            reps, j = 1, i + L
            while j + L <= n and words[j:j + L] == block:
                reps += 1
                j += L
            if reps >= 3:
                out.extend(block)   # keep a single copy of the looped phrase
                i = j
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return " ".join(out)


def is_whisper_hallucination(text: str) -> bool:
    """Return True if Whisper likely hallucinated on silence/noise."""
    if not text:
        return False
    norm = text.strip().lower()
    if norm in WHISPER_HALLUCINATIONS:
        return True
    # Whisper sometimes appends "thank you" or "thanks for watching"
    # to otherwise empty transcripts.
    if norm.endswith("thanks for watching.") or norm.endswith("thank you for watching."):
        return True
    return False


class SpeechToText:
    """Whisper speech-to-text wrapper (faster-whisper backend).

    Lazy-loads the model on first use, runs int8 inference across all CPU
    cores, and applies repetition-loop guards plus a hallucination filter so
    near-silent audio does not produce stock phrases. Two instances exist in
    the server: the accurate default model, and a 'tiny' one for streaming
    partial captions.
    """

    def __init__(self, model_size: str = "base") -> None:
        # Default 'base' (~74 MB, fast on CPU). Override via
        # ACCESSIBILITY_WHISPER_SIZE env var: tiny | base | small | medium.
        # `tiny` is ~10x faster than `small` at the cost of ~3-5% WER —
        # often a worthwhile trade for real-time captioning.
        self.model_size = os.environ.get("ACCESSIBILITY_WHISPER_SIZE", model_size)
        self._model = None
        log.info("STT initialised (lazy load, model=%s)", self.model_size)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            import os as _os
            # Use all CPU cores — on a multi-core laptop this roughly halves
            # transcription time versus the single-thread default.
            threads = max(4, _os.cpu_count() or 4)
            self._model = WhisperModel(
                self.model_size, device="cpu", compute_type="int8",
                cpu_threads=threads,
            )
            log.info("Whisper %s loaded (cpu_threads=%d)", self.model_size, threads)
        except Exception as e:
            log.warning("Whisper unavailable (%s) — STT disabled", e)
            self._model = "placeholder"

    def transcribe(
        self, audio_bytes: bytes, sample_rate: int = 16000,
        with_timestamps: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
              'text': str,
              'segments': [{'start': float, 'end': float, 'text': str}, ...],
              'language': str,
            }
        """
        self._ensure_loaded()
        if self._model == "placeholder":
            return {"text": "", "segments": [], "language": "en"}

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            path = tmp.name

        try:
            # First attempt with conservative VAD (skips obvious silence
            # but doesn't drop short utterances).
            segments, info = self._model.transcribe(
                path,
                beam_size=2,
                language="en",
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 250,  # was 500 — too aggressive
                    "speech_pad_ms": 200,
                    "threshold": 0.35,               # was default 0.5 — accept quieter speech
                },
                word_timestamps=with_timestamps,
                condition_on_previous_text=False,
                no_speech_threshold=0.45,            # default 0.6 — be more permissive
                # Repetition-loop guards. Greedy decoding (single temperature)
                # gets stuck repeating phrases like "a little bit of a little
                # bit of...". A temperature fallback list lets Whisper retry at
                # higher temperature when the output is too repetitive
                # (compression_ratio_threshold) or low-confidence.
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                compression_ratio_threshold=2.4,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
            )
            seg_list = list(segments)  # consume generator so we can check emptiness

            # If VAD ate everything, retry once WITHOUT VAD for safety.
            if not seg_list:
                segments2, _ = self._model.transcribe(
                    path, beam_size=2, language="en", vad_filter=False,
                    word_timestamps=with_timestamps,
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    compression_ratio_threshold=2.4,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    condition_on_previous_text=False,
                )
                seg_list = list(segments2)
            segments = seg_list  # rebind so the downstream loop works
            seg_list: List[Dict[str, Any]] = []
            full_text_parts: List[str] = []
            for seg in segments:
                seg_dict = {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text.strip(),
                }
                seg_list.append(seg_dict)
                full_text_parts.append(seg.text.strip())
            text = " ".join(full_text_parts).strip()
            # Safety net against any residual repetition loop.
            text = collapse_repetition(text)

            # FILTER WHISPER HALLUCINATIONS on near-silent / short audio.
            # If VAD removed most of the audio AND we got a stock phrase
            # like "Thank you" / "Thanks for watching", suppress it.
            audio_dur = float(getattr(info, "duration", 0.0)) or 1.0
            kept_dur = float(getattr(info, "duration_after_vad", audio_dur))
            vad_ate = 1.0 - (kept_dur / audio_dur) if audio_dur > 0 else 0.0
            if (vad_ate > 0.6 or kept_dur < 0.8) and is_whisper_hallucination(text):
                log.info(
                    "STT: suppressing likely hallucination %r (vad_ate=%.0f%%, kept=%.1fs)",
                    text, vad_ate * 100, kept_dur,
                )
                return {"text": "", "segments": [], "language": info.language}
            # Also: if the entire transcript is just a one-word stock phrase, drop it.
            if is_whisper_hallucination(text):
                log.info("STT: suppressing stock-phrase hallucination %r", text)
                return {"text": "", "segments": [], "language": info.language}

            log.info("STT: %d chars in %d segments (lang=%s)",
                     len(text), len(seg_list), info.language)
            return {"text": text, "segments": seg_list, "language": info.language}
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
