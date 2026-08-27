"""
tts.py — Text-to-speech for the Deaf/HoH Accessibility Assistant.

Used by two pipelines:
    1. Sign-language → speech (Deaf user signs; LLaMA polishes; this speaks
       the result aloud to a hearing companion).
    2. "Speak the last N captions" review feature in the session report.

Backend priority (first that works wins):
    1. ElevenLabs (set ELEVENLABS_API_KEY; high-quality neural TTS).
    2. OpenAI (set OPENAI_API_KEY).
    3. macOS `say` (zero-config local fallback).
    4. gTTS (free remote fallback).

The output is always MP3 base64 — the browser plays it directly via
`<audio src="data:audio/mpeg;base64,...">`.

References
    - ElevenLabs API documentation, https://elevenlabs.io/docs
    - macOS `say(1)` man page.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional

log = logging.getLogger("accessibility.tts")


class TTSEngine:
    """
    Same architecture as the Empathica project's tts.py — provider chain
    with disk caching keyed by text+voice+version. Cache version
    invalidates if you change the voice or model.
    """

    def __init__(self) -> None:
        self._elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        self._openai_keys = [k for k in [
            os.environ.get("OPENAI_API_KEY"),
            os.environ.get("OPENAI_API_KEY_2"),
        ] if k]
        # Per-process circuit breakers (don't keep retrying a 402)
        self._elevenlabs_disabled = False
        self._openai_disabled = False
        # Track *why* the last ElevenLabs call failed — exposed via status()
        self._eleven_last_error = ""

        # On-disk cache for repeated phrases (e.g. "doorbell")
        self._cache_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "tts_cache",
        )
        os.makedirs(self._cache_dir, exist_ok=True)
        self._cache_version = os.environ.get(
            "ACCESSIBILITY_TTS_CACHE_VERSION", "v1",
        )

        if self._elevenlabs_key:
            log.info("TTS: ElevenLabs ENABLED (key set, %d chars)",
                     len(self._elevenlabs_key))
        else:
            log.warning(
                "TTS: ElevenLabs DISABLED — no ELEVENLABS_API_KEY in env. "
                "Voice will fall back to macOS `say` (robotic). "
                "Set ELEVENLABS_API_KEY=... in .env for natural neural voice."
            )

    def status(self) -> Dict[str, Any]:
        """Returns which TTS backends are currently usable. Surfaced to the
        frontend so the UI can warn 'voice is robotic — set an ElevenLabs key'."""
        return {
            "elevenlabs_configured": bool(self._elevenlabs_key),
            "elevenlabs_disabled": self._elevenlabs_disabled,
            "elevenlabs_last_error": self._eleven_last_error,
            "openai_configured": bool(self._openai_keys),
            "voice_id": os.environ.get(
                "ELEVENLABS_VOICE_ID", "XB0fDUnXU5powFXDhCwa"),
        }

    # ------------------------------------------------------------------

    def speak(self, text: str, voice_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize text → returns {"audio_b64": <mp3 base64>, "source": str}.
        Returns {"audio_b64": "", "error": str} on total failure.
        """
        t0 = time.perf_counter()
        if not text or not text.strip():
            return {"audio_b64": "", "source": "none", "latency_ms": 0}

        # Cache lookup
        key = hashlib.md5(
            f"{self._cache_version}|{voice_id or 'default'}|{text}".encode()
        ).hexdigest()
        cache_path = os.path.join(self._cache_dir, f"{key}.mp3")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data = f.read()
                if len(data) > 200:
                    return {
                        "audio_b64": base64.b64encode(data).decode(),
                        "source": "cache",
                        "latency_ms": int((time.perf_counter() - t0) * 1000),
                    }
            except Exception:
                pass

        # Chain
        raw = b""
        source = "none"
        if self._elevenlabs_key and not self._elevenlabs_disabled:
            raw = self._eleven(text, voice_id)
            if raw: source = "elevenlabs"
        if not raw and self._openai_keys and not self._openai_disabled:
            raw = self._openai(text, voice_id)
            if raw: source = "openai"
        if not raw and os.uname().sysname == "Darwin":
            raw = self._mac_say(text)
            if raw: source = "macos-say"
        if not raw:
            raw = self._gtts(text)
            if raw: source = "gtts"

        if not raw:
            return {"audio_b64": "", "source": "none", "error": "all backends failed",
                    "latency_ms": int((time.perf_counter() - t0) * 1000)}

        # Cache for repeats
        try:
            with open(cache_path, "wb") as f:
                f.write(raw)
        except Exception:
            pass

        return {
            "audio_b64": base64.b64encode(raw).decode(),
            "source": source,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    # ElevenLabs premade voices that are FREE on every account.
    # Charlotte and many others have been moved to the "library" (paid tier).
    # These four have been stable free-tier premade voices for years.
    _FREE_TIER_FALLBACK_VOICES = [
        ("21m00Tcm4TlvDq8ikWAM", "Rachel"),    # warm, calm female
        ("AZnzlk1XvdvUeBnXmlld", "Domi"),      # confident female
        ("EXAVITQu4vr4xnSDxMaL", "Sarah"),     # gentle female
        ("ErXwobaYiN019PkySvjV", "Antoni"),    # warm male
        ("TxGEqnHWrfWFTfGW9XjX", "Josh"),      # young male
    ]

    def _eleven(self, text: str, voice_id: Optional[str]) -> bytes:
        """
        Synthesize via ElevenLabs. Voice settings tuned for natural delivery:
          - stability LOW (0.35) — expressive, not monotone
          - similarity_boost HIGH (0.85) — keeps voice character
          - style 0.30 — emotional inflection
          - speaker_boost on — fuller chest tone

        If the requested voice returns 402 "library voice — paid plan required",
        automatically retry with free-tier premade voices in order.
        """
        try:
            import requests
        except ImportError:
            return b""

        # Build the voice queue: user's choice first, then free-tier fallbacks
        primary = (voice_id
                   or os.environ.get("ELEVENLABS_VOICE_ID")
                   or "21m00Tcm4TlvDq8ikWAM")  # Rachel
        queue = [(primary, "primary")] + [
            v for v in self._FREE_TIER_FALLBACK_VOICES if v[0] != primary
        ]

        model = os.environ.get(
            "ACCESSIBILITY_ELEVEN_MODEL", "eleven_multilingual_v2",
        )

        for vid, vname in queue:
            try:
                import requests
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
                r = requests.post(url, timeout=20,
                    headers={
                        "Accept": "audio/mpeg",
                        "Content-Type": "application/json",
                        "xi-api-key": self._elevenlabs_key,
                    },
                    json={
                        "text": text,
                        "model_id": model,
                        "voice_settings": {
                            "stability": 0.35,
                            "similarity_boost": 0.85,
                            "style": 0.30,
                            "use_speaker_boost": True,
                        },
                    })
                if r.status_code == 200:
                    self._eleven_last_error = ""
                    if vname != "primary":
                        log.info("ElevenLabs: used fallback voice %s (%s)", vname, vid)
                    return r.content

                # 402 = "library voice — paid plan required" — try next fallback
                err = r.text[:200]
                self._eleven_last_error = f"HTTP {r.status_code}: {err}"
                if r.status_code == 402 and "library voice" in err.lower():
                    log.warning("ElevenLabs voice %s is paid-only; trying next fallback", vid)
                    continue
                # Other 4xx errors — abort the whole chain
                log.warning("ElevenLabs %d: %s", r.status_code, err)
                if r.status_code in (401, 403, 429):
                    self._elevenlabs_disabled = True
                    log.error(
                        "ElevenLabs disabled for this session (HTTP %d). "
                        "Free tier exhausted? Check usage at elevenlabs.io.",
                        r.status_code,
                    )
                return b""
            except Exception as e:
                self._eleven_last_error = f"exception: {e}"
                log.warning("ElevenLabs %s exception: %s", vname, e)
                continue

        # All voices exhausted
        log.error("ElevenLabs: all %d fallback voices failed", len(queue))
        self._elevenlabs_disabled = True
        return b""

    def _openai(self, text: str, voice_id: Optional[str]) -> bytes:
        try:
            import requests
            for key in self._openai_keys:
                r = requests.post(
                    "https://api.openai.com/v1/audio/speech",
                    timeout=12,
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    json={"model": "tts-1", "input": text,
                          "voice": voice_id or os.environ.get(
                              "ACCESSIBILITY_OPENAI_TTS_VOICE", "nova")},
                )
                if r.status_code == 200:
                    return r.content
                if r.status_code in (401, 402, 403, 429):
                    self._openai_disabled = True
                    break
        except Exception as e:
            log.warning("OpenAI TTS exception: %s", e)
        return b""

    def _mac_say(self, text: str) -> bytes:
        try:
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
                aiff_path = tmp.name
            mp3_path = aiff_path.replace(".aiff", ".mp3")
            voice = os.environ.get("ACCESSIBILITY_MACOS_VOICE", "Samantha")
            rate = os.environ.get("ACCESSIBILITY_MACOS_RATE", "170")
            subprocess.run(
                ["say", "-v", voice, "-r", rate, "-o", aiff_path, text],
                check=True, timeout=20,
            )
            # Convert AIFF → MP3 via afconvert
            try:
                subprocess.run(
                    ["afconvert", "-f", "mp4f", "-d", "aac", aiff_path, mp3_path],
                    check=True, timeout=15,
                )
                with open(mp3_path, "rb") as f:
                    return f.read()
            except Exception:
                # If conversion fails, return AIFF anyway (browser plays it)
                with open(aiff_path, "rb") as f:
                    return f.read()
            finally:
                for p in (aiff_path, mp3_path):
                    try: os.unlink(p)
                    except Exception: pass
        except Exception as e:
            log.info("mac say failed: %s", e)
        return b""

    def _gtts(self, text: str) -> bytes:
        try:
            from gtts import gTTS  # type: ignore
            import io
            buf = io.BytesIO()
            gTTS(text=text, lang="en", slow=False).write_to_fp(buf)
            return buf.getvalue()
        except Exception as e:
            log.info("gTTS failed: %s", e)
        return b""
