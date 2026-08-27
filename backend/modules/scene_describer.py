"""
scene_describer.py — BLIP-based visual scene description.

For every webcam frame (or a periodic sample), BLIP (Li et al. 2022, ICML)
generates a one-sentence caption of what's visible. Examples:
    - "a person standing in a kitchen"
    - "two people sitting at a table with a dog"
    - "a child reaching for a cookie jar"

This is then fused with audio events by the LLM:
    Sound: "Doorbell" + Scene: "two people in living room"
    → notification: "Doorbell rang while you and your guest were chatting."

Caches recent captions to avoid redundant inference (BLIP is heavy).
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import List, Optional, Tuple

log = logging.getLogger("accessibility.scene")


class SceneDescriber:
    def __init__(self, sample_every_s: float = 2.0) -> None:
        self._blip = None
        self._processor = None
        self.sample_every_s = sample_every_s
        self._last_caption: Optional[str] = None
        self._last_caption_ts: float = 0.0
        log.info("SceneDescriber initialised (lazy load, period=%.1fs)",
                 sample_every_s)

    def _ensure_loaded(self) -> None:
        if self._blip is not None:
            return
        mid = os.environ.get(
            "ACCESSIBILITY_BLIP_MODEL", "Salesforce/blip-image-captioning-base"
        )
        # Retry a few times: under concurrent startup the transformers lazy
        # module can transiently raise "cannot import name 'BlipProcessor'".
        # A short retry lets the other thread finish initialising it.
        last_err = None
        for attempt in range(4):
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self._processor = BlipProcessor.from_pretrained(mid)
                self._blip = BlipForConditionalGeneration.from_pretrained(mid)
                self._blip.eval()
                log.info("BLIP loaded: %s", mid)
                return
            except ImportError as e:
                last_err = e
                time.sleep(0.75)   # transient concurrent-import race — retry
            except Exception as e:
                last_err = e
                break
        log.warning("BLIP unavailable (%s) — scene description disabled", last_err)
        self._blip = "placeholder"

    def describe_jpeg(self, jpeg_b64: str) -> Optional[str]:
        """Caption a single JPEG-encoded base64 frame."""
        # Throttle to avoid overloading on every frame
        now = time.time()
        if now - self._last_caption_ts < self.sample_every_s and self._last_caption:
            return self._last_caption

        self._ensure_loaded()
        if self._blip == "placeholder":
            return None

        try:
            from PIL import Image
            import torch

            img_bytes = base64.b64decode(jpeg_b64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            inputs = self._processor(image, return_tensors="pt")
            with torch.no_grad():
                out = self._blip.generate(**inputs, max_new_tokens=40)
            caption = self._processor.decode(out[0], skip_special_tokens=True).strip()
            self._last_caption = caption
            self._last_caption_ts = now
            log.info("scene: %s", caption)
            return caption
        except Exception as e:
            log.warning("BLIP describe failed: %s", e)
            return None

    def describe_batch(self, frames_b64: List[str]) -> List[str]:
        """Caption a list of frames; returns one caption per kept frame."""
        captions = []
        for f in frames_b64:
            cap = self.describe_jpeg(f)
            if cap:
                captions.append(cap)
        return captions
