"""
llm.py — LLaMA 3 wrapper (via Groq) for situational reasoning.

Three roles:
    1. Compose user-facing notifications from a set of multimodal events
       (e.g. "[CRITICAL] Smoke alarm — kitchen, behind you")
    2. Long-form summarisation: condense an hour of events into one
       paragraph for users who couldn't follow live
    3. Disambiguate ambiguous YamNet labels using scene context
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

log = logging.getLogger("accessibility.llm")


# Models to try in order. Hosted model catalogues change without notice: the
# LLaMA 3 models this project originally used were withdrawn by the provider
# and began returning "model_not_found", which surfaced as an error string in
# the user interface. The chain below is tried in order so a single withdrawn
# model degrades to the next rather than breaking the feature.
_MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",    # measured fastest and most consistent on this task
    "openai/gpt-oss-20b",     # smaller, more variable latency
    "groq/compound-mini",
    "qwen/qwen3.8-27b",       # last: occasionally returns empty content
]


class LLM:
    def __init__(self) -> None:
        self._key = os.environ.get("GROQ_API_KEY", "")
        configured = os.environ.get("ACCESSIBILITY_LLM_MODEL", "").strip()
        # Preferred model first, then the fallbacks, without duplicates.
        self._models = ([configured] if configured else []) + [
            m for m in _MODEL_FALLBACKS if m != configured
        ]
        self._model = self._models[0]
        self._dead: set = set()      # models that returned model_not_found
        self._client = None
        log.info("LLM initialised (models=%s, key=%s)",
                 self._models, "set" if self._key else "missing")

    def _ensure_client(self) -> None:
        if self._client is not None or not self._key:
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=self._key)
        except Exception as e:
            log.warning("Groq unavailable (%s)", e)
            self._client = None

    def _chat(self, messages, max_tokens: int = 250, temperature: float = 0.3) -> str:
        """Call the LLM, trying each model in the fallback chain.

        Returns an empty string on total failure rather than an error
        message. Callers treat empty as "no polished text available" and
        fall back to their own plain-text rendering, so a provider outage
        degrades the wording rather than printing an API error into the
        user interface.
        """
        self._ensure_client()
        if self._client is None:
            return ""
        for model in self._models:
            if model in self._dead:
                continue
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    # gpt-oss models emit reasoning tokens that consume the
                    # completion budget, so allow headroom and ask for the
                    # shortest reasoning pass. Unsupported keys are ignored
                    # by providers that do not implement them.
                    max_tokens=max(max_tokens, 150),
                    temperature=temperature,
                    reasoning_effort="low",
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    if model != self._model:
                        log.info("LLM: fell back to %s", model)
                        self._model = model
                    return text
            except TypeError:
                # Client rejected reasoning_effort; retry without it.
                try:
                    resp = self._client.chat.completions.create(
                        model=model, messages=messages,
                        max_tokens=max(max_tokens, 150), temperature=temperature,
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    if text:
                        self._model = model
                        return text
                except Exception as e:
                    log.warning("LLM %s failed: %s", model, e)
            except Exception as e:
                msg = str(e)
                if "model_not_found" in msg or "does not exist" in msg:
                    log.warning("LLM model %s unavailable, trying next", model)
                    self._dead.add(model)
                    continue
                log.warning("LLM %s failed: %s", model, msg[:160])
        log.warning("LLM: all models failed; caller will use plain text")
        return ""

    # ------------------------------------------------------------------
    # Notification composer
    # ------------------------------------------------------------------

    def compose_notification(self, events: List[Dict[str, Any]],
                             scene: str = "") -> str:
        """
        Given a list of recent events + the current scene description,
        produce one short user-facing sentence appropriate for a Deaf user.
        """
        if not events:
            return ""
        sys = (
            "You are an accessibility assistant for a Deaf user. "
            "Given a list of detected sounds, speech, and scene context, "
            "compose ONE short sentence (max 18 words) describing what "
            "is happening. If a CRITICAL event is present, lead with a "
            "warning emoji and the event. If multiple events are routine, "
            "summarise. Never invent events. Write in plain English."
        )
        user = json.dumps({"events": events, "scene": scene}, indent=0)
        return self._chat(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            max_tokens=80, temperature=0.2,
        )

    # ------------------------------------------------------------------
    # Long-form summarisation (Summary mode)
    # ------------------------------------------------------------------

    def summarise(
        self, events: List[Dict[str, Any]], window_label: str = "the past hour"
    ) -> str:
        if not events:
            return f"Nothing notable happened in {window_label}."
        sys = (
            "You are an accessibility assistant. Summarise the list of "
            "events into ONE paragraph for a Deaf user who could not "
            "follow them live. Group conversations together. Note "
            "safety-critical events explicitly. Be concise (max 80 words)."
        )
        user = json.dumps({"window": window_label, "events": events}, indent=0)
        return self._chat(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            max_tokens=200, temperature=0.3,
        )

    # ------------------------------------------------------------------
    # Scene-aware label disambiguation
    # ------------------------------------------------------------------

    def disambiguate_label(
        self, top_labels: List[Dict[str, Any]], scene: str
    ) -> str:
        """
        Sometimes YamNet returns ambiguous neighbours like ('Bell',
        'Doorbell', 'Church bell'). Given the scene context, pick the most
        plausible one.
        """
        if not top_labels:
            return ""
        sys = (
            "Given a list of candidate audio labels with confidences and "
            "the visual scene context, return ONLY the most plausible "
            "label as a single word/phrase. No explanation."
        )
        user = json.dumps(
            {"candidates": top_labels[:5], "scene": scene}, indent=0
        )
        return self._chat(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            max_tokens=15, temperature=0.0,
        ).strip().strip('"').strip()
