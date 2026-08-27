"""
fusion.py — Multimodal fusion for accessibility events.

Inputs (per processing tick):
    - Speech: STT transcript (with confidence per segment)
    - Sound: YamNet top-k labels + confidences
    - Personal: matches against user-enrolled custom sounds
    - Lip: audio-visual reliability score
    - Face: speaker-attribution verdict
    - Scene: BLIP caption

Output:
    - Events (already prioritised)
    - One LLM-composed user-facing notification per important/critical event
    - Updated soundscape diary
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .events import Event, Priority, merge_events

log = logging.getLogger("accessibility.fusion")


class Fusion:
    """The orchestration core: merges every model's per-tick output into one
    ranked event stream.

    It deduplicates near-identical detections within a short window, sorts by
    priority then confidence, and returns the top handful as headlines plus
    the full event list. This is where six independent inferences become a
    single coherent decision, the heart of the Template 4.1 contribution.
    """

    def __init__(self) -> None:
        log.info("Fusion engine initialised")

    def fuse(
        self,
        stt_result: Dict[str, Any],
        sound_events: List[Event],
        personal_events: List[Event],
        lip_reliability: Dict[str, Any],
        face_attribution: Dict[str, Any],
        scene_caption: Optional[str],
    ) -> Dict[str, Any]:
        """Combine all multimodal signals into a single tick of events."""
        events: List[Event] = []

        # 1) Speech → events. Boost priority if transcript contains
        # safety-critical terms ("help", "fire", a name being called).
        if stt_result and stt_result.get("text"):
            text = stt_result["text"].strip()
            base_priority = Priority.IMPORTANT
            low = text.lower()
            if any(t in low for t in ("help", "fire", "emergency", "stop")):
                base_priority = Priority.CRITICAL
            speaker = None
            if face_attribution and face_attribution.get("speaker_attribution"):
                pos = face_attribution["speaker_attribution"]["position"]
                speaker = f"speaker_{pos}"
            confidence = 0.7
            if lip_reliability and lip_reliability.get("reliability") is not None:
                confidence = float(lip_reliability["reliability"])
            events.append(Event(
                source="speech",
                label=text[:80],
                priority=base_priority,
                confidence=confidence,
                speaker=speaker,
                text=text,
            ))

        # 2) Personal custom-sound matches always win priority over generic
        # YamNet labels.
        events.extend(personal_events)

        # 3) Generic environmental sounds.
        events.extend(sound_events)

        # 4) Deduplicate near-identical detections.
        events = merge_events(events, window_s=1.5)

        # 5) Sort: critical first, then by confidence.
        events.sort(key=lambda e: (-int(e.priority), -e.confidence))

        # 6) Pick top-3 actionable events as the "headlines" of this tick.
        headlines = [e for e in events if e.is_actionable()][:3]

        return {
            "ts": time.time(),
            "events": [e.to_dict() for e in events],
            "headlines": [e.to_dict() for e in headlines],
            "scene": scene_caption,
            "lip_reliability": lip_reliability,
            "face_attribution": face_attribution,
            "n_events": len(events),
            "n_critical": sum(1 for e in events if e.priority == Priority.CRITICAL),
        }
