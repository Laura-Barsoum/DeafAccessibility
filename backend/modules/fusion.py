"""
fusion.py — Multimodal fusion for accessibility events.

Inputs (per processing tick):
    - Speech: STT transcript (with confidence per segment)
    - Sound: AST top-k labels + confidences, and AST's speech probability
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

from .events import Event, Priority, is_speech_label, merge_events

log = logging.getLogger("accessibility.fusion")

# AST's speech probability below which a transcript is treated as Whisper
# hallucinating on a non-speech sound. None disables the check. 0.18 is the
# 95th percentile of AST's speech probability over the 2,000 ESC-50 clips, none
# of which is speech (scripts/report_experiments/fewshot_embeddings.py), so 95%
# of non-speech audio falls below it. Live captions (/stt/stream) are unaffected.
SPEECH_GATE: Optional[float] = 0.18
SAFETY_WORDS = ("help", "fire", "emergency", "stop")


class Fusion:
    """The orchestration core: merges every model's per-tick output into one
    ranked event stream.

    Detections of one event are collapsed (same label, related sound labels,
    a caption and the alerts raised from it), speech-only sound labels are left
    to the captions, and a transcript is ignored when AST hears no speech. The
    rest is sorted by priority then confidence and the top three actionable
    events become the tick's headlines.
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
        speech_probability: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Combine all multimodal signals into a single tick of events."""
        events: List[Event] = []
        text = ((stt_result or {}).get("text") or "").strip()
        heard = speech_probability is None or SPEECH_GATE is None or speech_probability >= SPEECH_GATE
        suppressed: Optional[str] = None

        # 1) Speech → an event, escalated for safety words, unless AST heard no
        #    speech in the same audio, in which case the words were invented.
        if text and heard:
            base_priority = Priority.IMPORTANT
            if any(t in text.lower() for t in SAFETY_WORDS):
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
        elif text:
            suppressed = text
            log.info("fusion: ignoring transcript %r, AST speech probability %.2f", text[:40], speech_probability)

        # 2) Personal custom-sound matches always win priority over generic labels.
        events.extend(personal_events)

        # 3) Sounds, hazards, signs and name or keyword alerts.
        for e in sound_events:
            if e.source == "speech" and not (text and heard):
                continue                    # alerts raised from words nobody spoke
            if e.source == "sound" and is_speech_label(e.label) and e.priority > Priority.INFORM:
                e.priority = Priority.INFORM  # the captions already show speech
            events.append(e)

        # 4) One event per real event.
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
            "speech_probability": speech_probability,
            "suppressed_transcript": suppressed,
            "n_events": len(events),
            "n_critical": sum(1 for e in events if e.priority == Priority.CRITICAL),
        }
