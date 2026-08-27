"""
events.py — Event types, priority classification, and event records.

Every signal that flows through the system is normalised into an `Event`
with:
    - source modality (speech / sound / scene / lip / face / fusion)
    - priority band (CRITICAL / IMPORTANT / AMBIENT)
    - confidence
    - direction (if known)
    - timestamp

The priority logic is rule-based for the prototype; in a full system this
would be a learned classifier conditioned on user context (asleep vs.
awake, location, etc.).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional


class Priority(IntEnum):
    """Four-band severity ranking shared by every detector.

    Being an IntEnum lets the fusion engine sort events numerically
    (CRITICAL first) and lets the diary filter by a minimum priority.
    """
    AMBIENT = 0       # background TV, traffic — show in "ambient" stream only
    INFORM = 1        # someone enters room, doorbell, conversation
    IMPORTANT = 2     # name called, phone ringing, urgent speech
    CRITICAL = 3      # smoke alarm, glass break, scream, baby cry urgent


PRIORITY_BANDS = {
    Priority.AMBIENT:   "🟢 ambient",
    Priority.INFORM:    "🟡 inform",
    Priority.IMPORTANT: "🟠 important",
    Priority.CRITICAL:  "🔴 CRITICAL",
}


# ----------------------------------------------------------------------
# Static priority rules: AudioSet labels → priority band.
# This is the safety-critical core. We bias toward over-alerting
# (better to over-notify than miss a smoke alarm).
# ----------------------------------------------------------------------

CRITICAL_LABELS = {
    "smoke detector", "smoke alarm", "fire alarm", "carbon monoxide alarm",
    "alarm", "siren", "civil defense siren",
    "screaming", "scream", "shriek",
    "glass", "shatter", "shatters",
    "explosion", "boom", "gunshot, gunfire",
    "baby cry, infant cry", "baby crying",
    "crash",
}

IMPORTANT_LABELS = {
    "doorbell", "ding-dong", "knock",
    "telephone bell ringing", "ringtone", "telephone",
    "shout", "yell", "bellow",
    "speech",  # someone is speaking (conversational)
    "narration, monologue",
    "child speech, kid speaking",
    "whoop",
    "applause",
    "dog", "bark", "yip",  # likely warns of someone outside
}

INFORM_LABELS = {
    "footsteps",
    "door",
    "drawer open or close",
    "microwave oven",
    "alarm clock",
    "buzzer",
    "beep, bleep",
    "vehicle", "car", "engine",
    "music",
}


def label_priority(label: str) -> Priority:
    """Map a YamNet/AudioSet label string to a Priority band."""
    if not label:
        return Priority.AMBIENT
    low = label.lower()
    for kw in CRITICAL_LABELS:
        if kw in low:
            return Priority.CRITICAL
    for kw in IMPORTANT_LABELS:
        if kw in low:
            return Priority.IMPORTANT
    for kw in INFORM_LABELS:
        if kw in low:
            return Priority.INFORM
    return Priority.AMBIENT


@dataclass
class Event:
    """A single multimodal observation."""
    source: str                          # "speech" | "sound" | "scene" | "lip" | "face" | "fusion"
    label: str                           # e.g. "smoke alarm" | "Hello, Mary"
    priority: Priority
    confidence: float                    # [0, 1]
    direction: Optional[str] = None      # "left" | "right" | "behind" | None
    speaker: Optional[str] = None        # speaker id if attributable
    text: Optional[str] = None           # full transcript if speech
    ts: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to JSON for the frontend, adding the human-readable
        priority name and coloured band the UI renders."""
        d = asdict(self)
        d["priority_name"] = self.priority.name
        d["priority_band"] = PRIORITY_BANDS[self.priority]
        return d

    def is_actionable(self) -> bool:
        """True for events worth surfacing as a headline (IMPORTANT or
        CRITICAL), used by the fusion engine to filter ambient noise."""
        return self.priority >= Priority.IMPORTANT


def merge_events(events: List[Event], window_s: float = 1.5) -> List[Event]:
    """
    Collapse near-duplicate events within a small time window.
    E.g. two simultaneous "doorbell" detections from speech + sound modules.
    Keeps the highest-confidence representative.
    """
    if not events:
        return []
    events_sorted = sorted(events, key=lambda e: e.ts)
    merged: List[Event] = []
    for e in events_sorted:
        if merged and (e.ts - merged[-1].ts) < window_s and e.label == merged[-1].label:
            if e.confidence > merged[-1].confidence:
                merged[-1] = e
            continue
        merged.append(e)
    return merged
