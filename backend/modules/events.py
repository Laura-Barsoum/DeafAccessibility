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

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set


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


# ----------------------------------------------------------------------
# Which detections describe the same event
# ----------------------------------------------------------------------

_ONTOLOGY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "audioset_ontology.json")
_ANCESTORS: Optional[Dict[str, Set[str]]] = None
_STOP_WORDS = {"and", "the", "sound", "sounds", "noise", "personal"}
SPEECH_LABEL_WORDS = ("speech", "conversation", "narration", "babbling", "whispering")


def _ancestors() -> Dict[str, Set[str]]:
    """Lower-case AudioSet label name -> the names of itself and every ancestor,
    from Google's AudioSet ontology. Empty if the ontology file is missing."""
    global _ANCESTORS
    if _ANCESTORS is not None:
        return _ANCESTORS
    try:
        with open(_ONTOLOGY_PATH) as f:
            nodes = json.load(f)
    except Exception:
        _ANCESTORS = {}
        return _ANCESTORS
    by_id = {n["id"]: n for n in nodes}
    parents: Dict[str, Set[str]] = {}
    for n in nodes:
        for child in n.get("child_ids", []):
            parents.setdefault(child, set()).add(n["id"])
    memo: Dict[str, Set[str]] = {}

    def up(node_id: str) -> Set[str]:
        if node_id not in memo:
            memo[node_id] = {node_id}
            for p in parents.get(node_id, ()):
                memo[node_id] = memo[node_id] | up(p)
        return memo[node_id]

    _ANCESTORS = {n["name"].lower(): {by_id[a]["name"].lower() for a in up(n["id"])} for n in nodes}
    return _ANCESTORS


def _content_words(label: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z]+", label.lower()) if len(w) > 2 and w not in _STOP_WORDS}


def is_speech_label(label: str) -> bool:
    """True for AudioSet labels that only say someone is speaking."""
    return any(w in (label or "").lower() for w in SPEECH_LABEL_WORDS)


def same_sound_event(a: str, b: str) -> bool:
    """True when two sound labels most likely describe one event: one is a
    kind of the other in the AudioSet ontology ("Dog" and "Bark"), or they
    share a content word ("Siren" and "Police car (siren)")."""
    a0, b0 = (a or "").lower().strip(), (b or "").lower().strip()
    if a0 == b0:
        return True
    anc = _ancestors()
    if a0 in anc and b0 in anc and (a0 in anc[b0] or b0 in anc[a0]):
        return True
    return bool(_content_words(a0) & _content_words(b0))


def _same_event(a: Event, b: Event) -> bool:
    if a.label == b.label:
        return True
    if a.source == b.source == "sound":
        return same_sound_event(a.label, b.label)
    if a.source == b.source == "speech":
        ta, tb = (a.text or "")[:120].strip(), (b.text or "")[:120].strip()
        return bool(ta) and ta == tb   # a caption and the alerts raised from its words
    return False


def _combine(group: List[Event]) -> Event:
    if len(group) == 1:
        return group[0]
    personal = [e for e in group if (e.extra or {}).get("matched_via") == "personalizer"]
    kept = personal[0] if personal else max(group, key=lambda e: (int(e.priority), e.confidence))
    kept.priority = max(e.priority for e in group)
    kept.extra = dict(kept.extra or {}, merged=[e.label for e in group if e is not kept])
    alerts = [e for e in group if e.source == "speech" and (e.extra or {}).get("trigger") in ("self_name", "keyword")]
    words = next((e.text for e in group if e.source == "speech" and e.text and not (e.extra or {}).get("trigger")), None)
    if alerts and words:
        kept.label = f"{alerts[0].label}: {words}"[:80]
        kept.text = words
    return kept


def merge_events(events: List[Event], window_s: float = 1.5) -> List[Event]:
    """
    Collapse detections of one event inside a short window into one event.

    Detections are one event when they carry the same label; when both are
    sounds whose labels AudioSet relates or that share a content word; or when
    both come from one transcript (the caption and the name or keyword alerts
    raised from it). The kept event is the user's own personal-sound match if
    one is present, otherwise the highest priority then confidence. It takes
    the group's highest priority and lists the other labels in extra["merged"];
    a caption merged with its alerts is labelled with the first alert followed
    by the words, for example "Your name was called: Laura, come here".
    """
    if not events:
        return []
    groups: List[List[Event]] = []
    for e in sorted(events, key=lambda e: e.ts):
        for g in groups:
            if e.ts - g[0].ts < window_s and any(_same_event(e, m) for m in g):
                g.append(e)
                break
        else:
            groups.append([e])
    return [_combine(g) for g in groups]
