"""
summarizer.py — Long-form summary mode.

Buffers all events from a configurable window (default 1 hour), groups
them by type, and asks the LLM to produce one paragraph summarising what
happened. Designed for Deaf users who couldn't follow everything live and
want a recap.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, List

from .events import Event, Priority

log = logging.getLogger("accessibility.summarizer")


class Summarizer:
    def __init__(self, window_s: float = 3600.0, max_events: int = 5000) -> None:
        self.window_s = window_s
        self._buffer: deque = deque(maxlen=max_events)

    def add(self, event: Event) -> None:
        self._buffer.append(event)

    def add_many(self, events: List[Event]) -> None:
        for e in events:
            self._buffer.append(e)

    def _recent(self) -> List[Event]:
        cutoff = time.time() - self.window_s
        return [e for e in self._buffer if e.ts >= cutoff]

    def stats(self) -> Dict[str, Any]:
        events = self._recent()
        priority_counts: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}
        critical_events: List[Dict[str, Any]] = []
        for e in events:
            priority_counts[e.priority.name] = priority_counts.get(e.priority.name, 0) + 1
            source_counts[e.source] = source_counts.get(e.source, 0) + 1
            if e.priority >= Priority.IMPORTANT:
                critical_events.append(e.to_dict())
        return {
            "window_s": self.window_s,
            "total_events": len(events),
            "priority_counts": priority_counts,
            "source_counts": source_counts,
            "critical_events": critical_events[:20],
        }

    def summarise(self, llm) -> str:
        events = self._recent()
        if not events:
            return "Nothing notable happened in the past hour."
        # Compress to a manageable list for the LLM
        compressed: List[Dict[str, Any]] = []
        for e in events:
            compressed.append({
                "source": e.source,
                "label": e.label[:80],
                "priority": e.priority.name,
                "ts": round(e.ts, 1),
            })
        # Limit to 200 events to keep tokens reasonable.
        if len(compressed) > 200:
            compressed = compressed[-200:]
        return llm.summarise(compressed, "the past hour")
