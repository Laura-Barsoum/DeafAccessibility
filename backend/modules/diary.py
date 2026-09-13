"""
diary.py — SQLite-backed soundscape diary.

Persists events for longitudinal queries:
    - "What sounds happened in my home today?"
    - "How many times did the dog bark this week?"
    - "When did the smoke alarm go off?"

Stores label + priority + timestamp, never the audio itself. For speech the
label is the first 80 characters of the caption, so the diary does keep what
was said, in text.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from typing import Any, Dict, List

from .events import Event, Priority

log = logging.getLogger("accessibility.diary")


class Diary:
    def __init__(self, db_path: str = "./data/diary.db") -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.path = db_path
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._init_schema()
        log.info("Diary at %s", self.path)

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts REAL NOT NULL,
              source TEXT,
              label TEXT,
              priority INTEGER,
              confidence REAL,
              direction TEXT,
              speaker TEXT
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_events_priority ON events(priority)"
        )
        self._conn.commit()

    def log(self, event: Event) -> None:
        self._conn.execute(
            "INSERT INTO events (ts, source, label, priority, confidence, direction, speaker) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                event.ts, event.source, event.label, int(event.priority),
                event.confidence, event.direction, event.speaker,
            ),
        )
        self._conn.commit()

    def log_many(self, events: List[Event]) -> None:
        self._conn.executemany(
            "INSERT INTO events (ts, source, label, priority, confidence, direction, speaker) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (e.ts, e.source, e.label, int(e.priority),
                 e.confidence, e.direction, e.speaker)
                for e in events
            ],
        )
        self._conn.commit()

    def query(
        self, since_s: float = 86400.0, min_priority: int = 0
    ) -> List[Dict[str, Any]]:
        """Return events newer than `since_s` seconds with priority >= min."""
        cutoff = time.time() - since_s
        cur = self._conn.execute(
            "SELECT ts, source, label, priority, confidence, direction, speaker "
            "FROM events WHERE ts >= ? AND priority >= ? ORDER BY ts DESC",
            (cutoff, min_priority),
        )
        rows = cur.fetchall()
        names = {0: "AMBIENT", 1: "INFORM", 2: "IMPORTANT", 3: "CRITICAL"}
        return [
            {"ts": r[0], "source": r[1], "label": r[2], "priority": r[3],
             "priority_name": names.get(r[3], "AMBIENT"),
             "confidence": r[4], "direction": r[5], "speaker": r[6]}
            for r in rows
        ]

    def label_histogram(
        self, since_s: float = 86400.0, min_priority: int = 0
    ) -> Dict[str, int]:
        cutoff = time.time() - since_s
        cur = self._conn.execute(
            "SELECT label, COUNT(*) FROM events WHERE ts >= ? AND priority >= ? "
            "GROUP BY label ORDER BY 2 DESC",
            (cutoff, min_priority),
        )
        return {r[0]: r[1] for r in cur.fetchall()}

    def close(self) -> None:
        self._conn.close()
