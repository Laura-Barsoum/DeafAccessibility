"""Unit tests for personalizer + summarizer + diary (no heavy ML dependencies)."""
import sys, os, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from modules import diary as diary_mod
from modules import summarizer as summer_mod
from modules.events import Event, Priority


class DiaryTests(unittest.TestCase):
    """The anonymised soundscape diary: logging and querying."""
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.diary = diary_mod.Diary(self.tmp.name)

    def tearDown(self):
        self.diary.close()
        os.unlink(self.tmp.name)

    def test_log_and_query(self):
        e = Event(source="sound", label="Doorbell",
                  priority=Priority.IMPORTANT, confidence=0.9, ts=time.time())
        self.diary.log(e)
        rows = self.diary.query(since_s=86400, min_priority=0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "Doorbell")

    def test_log_many(self):
        evs = [
            Event(source="sound", label="Bark", priority=Priority.INFORM,
                  confidence=0.5, ts=time.time()),
            Event(source="sound", label="Smoke alarm", priority=Priority.CRITICAL,
                  confidence=0.95, ts=time.time()),
        ]
        self.diary.log_many(evs)
        rows = self.diary.query(since_s=86400, min_priority=int(Priority.IMPORTANT))
        # Only the CRITICAL one passes the IMPORTANT filter
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "Smoke alarm")

    def test_label_histogram(self):
        for label in ["Doorbell", "Doorbell", "Bark"]:
            self.diary.log(Event(source="sound", label=label,
                                 priority=Priority.INFORM, confidence=0.5))
        hist = self.diary.label_histogram(since_s=86400)
        self.assertEqual(hist["Doorbell"], 2)
        self.assertEqual(hist["Bark"], 1)


class SummarizerTests(unittest.TestCase):
    """The rolling event summariser buffer."""
    def test_stats_structure(self):
        s = summer_mod.Summarizer(window_s=3600)
        s.add(Event(source="sound", label="Doorbell", priority=Priority.IMPORTANT, confidence=0.9))
        s.add(Event(source="speech", label="Hi", priority=Priority.IMPORTANT, confidence=0.8))
        st = s.stats()
        self.assertEqual(st["total_events"], 2)
        self.assertIn("source_counts", st)
        self.assertEqual(st["source_counts"]["sound"], 1)
        self.assertEqual(st["source_counts"]["speech"], 1)


if __name__ == "__main__":
    unittest.main()
