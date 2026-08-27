"""Unit tests for the events / priority logic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from modules.events import (
    Event, Priority, PRIORITY_BANDS, label_priority, merge_events,
)


class PriorityClassificationTests(unittest.TestCase):
    """Priority band assignment for AudioSet labels (safety-critical mapping)."""
    def test_smoke_alarm_critical(self):
        self.assertEqual(label_priority("Smoke detector"), Priority.CRITICAL)
        self.assertEqual(label_priority("Smoke alarm, smoke detector"), Priority.CRITICAL)
        self.assertEqual(label_priority("Civil defense siren"), Priority.CRITICAL)

    def test_baby_cry_critical(self):
        self.assertEqual(label_priority("Baby cry, infant cry"), Priority.CRITICAL)

    def test_doorbell_important(self):
        self.assertEqual(label_priority("Doorbell"), Priority.IMPORTANT)
        self.assertEqual(label_priority("Telephone bell ringing"), Priority.IMPORTANT)

    def test_speech_important(self):
        self.assertEqual(label_priority("Speech"), Priority.IMPORTANT)

    def test_music_inform_or_ambient(self):
        # Music is INFORM-tier in our taxonomy
        self.assertEqual(label_priority("Music"), Priority.INFORM)

    def test_unknown_is_ambient(self):
        self.assertEqual(label_priority("xyz unknown sound"), Priority.AMBIENT)
        self.assertEqual(label_priority(""), Priority.AMBIENT)

    def test_priority_bands_complete(self):
        for p in Priority:
            self.assertIn(p, PRIORITY_BANDS)


class EventMergingTests(unittest.TestCase):
    """Deduplication of near-identical events within a time window."""
    def _ev(self, label, ts, conf, prio=Priority.INFORM):
        return Event(source="sound", label=label, priority=prio,
                     confidence=conf, ts=ts)

    def test_merges_duplicates_in_window(self):
        events = [
            self._ev("Doorbell", 100.0, 0.6),
            self._ev("Doorbell", 100.5, 0.85),  # higher conf → wins
            self._ev("Doorbell", 100.9, 0.4),
        ]
        merged = merge_events(events, window_s=1.5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].confidence, 0.85)

    def test_keeps_separate_outside_window(self):
        events = [
            self._ev("Doorbell", 100.0, 0.6),
            self._ev("Doorbell", 105.0, 0.7),  # 5s later — separate event
        ]
        merged = merge_events(events, window_s=1.5)
        self.assertEqual(len(merged), 2)

    def test_keeps_different_labels(self):
        events = [
            self._ev("Doorbell", 100.0, 0.6),
            self._ev("Dog bark", 100.2, 0.5),
        ]
        merged = merge_events(events, window_s=1.5)
        self.assertEqual(len(merged), 2)


class EventActionableTests(unittest.TestCase):
    """The is_actionable() headline filter."""
    def test_critical_actionable(self):
        e = Event(source="sound", label="Smoke alarm",
                  priority=Priority.CRITICAL, confidence=0.9)
        self.assertTrue(e.is_actionable())

    def test_ambient_not_actionable(self):
        e = Event(source="sound", label="background", priority=Priority.AMBIENT,
                  confidence=0.9)
        self.assertFalse(e.is_actionable())


if __name__ == "__main__":
    unittest.main()
