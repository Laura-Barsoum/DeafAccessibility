"""Unit tests for how fusion collapses detections of one event.

Regression tests for measured defects: on scripted scenes 38% of headline slots
repeated an event already shown ("Siren" beside "Police car (siren)", "Dog"
beside "Bark", a warning shown as its transcript and again as keyword alerts),
and Whisper's transcripts of non-speech sounds took further slots.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules import fusion as fusion_mod                                   # noqa: E402
from modules.events import Event, Priority, merge_events, same_sound_event  # noqa: E402
from modules.fusion import Fusion                                          # noqa: E402

LIP = {"reliability": 0.7}
FACE = {"speaker_attribution": None, "faces": []}


def snd(label, prio, conf=0.5, **extra):
    return Event(source="sound", label=label, priority=prio, confidence=conf, ts=100.0, extra=extra)


class SameEventTests(unittest.TestCase):
    """Which sound labels describe one event."""

    def test_labels_sharing_a_word_are_one_event(self):
        self.assertTrue(same_sound_event("Siren", "Police car (siren)"))
        self.assertTrue(same_sound_event("Alarm clock", "Alarm"))

    def test_audioset_parent_and_child_are_one_event(self):
        self.assertTrue(same_sound_event("Dog", "Bark"))
        self.assertTrue(same_sound_event("Glass", "Shatter"))

    def test_unrelated_sounds_stay_apart(self):
        self.assertFalse(same_sound_event("Doorbell", "Dog bark"))
        self.assertFalse(same_sound_event("Siren", "Baby cry, infant cry"))

    def test_merge_keeps_the_strongest_label_and_lists_the_rest(self):
        merged = merge_events([snd("Police car (siren)", Priority.CRITICAL, 0.6),
                               snd("Siren", Priority.CRITICAL, 0.8)])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].label, "Siren")
        self.assertEqual(merged[0].extra["merged"], ["Police car (siren)"])

    def test_personal_label_is_kept_with_the_group_priority(self):
        merged = merge_events([snd("Alarm", Priority.CRITICAL, 0.7),
                               snd("Kitchen alarm (personal)", Priority.IMPORTANT, 0.6, matched_via="personalizer")])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].label, "Kitchen alarm (personal)")
        self.assertEqual(merged[0].priority, Priority.CRITICAL)


class FusionHeadlineTests(unittest.TestCase):
    """Headline slots after fusion."""

    def setUp(self):
        self.fuser = Fusion()

    def fuse(self, text="", sounds=(), speech_probability=None):
        return self.fuser.fuse({"text": text, "segments": []}, list(sounds), [], LIP, FACE, None,
                               speech_probability=speech_probability)

    def test_caption_and_its_name_alert_take_one_slot(self):
        words = "Laura, can you come here for a moment?"
        alert = Event(source="speech", label="Your name was called", priority=Priority.CRITICAL,
                      confidence=0.95, text=words, extra={"trigger": "self_name"})
        out = self.fuse(words, [alert])
        self.assertEqual(len(out["headlines"]), 1)
        self.assertTrue(out["headlines"][0]["label"].startswith("Your name was called: Laura"))
        self.assertEqual(out["headlines"][0]["priority_name"], "CRITICAL")

    def test_speech_labels_never_take_a_headline(self):
        out = self.fuse("Hello there", [snd("Speech", Priority.IMPORTANT, 0.9),
                                        snd("Speech synthesizer", Priority.IMPORTANT, 0.5)])
        self.assertEqual([h["source"] for h in out["headlines"]], ["speech"])

    def test_transcript_is_ignored_when_ast_hears_no_speech(self):
        with patch.object(fusion_mod, "SPEECH_GATE", 0.3):
            out = self.fuse("What?", [snd("Dog", Priority.IMPORTANT, 0.6)], speech_probability=0.05)
        self.assertEqual([h["label"] for h in out["headlines"]], ["Dog"])
        self.assertEqual(out["suppressed_transcript"], "What?")

    def test_transcript_is_kept_when_speech_is_heard(self):
        with patch.object(fusion_mod, "SPEECH_GATE", 0.3):
            out = self.fuse("What?", [snd("Dog", Priority.IMPORTANT, 0.6)], speech_probability=0.8)
        self.assertIn("speech", [h["source"] for h in out["headlines"]])
        self.assertIsNone(out["suppressed_transcript"])

    def test_no_check_without_a_speech_probability(self):
        with patch.object(fusion_mod, "SPEECH_GATE", 0.3):
            out = self.fuse("Hello there", [], speech_probability=None)
        self.assertEqual(len(out["headlines"]), 1)


if __name__ == "__main__":
    unittest.main()
