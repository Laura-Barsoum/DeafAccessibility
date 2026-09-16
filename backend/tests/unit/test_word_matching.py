"""Unit tests for whole-word, case-insensitive matching of names, keywords and
safety words.

Fusion once escalated speech with a substring test, so "helpful" raised a
critical alert and "firefighter" counted as "fire", the case whole-word name
matching was written to avoid. Names, keywords and safety words now share one
matcher, events.contains_word.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules.events import contains_word  # noqa: E402
from modules.fusion import Fusion         # noqa: E402
from modules.people import PeopleRegistry  # noqa: E402


class ContainsWordTests(unittest.TestCase):

    def test_whole_words_match_ignoring_case(self):
        self.assertTrue(contains_word("help", "HELP! Somebody help me"))
        self.assertTrue(contains_word("fire", "There is a Fire in the kitchen."))

    def test_words_inside_longer_words_do_not_match(self):
        for word, text in (("help", "That was really helpful"), ("fire", "The firefighters arrived"),
                           ("stop", "Start the stopwatch"), ("laura", "Laurel is here")):
            self.assertFalse(contains_word(word, text), (word, text))


class FusionSafetyWordTests(unittest.TestCase):

    def fuse(self, text):
        out = Fusion().fuse({"text": text, "segments": []}, [], [], {"reliability": 0.7},
                            {"speaker_attribution": None, "faces": []}, None)
        return out["headlines"][0]["priority_name"]

    def test_helpful_speech_is_not_escalated(self):
        self.assertEqual(self.fuse("That was really helpful, thank you"), "IMPORTANT")

    def test_a_safety_word_still_escalates(self):
        self.assertEqual(self.fuse("Stop, the pan is on fire!"), "CRITICAL")


class NameAlertTests(unittest.TestCase):

    def setUp(self):
        self.reg = PeopleRegistry.__new__(PeopleRegistry)
        self.reg.profile = {"self_name": "Laura", "keywords": ["fire"], "people": {}}

    def test_name_matches_in_any_case(self):
        self.assertTrue(self.reg.detect_name_calls("LAURA, come here")["self_called"])

    def test_longer_words_raise_no_alert(self):
        out = self.reg.detect_name_calls("Laurel said the firefighters were helpful")
        self.assertFalse(out["self_called"])
        self.assertEqual(out["keywords_heard"], [])


if __name__ == "__main__":
    unittest.main()
