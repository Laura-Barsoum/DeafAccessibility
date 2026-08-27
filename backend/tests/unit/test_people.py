"""Unit tests for the People registry — name-call detection, enrolment,
self-name handling, false-positive resistance."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Redirect the people module's storage to a temp dir BEFORE importing.
_TMP_DIR = tempfile.mkdtemp(prefix="people_test_")
os.environ["_TEST_PEOPLE_DIR"] = _TMP_DIR
# We patch by monkey-patching the module constants after import.
from modules import people as people_mod  # noqa: E402
people_mod.PROFILE_DIR = _TMP_DIR
people_mod.PROFILE_PATH = os.path.join(_TMP_DIR, "profile.json")


class NameCallTests(unittest.TestCase):
    """The self-name flash — most safety-critical path in this module."""

    def setUp(self):
        # Fresh registry per test
        people_mod._registry_singleton = None
        self.reg = people_mod.get_people()
        # Wipe any persisted state from a previous run
        self.reg.profile = {"people": {}, "self_name": None}
        self.reg._save()

    def test_self_name_triggers_critical(self):
        self.reg.set_self_name("Laura")
        out = self.reg.detect_name_calls("Hey Laura, can you come here?")
        self.assertTrue(out["self_called"])
        self.assertEqual(len(out["events"]), 1)
        ev = out["events"][0]
        self.assertEqual(ev.label, "Your name was called")
        self.assertEqual(ev.priority.name, "CRITICAL")

    def test_no_false_positive_substring_match(self):
        """'Laura' must NOT match inside 'Laurel' or 'auraltest'."""
        self.reg.set_self_name("Laura")
        for sentence in [
            "Hey Laurel, can you come here?",
            "The aurora was beautiful last night",
            "Auralization is a research topic",
        ]:
            out = self.reg.detect_name_calls(sentence)
            self.assertFalse(out["self_called"],
                             msg=f"False positive on: {sentence!r}")

    def test_case_insensitive_match(self):
        self.reg.set_self_name("Laura")
        for sentence in ["LAURA come here", "laura, look", "Laura?"]:
            out = self.reg.detect_name_calls(sentence)
            self.assertTrue(out["self_called"], msg=f"Case fail on: {sentence!r}")

    def test_empty_self_name_safe(self):
        # Never set self_name → empty transcript should be safe
        out = self.reg.detect_name_calls("Anything could happen here")
        self.assertFalse(out["self_called"])
        self.assertEqual(out["events"], [])

    def test_other_person_mention_is_inform_not_critical(self):
        self.reg.set_self_name("Laura")
        # Enrol someone by writing directly to the profile (skip face/voice
        # which need real models).
        self.reg.profile["people"]["Mike"] = {
            "name": "Mike", "n_face_examples": 0, "n_voice_examples": 0,
        }
        out = self.reg.detect_name_calls("Mike said hello to me")
        self.assertFalse(out["self_called"])
        self.assertIn("Mike", out["mentioned"])
        self.assertEqual(out["events"][0].priority.name, "INFORM")

    def test_self_name_takes_precedence_over_other_mention(self):
        """If both 'Laura' and 'Mike' appear, CRITICAL must fire."""
        self.reg.set_self_name("Laura")
        self.reg.profile["people"]["Mike"] = {
            "name": "Mike", "n_face_examples": 0, "n_voice_examples": 0,
        }
        out = self.reg.detect_name_calls("Mike asked Laura a question")
        self.assertTrue(out["self_called"])
        # Both events present, CRITICAL first
        priorities = [e.priority.name for e in out["events"]]
        self.assertIn("CRITICAL", priorities)
        self.assertIn("INFORM", priorities)


class RegistryPersistenceTests(unittest.TestCase):
    """Ensure the JSON store survives a restart and reserved names are blocked."""

    def setUp(self):
        people_mod._registry_singleton = None
        self.reg = people_mod.get_people()
        self.reg.profile = {"people": {}, "self_name": None}
        self.reg._save()

    def test_set_self_name_persists(self):
        self.reg.set_self_name("Laura")
        # Force a fresh load
        people_mod._registry_singleton = None
        reg2 = people_mod.get_people()
        self.assertEqual(reg2.get_self_name(), "Laura")

    def test_remove_unknown_returns_false(self):
        self.assertFalse(self.reg.remove("nobody"))

    def test_reserved_name_rejected(self):
        # Names starting with "_" are reserved for internal keys
        res = self.reg.enrol("_self", frames_b64=[], audio_clips_bytes=[])
        self.assertFalse(res["ok"])

    def test_empty_name_rejected(self):
        res = self.reg.enrol("   ", frames_b64=[], audio_clips_bytes=[])
        self.assertFalse(res["ok"])

    def test_no_samples_rejected(self):
        # We deliberately give no usable input → enrolment must fail cleanly
        res = self.reg.enrol("Bob", frames_b64=[], audio_clips_bytes=[])
        self.assertFalse(res["ok"])
        self.assertIn("no usable", res.get("error", ""))


def tearDownModule():
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
