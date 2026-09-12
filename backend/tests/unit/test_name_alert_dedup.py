"""Regression test: calling the user's name must not also report them as a
"mentioned" enrolled person when their own name is enrolled with a surname."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules import people as pm  # noqa: E402


class NameAlertDedupTests(unittest.TestCase):
    """Own-name calls produce one critical alert; other people are still mentioned."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="people_dedup_")
        self._saved = (pm.PROFILE_DIR, pm.PROFILE_PATH)
        pm.PROFILE_DIR = self.tmp
        pm.PROFILE_PATH = os.path.join(self.tmp, "profile.json")
        self.reg = pm.PeopleRegistry()
        self.reg.profile = {"people": {"Laura Barsoum": {}, "Sarah Jones": {}}, "self_name": "Laura", "keywords": []}

    def tearDown(self):
        pm.PROFILE_DIR, pm.PROFILE_PATH = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_own_name_is_not_also_a_mention(self):
        out = self.reg.detect_name_calls("Laura, dinner is ready in the kitchen.")
        self.assertTrue(out["self_called"])
        self.assertEqual(out["mentioned"], [])
        self.assertEqual(sum(1 for e in out["events"] if "mentioned" in e.label), 0)

    def test_other_enrolled_people_are_still_mentioned(self):
        out = self.reg.detect_name_calls("Sarah is at the door.")
        self.assertFalse(out["self_called"])
        self.assertEqual(out["mentioned"], ["Sarah Jones"])


if __name__ == "__main__":
    unittest.main()
