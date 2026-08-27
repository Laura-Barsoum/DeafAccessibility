"""Unit tests for the prototypical-network personal-sound matcher.

We stub the YamNet embedding so the tests are deterministic and fast:
audio bytes starting with b'A' embed near prototype A, b'B' near B.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules import personalizer as pm  # noqa: E402
from modules.events import Priority       # noqa: E402

_A = np.random.default_rng(1).standard_normal(1024).astype(np.float32)
_B = np.random.default_rng(2).standard_normal(1024).astype(np.float32)


def _stub_embed(self, audio_bytes):
    base = _A if audio_bytes[:1] == b"A" else _B
    noise = np.random.default_rng(len(audio_bytes)).standard_normal(1024).astype(np.float32) * 0.05
    return base + noise


class PrototypicalTests(unittest.TestCase):
    """The prototypical-network personal-sound matcher (Snell et al., 2017): enrolment, same-sound firing and open-set rejection."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="proto_test_")
        pm.PROFILE_DIR = self.tmp
        pm.PROFILE_PATH = os.path.join(self.tmp, "profile.json")
        pm.Personalizer._embed = _stub_embed
        self.p = pm.Personalizer()
        self.p.profile = {"sounds": {}}
        # Enrol "doorbell" with three A-clips.
        self.p.enrol("doorbell", Priority.IMPORTANT, [b"A" * 100, b"A" * 101, b"A" * 102])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_enrol_stores_prototype_and_radius(self):
        s = self.p.profile["sounds"]["doorbell"]
        self.assertIn("prototype", s)
        self.assertIn("radius", s)
        self.assertEqual(len(s["prototype"]), 1024)

    def test_same_sound_fires(self):
        ev = self.p.match_prototypical(b"A" * 200)
        self.assertEqual(len(ev), 1)
        self.assertIn("doorbell", ev[0].label)
        self.assertEqual(ev[0].to_dict()["extra"]["method"], "prototypical")

    def test_different_sound_is_rejected(self):
        # Background audio far from the prototype must be open-set rejected,
        # not snapped to the only enrolled class.
        ev = self.p.match_prototypical(b"B" * 200)
        self.assertEqual(len(ev), 0)

    def test_match_current_dispatches_to_prototypical(self):
        os.environ.pop("ACCESSIBILITY_PERSONALIZER_METHOD", None)  # default
        ev = self.p.match_current(b"A" * 200)
        self.assertEqual(ev[0].to_dict()["extra"]["method"], "prototypical")

    def test_match_current_cosine_when_configured(self):
        os.environ["ACCESSIBILITY_PERSONALIZER_METHOD"] = "cosine"
        try:
            ev = self.p.match_current(b"A" * 200)
            self.assertTrue(ev and ev[0].to_dict()["extra"]["method"] == "cosine")
        finally:
            os.environ.pop("ACCESSIBILITY_PERSONALIZER_METHOD", None)

    def test_two_sounds_discriminated(self):
        self.p.enrol("phone", Priority.IMPORTANT, [b"B" * 100, b"B" * 101, b"B" * 102])
        ev_a = self.p.match_prototypical(b"A" * 200)
        ev_b = self.p.match_prototypical(b"B" * 200)
        self.assertIn("doorbell", ev_a[0].label)
        self.assertIn("phone", ev_b[0].label)


if __name__ == "__main__":
    unittest.main()
