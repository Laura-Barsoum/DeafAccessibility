"""Unit tests for the continuous-learning loop on Personalizer.

We bypass YamNet (TF Hub load is slow) by monkey-patching `_embed` to
return a deterministic random vector seeded by the audio length, so the
match → feedback flow is testable without GPU/TF.
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

_TMP_DIR = tempfile.mkdtemp(prefix="personalizer_test_")

# Redirect storage BEFORE importing
from modules import personalizer as pm  # noqa: E402
pm.PROFILE_DIR = _TMP_DIR
pm.PROFILE_PATH = os.path.join(_TMP_DIR, "profile.json")


def _stub_embed(self, audio_bytes):
    """Replacement for the real YamNet embed. Deterministic per byte length."""
    rng = np.random.default_rng(seed=len(audio_bytes) % 100)
    return rng.standard_normal(1024).astype(np.float32)


class PersonalizerFeedbackTests(unittest.TestCase):
    """The continuous-learning thumbs-up/down feedback loop."""
    def setUp(self):
        pm.Personalizer._embed = _stub_embed
        self.p = pm.Personalizer()
        # Pre-populate one enrolled sound; bypass YamNet by writing the
        # embedding directly.
        ref = _stub_embed(self.p, b"x" * 32000)
        self.p.profile["sounds"]["test_doorbell"] = {
            "label": "test_doorbell",
            "priority": 2,
            "embedding": ref.tolist(),
            "n_examples": 3,
        }
        self.p._save()

    def tearDown(self):
        # Wipe any data left by the test
        try:
            os.unlink(pm.PROFILE_PATH)
        except FileNotFoundError:
            pass

    # ── Match emits match_id ─────────────────────────────────────────
    def test_match_emits_match_id(self):
        events = self.p.match(b"x" * 32000, threshold=0.0)  # always match
        self.assertEqual(len(events), 1)
        ev = events[0].to_dict()
        self.assertIn("match_id", ev["extra"])
        self.assertTrue(ev["extra"].get("feedback_pending"))

    # ── Positive feedback grows n_examples ───────────────────────────
    def test_positive_feedback_increments_n_examples(self):
        events = self.p.match(b"x" * 32000, threshold=0.0)
        mid = events[0].to_dict()["extra"]["match_id"]

        before = self.p.profile["sounds"]["test_doorbell"]["n_examples"]
        r = self.p.feedback(mid, is_positive=True)
        after = self.p.profile["sounds"]["test_doorbell"]["n_examples"]

        self.assertTrue(r["ok"])
        self.assertEqual(after, before + 1)

    def test_positive_feedback_changes_embedding(self):
        # Use a DIFFERENT byte length than the enrolment so the stub embed
        # returns a different vector — otherwise the running mean stays
        # identical to the original.
        events = self.p.match(b"y" * 16384, threshold=0.0)
        mid = events[0].to_dict()["extra"]["match_id"]
        emb_before = np.array(self.p.profile["sounds"]["test_doorbell"]["embedding"])
        self.p.feedback(mid, is_positive=True)
        emb_after = np.array(self.p.profile["sounds"]["test_doorbell"]["embedding"])
        self.assertFalse(np.allclose(emb_before, emb_after))

    # ── Negative feedback bumps the per-sound threshold ──────────────
    def test_negative_feedback_bumps_threshold(self):
        events = self.p.match(b"x" * 32000, threshold=0.0)
        mid = events[0].to_dict()["extra"]["match_id"]
        r = self.p.feedback(mid, is_positive=False)
        self.assertTrue(r["ok"])
        self.assertIn("threshold", r)
        # Default base 0.90 → 0.92 after one negative-feedback bump.
        self.assertAlmostEqual(r["threshold"], 0.92, places=5)

    def test_negative_feedback_tightens_the_prototypical_gate(self):
        # Regression: rejection used to change only the cosine threshold, which
        # the default prototypical matcher never reads.
        events = self.p.match(b"x" * 32000, threshold=0.0)
        mid = events[0].to_dict()["extra"]["match_id"]
        before = self.p.profile["sounds"]["test_doorbell"].get("proto_gate")
        r = self.p.feedback(mid, is_positive=False)
        after = self.p.profile["sounds"]["test_doorbell"]["proto_gate"]
        self.assertIn("proto_gate", r)
        self.assertLess(after, before if before is not None else pm.PROTO_GATE_FLOOR + 1e-9)

    def test_repeated_rejection_never_makes_a_sound_unmatchable(self):
        self.p.profile["sounds"]["test_doorbell"]["proto_gate"] = 0.10
        for _ in range(60):
            events = self.p.match(b"x" * 32000, threshold=0.0)
            self.p.feedback(events[0].to_dict()["extra"]["match_id"], is_positive=False)
        self.assertGreaterEqual(self.p.profile["sounds"]["test_doorbell"]["proto_gate"], 0.02)

    def test_negative_feedback_caps_at_0_95(self):
        # Hammer it 20 times — must never exceed 0.95
        self.p.profile["sounds"]["test_doorbell"]["threshold"] = 0.90
        for _ in range(20):
            events = self.p.match(b"x" * 32000, threshold=0.0)
            mid = events[0].to_dict()["extra"]["match_id"]
            self.p.feedback(mid, is_positive=False)
        final = self.p.profile["sounds"]["test_doorbell"]["threshold"]
        self.assertLessEqual(final, 0.95)

    # ── Edge cases ──────────────────────────────────────────────────
    def test_unknown_match_id_returns_error(self):
        r = self.p.feedback("does_not_exist", is_positive=True)
        self.assertFalse(r["ok"])
        self.assertIn("not found", r.get("error", ""))

    def test_feedback_on_removed_sound_returns_error(self):
        events = self.p.match(b"x" * 32000, threshold=0.0)
        mid = events[0].to_dict()["extra"]["match_id"]
        self.p.remove("test_doorbell")
        r = self.p.feedback(mid, is_positive=True)
        self.assertFalse(r["ok"])

    def test_match_id_buffer_bounded(self):
        # maxlen=200; flooding shouldn't grow unbounded
        for i in range(300):
            self.p.match(b"x" * (1000 + i), threshold=0.0)
        self.assertLessEqual(len(self.p._recent_matches), 200)


def tearDownModule():
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
