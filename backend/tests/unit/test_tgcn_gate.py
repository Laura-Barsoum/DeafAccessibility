"""Unit tests for when the TGCN sign tier runs and how its answer is ranked.

The tier runs by default only when weights trained on MediaPipe keypoints are
present, and ACCESSIBILITY_ENABLE_TGCN overrides that either way. When it fires,
its label leads the sequence: its softmax probability and the window voters'
weighted vote counts are not comparable, and ranking them against each other
left the handshape rules in front of the trained network.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules import sign_language as sl  # noqa: E402
from modules.tgcn_sign_model import TGCNSignRecognizer  # noqa: E402


class _StubTGCN:
    """Stands in for a loaded TGCN: always returns the same top-3."""
    threshold = 0.05
    layout = "openpose_swapped"

    def __init__(self, preds):
        self.preds = preds

    def is_available(self):
        return True

    def predict(self, keypoints, top_k=3):
        return self.preds[:top_k]


class TGCNLeadsTests(unittest.TestCase):
    """A fired TGCN prediction leads, however confident the window voters are."""

    def _recogniser(self, preds):
        slr = sl.SignLanguageRecognizer.__new__(sl.SignLanguageRecognizer)
        slr._tgcn = _StubTGCN(preds)
        slr._ensure_tgcn = lambda: None
        slr._ensure_gesture_recognizer = lambda: None
        slr._ensure_asl_letter_classifier = lambda: None
        # Every frame: a confident "thumbs up" from the gesture tier, no letters,
        # and keypoints good enough for the TGCN.
        slr.recognize_pretrained = lambda f: ("thumbs up", 0.99)
        slr.recognize_asl_letter = lambda f: None
        slr.extract_landmarks = lambda f: None
        return slr

    def test_fired_prediction_leads_a_confident_window_vote(self):
        slr = self._recogniser([("book", 0.31), ("drink", 0.12), ("cool", 0.05)])
        with patch.object(sl, "extract_55_keypoints", lambda raw, layout: [0.0]):
            seq, diag = slr.classify_all_frames_combined(["f"] * 12, window_size=8, stride=3)
        self.assertEqual(seq[0]["label"], "book")
        self.assertEqual(seq[0]["source"], "tgcn_wlasl")
        self.assertEqual(diag["tgcn_top_prediction"], ["book", 0.31])
        # the window detection is kept, behind it
        self.assertIn("thumbs up", [s["label"] for s in seq[1:]])

    def test_prediction_below_the_threshold_does_not_lead(self):
        slr = self._recogniser([("book", 0.04), ("drink", 0.02), ("cool", 0.01)])
        with patch.object(sl, "extract_55_keypoints", lambda raw, layout: [0.0]):
            seq, diag = slr.classify_all_frames_combined(["f"] * 12, window_size=8, stride=3)
        self.assertIsNone(diag["tgcn_top_prediction"])
        self.assertEqual(seq[0]["label"], "thumbs up")

    def test_the_leading_label_is_not_repeated(self):
        slr = self._recogniser([("thumbs up", 0.40), ("book", 0.10), ("cool", 0.05)])
        with patch.object(sl, "extract_55_keypoints", lambda raw, layout: [0.0]):
            seq, _diag = slr.classify_all_frames_combined(["f"] * 12, window_size=8, stride=3)
        self.assertEqual([s["label"] for s in seq].count("thumbs up"), 1)
        self.assertEqual(seq[0]["source"], "tgcn_wlasl")


class TGCNGateTests(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        patcher = patch.object(TGCNSignRecognizer, "LOCAL_DIR", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _trained_weights(self):
        (self.dir / "asl100_mediapipe").mkdir()
        (self.dir / "asl100_mediapipe" / "pytorch_model.bin").write_bytes(b"x")

    def test_off_without_trained_weights(self):
        with patch.dict(os.environ, {"ACCESSIBILITY_ENABLE_TGCN": ""}):
            self.assertFalse(sl.tgcn_enabled())

    def test_on_with_trained_weights(self):
        self._trained_weights()
        with patch.dict(os.environ, {"ACCESSIBILITY_ENABLE_TGCN": ""}):
            self.assertTrue(sl.tgcn_enabled())

    def test_environment_overrides_both_ways(self):
        self._trained_weights()
        with patch.dict(os.environ, {"ACCESSIBILITY_ENABLE_TGCN": "0"}):
            self.assertFalse(sl.tgcn_enabled())
        (self.dir / "asl100_mediapipe" / "pytorch_model.bin").unlink()
        with patch.dict(os.environ, {"ACCESSIBILITY_ENABLE_TGCN": "1"}):
            self.assertTrue(sl.tgcn_enabled())


if __name__ == "__main__":
    unittest.main()
