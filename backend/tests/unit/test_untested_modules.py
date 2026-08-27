"""Unit tests for modules that previously had no test coverage.

These target pure logic that can be exercised without loading any heavy
model: the fingerspelling buffer, the sound-event thresholding rules, the
caption reliability scorer, and the TGCN keypoint/resampling helpers.
Model-dependent paths are covered by the integration tests instead.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules.spell_buffer import SpellBuffer            # noqa: E402
from modules.audio_scene import AudioSceneClassifier    # noqa: E402
from modules.lip_reader import LipReader                # noqa: E402
from modules import tgcn_sign_model as tg               # noqa: E402


class SpellBufferTests(unittest.TestCase):
    """Fingerspelling: letters become words only after stable repetition."""

    def test_stable_letters_form_a_word(self):
        # Three consecutive identical frames confirm a letter; a run of
        # None frames closes the word.
        seq = ["Y"]*3 + ["O"]*3 + ["U"]*3 + [None]*8
        self.assertEqual(SpellBuffer.from_letter_sequence(seq), ["YOU"])

    def test_two_words_separated_by_a_pause(self):
        seq = ["O"]*3 + ["K"]*3 + [None]*8 + ["Y"]*3 + ["E"]*3 + ["S"]*3 + [None]*8
        self.assertEqual(SpellBuffer.from_letter_sequence(seq), ["OK", "YES"])

    def test_flicker_is_not_committed(self):
        # A single stray frame must not become a letter.
        seq = ["A"] + ["B"]*3 + [None]*8
        self.assertEqual(SpellBuffer.from_letter_sequence(seq), ["B"])

    def test_empty_input_yields_no_words(self):
        self.assertEqual(SpellBuffer.from_letter_sequence([]), [])

    def test_reset_clears_state(self):
        b = SpellBuffer()
        for c in ["A"]*3:
            b.feed(c)
        b.reset()
        self.assertEqual(b.flush(), [])


class AudioSceneThresholdTests(unittest.TestCase):
    """Event construction from classifier output: thresholding and suppression."""

    def setUp(self):
        self.c = AudioSceneClassifier()

    def test_low_confidence_predictions_are_discarded(self):
        # Patch the classifier to return a weak prediction; below the
        # min_conf floor it must not reach the user.
        self.c.classify = lambda *a, **k: [("Stomach rumble", 0.06)]
        self.assertEqual(self.c.classify_to_events(b"x"), [])

    def test_confident_prediction_becomes_an_event(self):
        self.c.classify = lambda *a, **k: [("Doorbell", 0.42)]
        events = self.c.classify_to_events(b"x")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].label, "Doorbell")
        self.assertEqual(events[0].source, "sound")

    def test_pure_noise_labels_are_suppressed(self):
        # "Silence" is uninformative and must never be surfaced, even at
        # high confidence.
        self.c.classify = lambda *a, **k: [("Silence", 0.99)]
        self.assertEqual(self.c.classify_to_events(b"x"), [])

    def test_critical_sound_gets_critical_priority(self):
        self.c.classify = lambda *a, **k: [("Smoke detector", 0.55)]
        events = self.c.classify_to_events(b"x")
        self.assertEqual(events[0].priority.name, "CRITICAL")


class LipReliabilityTests(unittest.TestCase):
    """The caption reliability scorer (not lip reading, see module docstring)."""

    def setUp(self):
        self.lr = LipReader()

    def test_reliability_is_bounded(self):
        for mar in ([], [0.0], [0.5]*10, [0.01, 0.9, 0.02]):
            r = self.lr.reliability(mar, 0.02).get("reliability")
            self.assertIsInstance(r, float)
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)

    def test_no_mouth_data_still_returns_a_score(self):
        # Absent visual data must degrade to a neutral score rather than
        # raising, since the fusion layer consumes this every tick.
        self.assertIn("reliability", self.lr.reliability([], 0.02))


class TGCNHelperTests(unittest.TestCase):
    """Keypoint extraction and temporal resampling, independent of weights."""

    def test_extract_returns_none_without_landmarks(self):
        self.assertIsNone(tg.extract_55_keypoints(None))

    def test_resample_pads_short_sequences_to_fixed_length(self):
        seq = [np.zeros((55, 2), dtype=np.float32) for _ in range(7)]
        out = tg._resample_frames(seq, target_frames=50)
        self.assertEqual(out.shape, (55, 100))   # 50 frames x (x, y)

    def test_resample_compresses_long_sequences(self):
        seq = [np.ones((55, 2), dtype=np.float32) for _ in range(300)]
        out = tg._resample_frames(seq, target_frames=50)
        self.assertEqual(out.shape, (55, 100))

    def test_resample_of_empty_sequence_is_zeros(self):
        out = tg._resample_frames([], target_frames=50)
        self.assertEqual(out.shape, (55, 100))
        self.assertTrue(np.allclose(out, 0.0))

    def test_recogniser_reports_unavailable_without_checkpoint(self):
        # The public checkpoint could not be obtained, so the tier must
        # report itself unavailable rather than failing at inference time.
        rec = tg.TGCNSignRecognizer(variant="asl100")
        self.assertFalse(rec.is_available())


if __name__ == "__main__":
    unittest.main()
