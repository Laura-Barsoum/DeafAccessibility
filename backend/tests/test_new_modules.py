"""Unit tests for the new advanced modules: SLR, hazard detector,
emotion fusion, and sound localization (the math/logic parts that don't
require heavy ML deps)."""
import sys, os, math, struct, io, wave
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np

from modules import emotion, hazard_detector, localization, sign_language
from modules.events import Priority


# ---------------------------------------------------------------------
# Hazard detector — radar position math
# ---------------------------------------------------------------------

class HazardRadarTests(unittest.TestCase):
    """YOLO hazard-to-radar mapping (direction and approach)."""
    def test_centre(self):
        self.assertEqual(hazard_detector._radar_position(0.5, 0.5), "centre")

    def test_left(self):
        # Left of centre, mid height
        pos = hazard_detector._radar_position(0.1, 0.5)
        self.assertEqual(pos, "west")

    def test_right(self):
        pos = hazard_detector._radar_position(0.9, 0.5)
        self.assertEqual(pos, "east")

    def test_north(self):
        pos = hazard_detector._radar_position(0.5, 0.05)
        self.assertEqual(pos, "north")

    def test_south(self):
        pos = hazard_detector._radar_position(0.5, 0.95)
        self.assertEqual(pos, "south")

    def test_keyword_extraction(self):
        out = hazard_detector.HazardDetector.keywords_in_scene(
            "a kitchen with smoke and fire on the stove")
        labels = {e.label for e in out}
        self.assertTrue(any("fire" in l for l in labels))
        self.assertTrue(any("smoke" in l for l in labels))
        # All such matches should be CRITICAL
        for e in out:
            self.assertEqual(e.priority, Priority.CRITICAL)


# ---------------------------------------------------------------------
# Emotion fusion
# ---------------------------------------------------------------------

class EmotionFusionTests(unittest.TestCase):
    """Audio/visual emotion fusion and caption tagging."""
    def test_agreement(self):
        audio = {"label": "happy", "confidence": 0.9,
                 "probs": {"happy": 0.9, "neutral": 0.05, "sad": 0.05}}
        visual = {"label": "happy", "confidence": 0.8,
                  "probs": {"happy": 0.8, "surprise": 0.1, "neutral": 0.1}}
        out = emotion.fuse_emotions(audio, visual)
        self.assertEqual(out["label"], "happy")
        self.assertFalse(out["mixed_signal"])

    def test_mixed_signal(self):
        audio = {"label": "angry", "confidence": 0.85,
                 "probs": {"angry": 0.85, "neutral": 0.15}}
        visual = {"label": "happy", "confidence": 0.8,
                  "probs": {"happy": 0.8, "neutral": 0.2}}
        out = emotion.fuse_emotions(audio, visual)
        self.assertTrue(out["mixed_signal"])

    def test_caption_tag_neutral_unchanged(self):
        fused = {"label": "neutral", "emoji": "😐", "mixed_signal": False,
                 "audio": {"label": "neutral", "confidence": 0.5},
                 "visual": {"label": "neutral", "confidence": 0.5}}
        out = emotion.tag_caption("hello world", fused)
        self.assertEqual(out, "hello world")

    def test_caption_tag_angry_high_confidence(self):
        # High confidence + both channels agree → MUST tag.
        fused = {"label": "angry", "emoji": "😡", "mixed_signal": False,
                 "confidence": 0.78,
                 "audio": {"label": "angry", "confidence": 0.7},
                 "visual": {"label": "angry", "confidence": 0.7}}
        out = emotion.tag_caption("Stop it!", fused)
        self.assertIn("😡", out)
        self.assertIn("Angry", out)

    def test_caption_tag_angry_low_confidence_is_NOT_tagged(self):
        # Low fused confidence → must NOT tag (suppresses false positives).
        fused = {"label": "angry", "emoji": "😡", "mixed_signal": False,
                 "confidence": 0.25,
                 "audio": {"label": "angry", "confidence": 0.3},
                 "visual": {"label": "angry", "confidence": 0.3}}
        out = emotion.tag_caption("Hello there", fused)
        self.assertEqual(out, "Hello there")

    def test_caption_tag_mixed(self):
        fused = {"label": "happy", "emoji": "😊", "mixed_signal": True,
                 "confidence": 0.65,
                 "audio": {"label": "angry", "confidence": 0.7},
                 "visual": {"label": "happy", "confidence": 0.7}}
        out = emotion.tag_caption("I'm fine", fused)
        # Should call out the mismatch
        self.assertIn("voice", out)
        self.assertIn("face", out)


# ---------------------------------------------------------------------
# Sound localization — math
# ---------------------------------------------------------------------

class LocalizationTests(unittest.TestCase):
    """GCC-PHAT stereo sound direction-of-arrival."""
    def test_compass_zero_is_front(self):
        self.assertEqual(localization._azimuth_to_compass(0), "front")

    def test_compass_left(self):
        self.assertEqual(localization._azimuth_to_compass(-90), "left")

    def test_compass_right(self):
        self.assertEqual(localization._azimuth_to_compass(90), "right")

    def test_compass_front_right(self):
        self.assertEqual(localization._azimuth_to_compass(45), "front-right")

    def test_tau_to_azimuth_zero_delay(self):
        # Zero delay → directly in front
        a = localization._tau_to_azimuth(0.0, mic_distance_m=0.18)
        self.assertAlmostEqual(a, 0.0, places=2)

    def test_tau_to_azimuth_positive(self):
        # Positive delay = sound arrives at left mic later → source on right
        a = localization._tau_to_azimuth(0.0001, mic_distance_m=0.18)
        self.assertGreater(a, 0)

    def test_back_or_front_inference(self):
        loc = localization.SoundLocalizer()
        # 'front' compass + no visible speaker → infer behind
        self.assertEqual(loc.infer_back_or_front("front", False), "behind")
        # 'left' compass — keep regardless of speaker visibility
        self.assertEqual(loc.infer_back_or_front("left", False), "left")

    def test_localize_mono_returns_unknown(self):
        # Make a tiny mono wav
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            samples = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
                       * 16000).astype(np.int16)
            w.writeframes(samples.tobytes())
        loc = localization.SoundLocalizer()
        out = loc.localize(buf.getvalue(), sample_rate=16000)
        self.assertEqual(out["compass"], "unknown")
        self.assertFalse(out["stereo"])


# ---------------------------------------------------------------------
# Sign language — vocab + cosine math
# ---------------------------------------------------------------------

class SignLanguageTests(unittest.TestCase):
    """Sign-language recognition helpers and vocabulary."""
    def test_default_vocab_present(self):
        self.assertIn("help", sign_language.DEFAULT_VOCAB)
        self.assertIn("fire", sign_language.DEFAULT_VOCAB)
        self.assertIn("hello", sign_language.DEFAULT_VOCAB)

    def test_cosine_identical(self):
        v = np.array([1, 2, 3], dtype=np.float32)
        self.assertAlmostEqual(sign_language._cosine(v, v), 1.0, places=5)

    def test_cosine_orthogonal(self):
        a = np.array([1, 0, 0], dtype=np.float32)
        b = np.array([0, 1, 0], dtype=np.float32)
        self.assertAlmostEqual(sign_language._cosine(a, b), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
