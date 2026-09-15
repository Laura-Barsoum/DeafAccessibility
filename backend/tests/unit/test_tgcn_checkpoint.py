"""Unit tests for loading the WLASL TGCN checkpoint and preparing its input.

The network once had layers no checkpoint contains and loaded with
strict=False, so the sign tier predicted with random weights. No real
checkpoint is needed here: one is written from the network's own weights.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from modules import tgcn_sign_model as tg  # noqa: E402


@unittest.skipIf(torch is None, "torch not installed")
class TGCNCheckpointTests(unittest.TestCase):

    def _load(self, state):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "pytorch_model.bin")
        torch.save(state, path)
        rec = tg.TGCNSignRecognizer("asl100")
        with patch.dict(os.environ, {"ACCESSIBILITY_TGCN_LOCAL_PATH": path}):
            return rec, rec.load()

    def test_network_has_the_released_layers(self):
        keys = set(tg.build_TGCN(100, 64, 20).state_dict())
        self.assertIn("fc_out.weight", keys)
        self.assertFalse(any(k.startswith(("attention.", "gc7.", "fc.")) for k in keys))
        self.assertEqual(len(keys), 330)      # as in the published asl100 checkpoint

    def test_matching_checkpoint_loads_and_predicts(self):
        rec, ok = self._load({"state_dict": tg.build_TGCN(100, 64, 20).state_dict()})
        self.assertTrue(ok)
        seq = [np.random.default_rng(i).random((55, 2)).astype(np.float32) for i in range(20)]
        preds = rec.predict(seq, top_k=3)
        self.assertEqual(len(preds), 3)
        self.assertTrue(all(label in tg.WLASL100_DEFAULT_ORDER for label, _p in preds))

    def test_checkpoint_that_does_not_fit_is_refused(self):
        state = dict(tg.build_TGCN(100, 64, 20).state_dict())
        state["fc.weight"] = state.pop("fc_out.weight")
        rec, ok = self._load(state)
        self.assertFalse(ok)
        self.assertFalse(rec.is_available())

    def test_settings_saved_beside_a_checkpoint_are_used(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "pytorch_model.bin")
        torch.save(tg.build_TGCN(100, 64, 20).state_dict(), path)
        with open(os.path.join(tmp.name, "config.json"), "w") as f:
            json.dump({"layout": "openpose", "coords": "body", "class_order": "manifest", "threshold": 0.6}, f)
        rec = tg.TGCNSignRecognizer("asl100")
        with patch.dict(os.environ, {"ACCESSIBILITY_TGCN_LOCAL_PATH": path}):
            self.assertTrue(rec.load())
        self.assertEqual((rec.layout, rec.coords, rec.threshold), ("openpose", "body", 0.6))
        self.assertEqual(rec.vocab, list(tg.WLASL100_DEFAULT_ORDER))

    def test_vocabulary_is_the_100_glosses(self):
        self.assertEqual(len(tg.WLASL100_DEFAULT_ORDER), 100)
        self.assertEqual(len(set(tg.WLASL100_DEFAULT_ORDER)), 100)


class KeypointLayoutTests(unittest.TestCase):

    @staticmethod
    def _landmarks(points):
        return type("Landmarks", (), {"landmark": [type("P", (), {"x": x, "y": y})() for x, y in points]})()

    def _result(self):
        r = type("Result", (), {})()
        r.pose_landmarks = self._landmarks([(i / 100, i / 100) for i in range(33)])
        r.right_hand_landmarks = self._landmarks([(0.9, 0.9)] * 21)   # MediaPipe "Right": the signer's left
        r.left_hand_landmarks = self._landmarks([(0.1, 0.1)] * 21)
        return r

    def test_openpose_order(self):
        kp = tg.extract_55_keypoints(self._result(), "openpose")
        np.testing.assert_allclose(kp[1], [0.115, 0.115], atol=1e-6)   # neck: midpoint of shoulders 11 and 12
        np.testing.assert_allclose(kp[2], [0.12, 0.12], atol=1e-6)     # signer's right shoulder, MediaPipe 12
        np.testing.assert_allclose(kp[13], [0.9, 0.9], atol=1e-6)
        np.testing.assert_allclose(kp[34], [0.1, 0.1], atol=1e-6)

    def test_swapped_hands(self):
        kp = tg.extract_55_keypoints(self._result(), "openpose_swapped")
        np.testing.assert_allclose(kp[13], [0.1, 0.1], atol=1e-6)
        np.testing.assert_allclose(kp[34], [0.9, 0.9], atol=1e-6)

    def test_body_coordinates_centre_on_the_neck(self):
        kp = np.zeros((55, 2), np.float32)
        kp[1], kp[2], kp[5], kp[13] = (0.5, 0.4), (0.4, 0.4), (0.6, 0.4), (0.5, 0.6)
        out = tg.to_model_coords([kp], "body")[0]
        np.testing.assert_allclose(out[1], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(out[13], [0.0, 0.5], atol=1e-6)   # 0.2 below the neck over twice the shoulder width
        self.assertTrue(np.all(out[20] == 0))                         # an undetected point stays at 0

    def test_signed_coordinates_put_missing_points_at_minus_one(self):
        out = tg.to_model_coords([np.zeros((55, 2), np.float32)], "signed")
        self.assertTrue(np.all(out[0] == -1.0))


if __name__ == "__main__":
    unittest.main()
