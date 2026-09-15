"""Unit tests for when the TGCN sign tier runs.

The tier runs by default only when weights trained on MediaPipe keypoints are
present, and ACCESSIBILITY_ENABLE_TGCN overrides that either way.
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
