"""Regression test for the MediaPipe Tasks landmark adapter.

Hand landmarks from the Tasks API carry visibility=None. Converting that with
float() raised inside _TasksHolistic.process, whose error handling then left
both hand slots empty on every frame, silently removing hands from every
hand-based sign tier.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules.sign_language import _wrap_landmarks  # noqa: E402


class TasksLandmarkTests(unittest.TestCase):

    def test_hand_landmarks_without_visibility_are_kept(self):
        hand = [SimpleNamespace(x=0.1 * i, y=0.2, z=0.0, visibility=None, presence=None) for i in range(21)]
        wrapped = _wrap_landmarks(hand)
        self.assertEqual(len(wrapped), 21)
        self.assertAlmostEqual(wrapped.landmark[3].x, 0.3)
        self.assertEqual(wrapped.landmark[0].visibility, 1.0)

    def test_pose_visibility_is_preserved(self):
        wrapped = _wrap_landmarks([SimpleNamespace(x=0.5, y=0.5, z=0.1, visibility=0.25)])
        self.assertEqual(wrapped.landmark[0].visibility, 0.25)


if __name__ == "__main__":
    unittest.main()
