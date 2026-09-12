"""Unit tests for face tracking after the MediaPipe Tasks-API port.

Regression tests for a real defect: `mediapipe.solutions` was removed, the
legacy FaceMesh import failed, and the tracker silently became a placeholder,
so speaker attribution never ran (the stage profiled at 0 ms).
"""
from __future__ import annotations

import base64
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules import face_tracker as ft  # noqa: E402


def _blank_jpeg_b64(w=320, h=240):
    import cv2
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 127, dtype=np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def _face(cx, mouth_open):
    """A 478-point face whose centre is at cx and whose mouth gap is mouth_open."""
    pts = [SimpleNamespace(x=cx, y=0.5, z=0.0) for _ in range(478)]
    pts[ft.UPPER_LIP_TOP] = SimpleNamespace(x=cx, y=0.60, z=0.0)
    pts[ft.LOWER_LIP_BOT] = SimpleNamespace(x=cx, y=0.60 + mouth_open, z=0.0)
    pts[ft.LIP_LEFT] = SimpleNamespace(x=cx - 0.05, y=0.62, z=0.0)
    pts[ft.LIP_RIGHT] = SimpleNamespace(x=cx + 0.05, y=0.62, z=0.0)
    return pts


class FakeMesh:
    """Stands in for _TasksFaceMesh, returning scripted faces per call."""

    def __init__(self, script):
        self.script = list(script)

    def process(self, _rgb):
        faces = self.script.pop(0) if self.script else []
        return SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=f) for f in faces])


class FaceTrackerTests(unittest.TestCase):
    """Landmark handling, position, mouth-aspect ratio and speaking score."""

    def test_position_and_mar_are_computed(self):
        t = ft.FaceTracker()
        t._mesh = FakeMesh([[_face(0.2, 0.05)]])
        faces = t.analyse_frame(_blank_jpeg_b64())
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["position"], "left")
        self.assertAlmostEqual(faces[0]["mar"], 0.5, places=3)   # 0.05 / 0.10

    def test_moving_mouth_is_attributed_as_speaking(self):
        t = ft.FaceTracker()
        gaps = [0.01, 0.06, 0.01, 0.07, 0.01, 0.06, 0.02, 0.07]
        t._mesh = FakeMesh([[_face(0.5, g)] for g in gaps])
        frames = [_blank_jpeg_b64()] * len(gaps)
        verdict = t.analyse_frames(frames)
        self.assertIsNotNone(verdict["speaker_attribution"])
        self.assertEqual(verdict["speaker_attribution"]["position"], "centre")

    def test_still_mouth_is_not_attributed(self):
        t = ft.FaceTracker()
        t._mesh = FakeMesh([[_face(0.8, 0.02)] for _ in range(8)])
        verdict = t.analyse_frames([_blank_jpeg_b64()] * 8)
        self.assertIsNone(verdict["speaker_attribution"])

    def test_no_face_returns_empty(self):
        t = ft.FaceTracker()
        t._mesh = FakeMesh([[]])
        self.assertEqual(t.analyse_frame(_blank_jpeg_b64()), [])

    @unittest.skipUnless(os.path.exists(ft._FACE_MODEL), "face_landmarker.task not downloaded")
    def test_tasks_backend_loads_instead_of_placeholder(self):
        t = ft.FaceTracker()
        t._ensure_loaded()
        self.assertNotEqual(t._mesh, "placeholder")
        # A blank frame has no face, but the real detector must run cleanly.
        self.assertEqual(t.analyse_frame(_blank_jpeg_b64()), [])


if __name__ == "__main__":
    unittest.main()
