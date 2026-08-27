"""Route tests for sign recognition and sign-to-speech flow."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server  # noqa: E402


class DummySLR:
    """A stub sign-language recogniser used to exercise the /sign/recognize route without loading real models."""
    def __init__(self, event, *, configured=True, min_frames=1):
        self._event = event
        self._configured = configured
        self._min_frames = min_frames
        self.frames_seen = None
        self.reset_calls = 0
        self.gloss_calls = []

    def reset_buffer(self):
        self.reset_calls += 1

    def push_frames(self, frames):
        self.frames_seen = frames
        return len(frames)

    def minimum_frames_required(self):
        return self._min_frames

    def is_configured(self):
        return self._configured

    def classify_to_event(self):
        return self._event

    def classify_all_frames_combined(self, frames_b64, window_size=8, stride=3):
        # The /sign/recognize endpoint was extended to call this combined
        # multi-tier classifier. We mirror its (sequence, diag) return
        # shape so the existing tests keep exercising the route end-to-end.
        self.frames_seen = list(frames_b64)
        # Translate the old DummySLR knobs into the diagnostics the new
        # route reads:
        #   - configured=False  → 0 hand frames detected (looks like
        #     "no hands visible" to the route)
        #   - min_frames > 1    → 0 hand frames detected too
        #   - default           → frames_with_hand_* = len(frames) so the
        #     route hits the no_match branch instead.
        hands_detected = len(frames_b64) if (
            self._configured and self._min_frames <= 1
        ) else 0
        if self._event is None:
            return [], {"frames_received": len(frames_b64),
                        "frames_with_hand_pretrained": hands_detected,
                        "frames_with_hand_geometric": hands_detected,
                        "tgcn_top_prediction": None,
                        "spelled_words": []}
        label = self._event.label.replace("sign: ", "")
        return ([{
            "label": label,
            "confidence": getattr(self._event, "confidence", 0.9),
            "start_frame": 0,
            "end_frame": len(frames_b64),
            "source": "test",
        }], {
            "frames_received": len(frames_b64),
            "frames_with_hand_pretrained": len(frames_b64),
            "frames_with_hand_geometric": len(frames_b64),
            "tgcn_top_prediction": None,
            "spelled_words": [],
        })

    def sequence_to_gloss(self, sequence):
        # Concatenate the labels in the sequence into a single gloss line —
        # mirrors the real SLR's behaviour without pulling in any models.
        return " ".join(s.get("label", "") for s in sequence).strip()

    def gloss_to_speech(self, gloss, llm):
        self.gloss_calls.append((gloss, llm))
        return f"spoken:{gloss}"


class SignRecognizeRouteTests(unittest.TestCase):
    """The /sign/recognize endpoint contract."""
    def setUp(self):
        self.client = server.app.test_client()

    def test_sign_recognize_returns_label_and_text(self):
        slr = DummySLR(SimpleNamespace(
            label="sign: hello",
            text="hello",
            confidence=0.91,
        ))

        with patch.object(server, "get_slr", return_value=slr), patch.object(
            server, "get_llm", return_value=object()
        ):
            resp = self.client.post("/sign/recognize", json={"frames_b64": ["f1", "f2"]})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["label"], "hello")
        self.assertEqual(data["text"], "spoken:hello")
        self.assertAlmostEqual(data["confidence"], 0.91)
        self.assertEqual(slr.frames_seen, ["f1", "f2"])
        self.assertEqual(slr.reset_calls, 2)
        self.assertEqual(slr.gloss_calls[0][0], "hello")

    def test_sign_recognize_returns_empty_when_no_sign_matches(self):
        # configured=True + min_frames=1 → mock reports hands ARE present
        # but no sign matched. Route should return reason="no_match" and
        # only call reset_buffer once (early-return path).
        slr = DummySLR(None)

        with patch.object(server, "get_slr", return_value=slr), patch.object(
            server, "get_llm", return_value=object()
        ):
            resp = self.client.post("/sign/recognize", json={"frames_b64": ["f1"]})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["label"], "")
        self.assertEqual(data["text"], "")
        # The route's early-return path omits a "confidence" field (only
        # present on the happy path). Either way we expect 0 / missing.
        self.assertEqual(data.get("confidence", 0.0), 0.0)
        self.assertEqual(data["reason"], "no_match")
        self.assertEqual(slr.reset_calls, 1)

    def test_sign_recognize_requires_frames(self):
        resp = self.client.post("/sign/recognize", json={"frames_b64": []})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["reason"], "missing_frames")

    def test_sign_recognize_reports_missing_hands(self):
        # When the combined classifier returns an empty sequence AND the
        # diagnostics show zero hand-frames detected, the new route
        # responds with reason="no_hands_visible". (Older versions called
        # this "hands_not_detected" and relied on minimum_frames_required.)
        slr = DummySLR(None, min_frames=3)

        with patch.object(server, "get_slr", return_value=slr), patch.object(
            server, "get_llm", return_value=object()
        ):
            resp = self.client.post("/sign/recognize", json={"frames_b64": ["f1"]})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["reason"], "no_hands_visible")
        self.assertEqual(slr.reset_calls, 1)

    def test_sign_recognize_reports_unconfigured_recognizer(self):
        # The refactored route no longer pre-checks is_configured() — the
        # combined classifier handles that internally and an empty
        # sequence with zero hand-frames degenerates to the same
        # "no_hands_visible" response.
        slr = DummySLR(None, configured=False)

        with patch.object(server, "get_slr", return_value=slr), patch.object(
            server, "get_llm", return_value=object()
        ):
            resp = self.client.post("/sign/recognize", json={"frames_b64": ["f1"]})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["reason"], "no_hands_visible")
        self.assertEqual(slr.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
