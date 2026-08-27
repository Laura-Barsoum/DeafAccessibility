"""Unit tests for the fusion engine."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from modules.fusion import Fusion
from modules.events import Event, Priority


class FusionTests(unittest.TestCase):
    """The multimodal fusion engine: ranking, dedup and headline selection."""
    def setUp(self):
        self.fuser = Fusion()

    def test_no_input_no_events(self):
        out = self.fuser.fuse(
            stt_result={"text": "", "segments": []},
            sound_events=[],
            personal_events=[],
            lip_reliability={"reliability": 0.5},
            face_attribution={"speaker_attribution": None, "faces": []},
            scene_caption=None,
        )
        self.assertEqual(out["n_events"], 0)
        self.assertEqual(out["n_critical"], 0)
        self.assertEqual(len(out["headlines"]), 0)

    def test_critical_sound_makes_headline(self):
        sound_events = [
            Event(source="sound", label="Smoke alarm",
                  priority=Priority.CRITICAL, confidence=0.92),
        ]
        out = self.fuser.fuse(
            stt_result={"text": "", "segments": []},
            sound_events=sound_events,
            personal_events=[],
            lip_reliability={"reliability": 0.5},
            face_attribution={"speaker_attribution": None, "faces": []},
            scene_caption="kitchen",
        )
        self.assertEqual(out["n_critical"], 1)
        self.assertEqual(len(out["headlines"]), 1)
        self.assertEqual(out["headlines"][0]["priority_name"], "CRITICAL")

    def test_speech_help_word_promoted_to_critical(self):
        out = self.fuser.fuse(
            stt_result={"text": "Help me please", "segments": []},
            sound_events=[],
            personal_events=[],
            lip_reliability={"reliability": 0.8},
            face_attribution={"speaker_attribution": None, "faces": []},
            scene_caption=None,
        )
        self.assertEqual(out["n_critical"], 1)

    def test_personal_event_takes_priority(self):
        sound_events = [
            Event(source="sound", label="Bell", priority=Priority.INFORM,
                  confidence=0.4),
        ]
        personal_events = [
            Event(source="sound", label="my doorbell (personal)",
                  priority=Priority.IMPORTANT, confidence=0.9),
        ]
        out = self.fuser.fuse(
            stt_result={"text": "", "segments": []},
            sound_events=sound_events,
            personal_events=personal_events,
            lip_reliability={"reliability": 0.5},
            face_attribution={"speaker_attribution": None, "faces": []},
            scene_caption=None,
        )
        # First headline should be the IMPORTANT personal event
        self.assertGreaterEqual(len(out["headlines"]), 1)
        self.assertEqual(out["headlines"][0]["priority_name"], "IMPORTANT")
        self.assertIn("personal", out["headlines"][0]["label"].lower())

    def test_speech_speaker_attribution(self):
        face = {"speaker_attribution": {"position": "left", "mar": 0.3,
                                         "speaking_score": 0.04, "face_id": 0,
                                         "speaking": True},
                "faces": []}
        out = self.fuser.fuse(
            stt_result={"text": "Hello there", "segments": []},
            sound_events=[],
            personal_events=[],
            lip_reliability={"reliability": 0.85},
            face_attribution=face,
            scene_caption=None,
        )
        speech_evt = next(e for e in out["events"] if e["source"] == "speech")
        self.assertEqual(speech_evt["speaker"], "speaker_left")


if __name__ == "__main__":
    unittest.main()
