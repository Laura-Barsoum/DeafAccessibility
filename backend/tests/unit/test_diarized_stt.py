"""Unit tests for the DiarizedSTT merge logic and speaker-name store.

We don't run real Whisper / pyannote here — those are integration-level
concerns. Instead we test the pure-Python merge + persistence layer.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_TMP_DIR = tempfile.mkdtemp(prefix="diarized_test_")
from modules import diarized_stt as dz  # noqa: E402
# Redirect storage
dz._SPEAKERS_PATH = Path(_TMP_DIR) / "speakers.json"


class MergeTests(unittest.TestCase):
    """Time-overlap merge between Whisper segments and pyannote turns."""

    def setUp(self):
        # Pretend the module has never loaded the speakers file.
        dz._SPEAKERS_PATH = Path(_TMP_DIR) / f"speakers_{self._testMethodName}.json"

    def test_merge_picks_max_overlap(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        diar = [
            {"start": 0.0, "end": 0.3, "speaker": "SPEAKER_00"},
            {"start": 0.3, "end": 1.0, "speaker": "SPEAKER_01"},  # bigger overlap
        ]
        out = dz.DiarizedSTT._merge(segments, diar)
        self.assertEqual(out[0]["speaker"], "SPEAKER_01")

    def test_merge_no_diarization_returns_none_speaker(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "hi"}]
        out = dz.DiarizedSTT._merge(segments, [])
        self.assertIsNone(out[0]["speaker"])
        self.assertEqual(out[0]["text"], "hi")

    def test_merge_preserves_segment_timestamps(self):
        segments = [
            {"start": 0.1, "end": 0.9, "text": "a"},
            {"start": 1.2, "end": 2.0, "text": "b"},
        ]
        diar = [{"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}]
        out = dz.DiarizedSTT._merge(segments, diar)
        self.assertEqual(out[0]["start"], 0.1)
        self.assertEqual(out[1]["end"], 2.0)

    def test_merge_zero_overlap_speaker_is_none(self):
        segments = [{"start": 10.0, "end": 11.0, "text": "late"}]
        diar = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
        out = dz.DiarizedSTT._merge(segments, diar)
        self.assertIsNone(out[0]["speaker"])


class SpeakerNameStoreTests(unittest.TestCase):
    """The friendly speaker-name persistence store."""
    def setUp(self):
        dz._SPEAKERS_PATH = Path(_TMP_DIR) / f"speakers_{self._testMethodName}.json"
        # New instance per test
        self.d = dz.DiarizedSTT()

    def test_name_then_lookup(self):
        self.d.name_speaker("SPEAKER_00", "Sarah")
        self.assertEqual(self.d.get_speaker_name("SPEAKER_00"), "Sarah")
        self.assertIsNone(self.d.get_speaker_name("SPEAKER_99"))

    def test_persist_across_instance(self):
        self.d.name_speaker("SPEAKER_00", "Sarah")
        d2 = dz.DiarizedSTT()
        self.assertEqual(d2.get_speaker_name("SPEAKER_00"), "Sarah")

    def test_forget_removes_mapping(self):
        self.d.name_speaker("SPEAKER_00", "Sarah")
        self.d.forget_speaker("SPEAKER_00")
        self.assertIsNone(self.d.get_speaker_name("SPEAKER_00"))

    def test_list_includes_seen_unnamed(self):
        # Simulate having seen an un-named id from a real /process tick
        self.d._seen_speakers.add("SPEAKER_77")
        out = self.d.list_speakers()
        self.assertIn("SPEAKER_77", out["seen_unnamed"])

    def test_empty_name_or_id_silently_ignored(self):
        # Don't crash, don't create junk entries
        self.d.name_speaker("", "Sarah")
        self.d.name_speaker("SPEAKER_00", "")
        self.assertEqual(self.d._speaker_names, {})


def tearDownModule():
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
