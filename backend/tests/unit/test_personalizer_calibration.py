"""Unit tests for personal-sound settings chosen by embedding.

AST embeddings (768 values) need a wider gate than YamNet's (1,024); both sets
were chosen on the ESC-50 development split (fewshot_deployment.py). Here the
same geometry is enrolled in both sizes: a play between the two gate floors is
inside AST's gate and outside YamNet's.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules import personalizer as pm  # noqa: E402
from modules.events import Priority     # noqa: E402

# Other personaliser tests replace _embed on the class and do not restore it.
_REAL_EMBED = pm.Personalizer._embed
AST_DIM = pm.AST_EMBEDDING_DIM


class _Embedder:
    """Stands in for AST: returns a fixed vector per clip and counts calls."""
    def __init__(self, table):
        self.table, self.calls = table, 0

    def __call__(self, audio):
        self.calls += 1
        return self.table.get(audio)


class CalibrationByEmbeddingTests(unittest.TestCase):

    def setUp(self):
        for target, value in (("_embed", _REAL_EMBED), ("_save", lambda self: None)):
            patcher = patch.object(pm.Personalizer, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make(self, dim):
        rng = np.random.default_rng(0)
        base = np.zeros(dim, np.float32); base[0] = 1.0
        clips = {f"e{i}".encode(): base + rng.standard_normal(dim).astype(np.float32) * 0.002 for i in range(3)}
        d = (pm.CALIBRATION["ast"]["gate_floor"] + pm.CALIBRATION["yamnet"]["gate_floor"]) / 2   # squared distance
        cos = 1 - d / 2
        play = np.zeros(dim, np.float32); play[0], play[1] = cos, np.sqrt(1 - cos ** 2)
        embedder = _Embedder({**clips, b"play": play})
        p = pm.Personalizer(embedder=embedder)
        p.profile = {"sounds": {}}
        p.enrol("Kettle", Priority.IMPORTANT, list(clips))
        return p, embedder, play

    def test_ast_enrolment_is_recorded(self):
        p, _embedder, _play = self.make(AST_DIM)
        s = p.profile["sounds"]["Kettle"]
        self.assertEqual((s["embedder"], s["embedding_dim"]), ("ast", AST_DIM))

    def test_ast_settings_admit_a_play_yamnet_settings_reject(self):
        self.assertGreater(pm.CALIBRATION["ast"]["gate_floor"], pm.CALIBRATION["yamnet"]["gate_floor"])
        p, _embedder, _play = self.make(AST_DIM)
        self.assertEqual(len(p.match_prototypical(b"play")), 1)
        p, _embedder, _play = self.make(1024)
        self.assertEqual(p.match_prototypical(b"play"), [])

    def test_embedding_from_the_sound_classifier_is_reused(self):
        p, embedder, play = self.make(AST_DIM)
        calls = embedder.calls
        events = p.match_current(b"audio already embedded", embedding=play)
        self.assertEqual(embedder.calls, calls)
        self.assertEqual([e.label for e in events], ["Kettle (personal)"])

    def test_rejecting_an_alert_stops_that_play_matching(self):
        """Behaviour, not stored values: the tightened gate is applied at match time."""
        p, _embedder, _play = self.make(AST_DIM)
        events = p.match_prototypical(b"play")
        self.assertEqual(len(events), 1)
        for _ in range(5):
            if not events:
                break
            p.feedback(events[0].extra["match_id"], is_positive=False)
            events = p.match_prototypical(b"play")
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
