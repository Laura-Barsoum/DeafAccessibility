"""Unit tests for how the sound classifier chooses its backend.

Regression test for a real defect: transformers asks the Hugging Face Hub about
a model before using its cache, so a network failure while loading AST made the
classifier fall back to YamNet, the model the evaluation rejected, with only a
log warning to show for it. No model is downloaded or loaded here: transformers,
torch and TF-Hub are replaced by stand-ins.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules.audio_scene import AudioSceneClassifier  # noqa: E402

OFFLINE = OSError("[Errno 8] nodename nor servname provided, or not known")


def _fake_transformers(online_ok=False, local_ok=True):
    """A stand-in transformers module whose loaders record every call as
    (kind, local_files_only) and fail when the matching source is down."""
    calls = []
    model = MagicMock()
    model.config.id2label = {0: "Speech", 1: "Siren"}

    def loader(kind, result):
        def from_pretrained(model_id, local_files_only=False, **kwargs):
            calls.append((kind, local_files_only))
            if (local_files_only and not local_ok) or (not local_files_only and not online_ok):
                raise OFFLINE
            return result
        return from_pretrained

    fake = types.ModuleType("transformers")
    fake.AutoFeatureExtractor = types.SimpleNamespace(from_pretrained=loader("extractor", MagicMock()))
    fake.AutoModelForAudioClassification = types.SimpleNamespace(from_pretrained=loader("model", model))
    return fake, calls


class ASTLocalCacheFallbackTests(unittest.TestCase):
    """AST is retried from the local cache before any fallback, and the
    active backend is reported by status()."""

    def _load(self, fake_transformers, fake_hub=None):
        stand_ins = {"transformers": fake_transformers, "torch": MagicMock()}
        if fake_hub is not None:
            stand_ins.update({"tensorflow": MagicMock(), "tensorflow_hub": fake_hub})
        c = AudioSceneClassifier()
        # prepare_tfhub_cache would touch the real cache folder; stub it out.
        with patch.dict(sys.modules, stand_ins), patch("modules.model_cache.prepare_tfhub_cache"):
            c._ensure_loaded()
        return c

    def test_offline_load_retries_from_local_cache(self):
        fake, calls = _fake_transformers(online_ok=False, local_ok=True)
        hub = MagicMock()
        c = self._load(fake, hub)
        self.assertEqual(c._backend, "ast")
        self.assertEqual(c._labels, ["Speech", "Siren"])
        self.assertEqual(calls[0], ("extractor", False))
        self.assertIn(("extractor", True), calls)
        self.assertIn(("model", True), calls)
        hub.load.assert_not_called()            # YamNet was never tried
        status = c.status()
        self.assertFalse(status["degraded"])
        self.assertTrue(status["loaded_from_local_cache"])

    def test_online_load_needs_no_retry(self):
        fake, calls = _fake_transformers(online_ok=True)
        c = self._load(fake)
        self.assertEqual(c._backend, "ast")
        self.assertEqual(calls, [("extractor", False), ("model", False)])
        self.assertFalse(c.status()["loaded_from_local_cache"])

    def test_status_reports_degradation_when_ast_cannot_load(self):
        fake, _calls = _fake_transformers(online_ok=False, local_ok=False)
        hub = MagicMock()
        hub.load.side_effect = OSError("YamNet unavailable too")
        c = self._load(fake, hub)
        status = c.status()
        self.assertEqual(status["backend"], "heuristic")
        self.assertTrue(status["degraded"])
        self.assertIn("nodename", status["ast_load_error"])

    def test_status_before_loading_does_not_load(self):
        c = AudioSceneClassifier()
        status = c.status()
        self.assertEqual(status["backend"], "not_loaded")
        self.assertFalse(status["degraded"])
        self.assertIsNone(c._model)


if __name__ == "__main__":
    unittest.main()
