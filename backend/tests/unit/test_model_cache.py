"""Unit tests for the persistent, self-healing TF-Hub cache.

Regression test for a real defect: an empty YamNet cache entry was treated as
a cache hit, failed to load, and silently forced the personaliser onto its
weaker fallback embedding.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules import model_cache as mc  # noqa: E402


class PrepareTFHubCacheTests(unittest.TestCase):
    """Incomplete entries are removed; complete and in-progress ones are kept."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="tfhub_cache_test_"))
        self._saved = os.environ.get("TFHUB_CACHE_DIR")
        os.environ["TFHUB_CACHE_DIR"] = str(self.root)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("TFHUB_CACHE_DIR", None)
        else:
            os.environ["TFHUB_CACHE_DIR"] = self._saved
        shutil.rmtree(self.root, ignore_errors=True)

    def test_empty_entry_is_removed(self):
        # The exact state found on disk: sub-folders present, no model files.
        broken = self.root / "9616fd04ec2360621642ef9455b84f4b668e219e"
        (broken / "variables").mkdir(parents=True)
        (broken / "assets").mkdir()
        mc.prepare_tfhub_cache()
        self.assertFalse(broken.exists())

    def test_complete_entry_is_kept(self):
        good = self.root / "abc123"
        good.mkdir()
        (good / "saved_model.pb").write_bytes(b"graph")
        mc.prepare_tfhub_cache()
        self.assertTrue((good / "saved_model.pb").exists())

    def test_in_progress_download_is_not_touched(self):
        tmp = self.root / "abc123.5f2e.tmp"
        tmp.mkdir()
        lock = self.root / "abc123.lock"
        lock.write_text("pid")
        mc.prepare_tfhub_cache()
        self.assertTrue(tmp.exists())
        self.assertTrue(lock.exists())

    def test_explicit_cache_dir_is_respected(self):
        self.assertEqual(mc.prepare_tfhub_cache(), str(self.root))
        self.assertEqual(os.environ["TFHUB_CACHE_DIR"], str(self.root))

    def test_default_location_is_persistent_not_temp(self):
        os.environ.pop("TFHUB_CACHE_DIR", None)
        self.assertIn(os.path.join("data", "model_cache", "tfhub"), str(mc.DEFAULT_TFHUB_CACHE))
        self.assertNotIn(tempfile.gettempdir(), str(mc.DEFAULT_TFHUB_CACHE))


if __name__ == "__main__":
    unittest.main()
