"""Integration tests — exercise the Flask routes via the test client.

These verify the full request → handler → response shape for every new
endpoint added during the feature build, plus the critical happy path
of /process under graceful-degradation conditions (no audio / no
frames).

We do NOT run heavy models here. The /process tests pass tiny dummy
inputs and just assert the endpoint returns a well-shaped JSON dict
without crashing.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from base64 import b64encode
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_TMP = None
_SAVED = {}


def setUpModule():
    """Redirect every file these endpoints write into a temp directory.

    Without this the suite wrote to the user's real data: the round-trip test
    for /me/name overwrote the saved self-name (found as "TestLaura"), and the
    speaker tests touched the real speakers.json.
    """
    global _TMP
    _TMP = tempfile.mkdtemp(prefix="endpoint_tests_")
    import server
    from modules import diarized_stt as dz
    from modules import people as people_mod
    _SAVED.update(
        people_dir=people_mod.PROFILE_DIR, people_path=people_mod.PROFILE_PATH,
        people_singleton=people_mod._registry_singleton,
        speakers_path=dz._SPEAKERS_PATH, diar_singleton=dz._diarized_stt_singleton,
        db_env=os.environ.get("ACCESSIBILITY_DB_PATH"), diary=server._diary,
    )
    people_mod.PROFILE_DIR = os.path.join(_TMP, "people")
    people_mod.PROFILE_PATH = os.path.join(_TMP, "people", "profile.json")
    people_mod._registry_singleton = None
    dz._SPEAKERS_PATH = Path(_TMP) / "speakers.json"
    dz._diarized_stt_singleton = None
    os.environ["ACCESSIBILITY_DB_PATH"] = os.path.join(_TMP, "diary.db")
    server._diary = None


def tearDownModule():
    import server
    from modules import diarized_stt as dz
    from modules import people as people_mod
    people_mod.PROFILE_DIR = _SAVED["people_dir"]
    people_mod.PROFILE_PATH = _SAVED["people_path"]
    people_mod._registry_singleton = _SAVED["people_singleton"]
    dz._SPEAKERS_PATH = _SAVED["speakers_path"]
    dz._diarized_stt_singleton = _SAVED["diar_singleton"]
    if _SAVED["db_env"] is None:
        os.environ.pop("ACCESSIBILITY_DB_PATH", None)
    else:
        os.environ["ACCESSIBILITY_DB_PATH"] = _SAVED["db_env"]
    server._diary = _SAVED["diary"]
    shutil.rmtree(_TMP, ignore_errors=True)


class EndpointShapeTests(unittest.TestCase):
    """Verify every new endpoint returns the right JSON shape."""

    @classmethod
    def setUpClass(cls):
        # Import server lazily so the test runner can collect this module
        # even when heavy deps (torch, mediapipe) aren't loaded yet.
        from server import app
        cls.app = app
        cls.client = app.test_client()

    # ── /speakers ────────────────────────────────────────────────────
    def test_speakers_list_empty(self):
        r = self.client.get("/speakers")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("mappings", data)
        self.assertIn("seen_unnamed", data)

    def test_speakers_name_round_trip(self):
        r = self.client.post(
            "/speakers/name",
            data=json.dumps({"raw_id": "SPEAKER_42", "name": "TestAlice"}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])
        # GET should now show the mapping
        r2 = self.client.get("/speakers")
        self.assertEqual(r2.get_json()["mappings"]["SPEAKER_42"], "TestAlice")
        # DELETE
        r3 = self.client.delete("/speakers/SPEAKER_42")
        self.assertTrue(r3.get_json()["ok"])

    def test_speakers_name_missing_field_returns_400(self):
        r = self.client.post(
            "/speakers/name",
            data=json.dumps({"raw_id": "SPEAKER_00"}),  # no name
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    # ── /people ──────────────────────────────────────────────────────
    def test_people_list_empty(self):
        r = self.client.get("/people")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("people", data)
        self.assertIn("self_name", data)

    def test_me_name_round_trip(self):
        r = self.client.post(
            "/me/name",
            data=json.dumps({"name": "TestLaura"}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["self_name"], "TestLaura")
        # Confirm it surfaces in GET /people
        r2 = self.client.get("/people")
        self.assertEqual(r2.get_json()["self_name"], "TestLaura")

    def test_people_enrol_empty_payload_returns_400(self):
        r = self.client.post(
            "/people/enrol",
            data=json.dumps({}),  # no name, no samples
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    # ── /enrol personal sounds ─────────────────────────────────────────
    def test_personal_sound_enrol_converts_browser_audio_before_embedding(self):
        class StubPersonalizer:
            """A stub personalizer that records what the /enrol route passes it."""
            def __init__(self):
                self.received = None

            def enrol(self, label, priority, clips):
                self.received = (label, priority, clips)
                return {"ok": True, "label": label, "n_examples": len(clips)}

        stub = StubPersonalizer()

        with patch("server.get_personal", return_value=stub), patch(
            "server._ffmpeg_convert", side_effect=lambda raw: b"wav:" + raw
        ):
            r = self.client.post(
                "/enrol",
                data=json.dumps({
                    "label": "doorbell",
                    "priority": 2,
                    "clips_b64": [b64encode(b"webm-audio").decode()],
                }),
                content_type="application/json",
            )

        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])
        self.assertEqual(stub.received[0], "doorbell")
        self.assertEqual(stub.received[2], [b"wav:webm-audio"])

    def test_personal_sound_enrol_reports_audio_conversion_failure(self):
        with patch("server._ffmpeg_convert", return_value=b""):
            r = self.client.post(
                "/enrol",
                data=json.dumps({
                    "label": "doorbell",
                    "priority": 2,
                    "clips_b64": [b64encode(b"bad-audio").decode()],
                }),
                content_type="application/json",
            )

        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["reason"], "audio_conversion_failed")

    # ── /enrolled/feedback ───────────────────────────────────────────
    def test_feedback_missing_match_id_returns_400(self):
        r = self.client.post(
            "/enrolled/feedback",
            data=json.dumps({"is_positive": True}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    def test_feedback_unknown_match_id_returns_ok_false(self):
        r = self.client.post(
            "/enrolled/feedback",
            data=json.dumps({"match_id": "doesnt_exist_12345", "is_positive": True}),
            content_type="application/json",
        )
        # 200 with ok:false (it's a legitimate query, just no such match)
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.get_json()["ok"])

    # ── /tts/status ──────────────────────────────────────────────────
    def test_tts_status_shape(self):
        r = self.client.get("/tts/status")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("elevenlabs_configured", data)
        self.assertIn("voice_id", data)


class ProcessGracefulDegradationTests(unittest.TestCase):
    """The /process endpoint must NOT crash on missing inputs."""

    @classmethod
    def setUpClass(cls):
        from server import app
        cls.client = app.test_client()

    def test_empty_payload_does_not_crash(self):
        """No audio, no frames — graceful empty response."""
        r = self.client.post(
            "/process",
            data=json.dumps({}),
            content_type="application/json",
        )
        # We only require 2xx; the model results will all be empty.
        self.assertIn(r.status_code, (200, 204))
        if r.status_code == 200 and r.is_json:
            data = r.get_json()
            self.assertIn("events", data)

    def test_audio_only_payload(self):
        """1 KB silent audio, no frames — pipeline still composes a response."""
        import base64
        silent = bytes(32000)  # ~1 s of 16-bit @ 16 kHz silence
        r = self.client.post(
            "/process",
            data=json.dumps({"audio_b64": base64.b64encode(silent).decode()}),
            content_type="application/json",
        )
        self.assertIn(r.status_code, (200, 204))


if __name__ == "__main__":
    unittest.main()
