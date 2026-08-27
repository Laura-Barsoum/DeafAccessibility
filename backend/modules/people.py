"""
people.py — Few-shot person enrolment (face + voice + name).

Lets the Deaf user register the people who matter (family, flatmates,
manager) so the live pipeline can label captions and trigger an alert
when the user's own name is mentioned in speech.

Each person profile stores:
    - name (display name, also used for name-call detection in STT)
    - face_embedding: mean 4096-d DeepFace VGG-Face embedding over the
      enrolment frames
    - voice_embedding: mean mel-spectrogram embedding (128-d) — a simple
      but reasonably-discriminative voice fingerprint without pulling in
      a heavyweight speaker-id model
    - n_face_examples / n_voice_examples

We also persist ONE special key "_self" — the *user's* own name. When
that name appears in any speech transcript, the system emits a CRITICAL
event so the frontend can flash + vibrate.

Storage: JSON at backend/data/people/profile.json
Face thumbnails: backend/data/people/<name>.jpg (debug only)
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .events import Event, Priority

log = logging.getLogger("accessibility.people")

PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "people",
)
PROFILE_PATH = os.path.join(PROFILE_DIR, "profile.json")

# Cosine-similarity threshold for a face match. DeepFace's documented default
# for VGG-Face is a cosine DISTANCE of 0.68, i.e. a similarity of about 0.32.
# We sit slightly stricter than that to limit false identifications, but not so
# strict that a genuine match under different lighting is rejected. The actual
# similarity is logged on every attempt so this can be tuned against real data.
FACE_THRESHOLD = 0.35
FACE_THRESHOLD_ENV = "ACCESSIBILITY_FACE_THRESHOLD"
VOICE_THRESHOLD = 0.80  # cosine threshold for voice match (mel-spec)

# Default safety keywords that trigger a CRITICAL alert when spoken aloud.
# The user can edit this list via the /keywords endpoint.
DEFAULT_KEYWORDS = ["fire", "help", "emergency", "ambulance", "stop"]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip())


class PeopleRegistry:
    """Few-shot face + voice recogniser with persistent JSON storage."""

    def __init__(self) -> None:
        os.makedirs(PROFILE_DIR, exist_ok=True)
        self.profile: Dict[str, Any] = self._load()
        log.info(
            "PeopleRegistry initialised — %d people, self_name=%r",
            len([k for k in self.profile.get("people", {}) if k != "_self"]),
            self.profile.get("self_name"),
        )

    # ── Persistence ─────────────────────────────────────────────────
    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(PROFILE_PATH):
            return {"people": {}, "self_name": None,
                    "keywords": list(DEFAULT_KEYWORDS)}
        try:
            with open(PROFILE_PATH) as f:
                d = json.load(f)
                d.setdefault("people", {})
                d.setdefault("self_name", None)
                d.setdefault("keywords", list(DEFAULT_KEYWORDS))
                return d
        except Exception:
            return {"people": {}, "self_name": None,
                    "keywords": list(DEFAULT_KEYWORDS)}

    def _save(self) -> None:
        with open(PROFILE_PATH, "w") as f:
            json.dump(self.profile, f, indent=2)

    # ── Embedding extractors ────────────────────────────────────────
    def _face_embedding(self, frame_b64: str) -> Optional[np.ndarray]:
        """Single-frame face embedding via DeepFace.represent (VGG-Face).
        Returns None if no face detected."""
        try:
            from deepface import DeepFace
            import cv2  # type: ignore
            jpg = base64.b64decode(frame_b64)
            arr = np.frombuffer(jpg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            reps = DeepFace.represent(
                img_path=img, model_name="VGG-Face",
                enforce_detection=False, detector_backend="opencv",
            )
            if not reps:
                return None
            emb = np.array(reps[0]["embedding"], dtype=np.float32)
            return emb
        except Exception as e:
            log.debug("face embedding failed: %s", e)
            return None

    def _voice_embedding(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Mean log-mel spectrogram (128 bins) as a simple voice
        fingerprint. Discriminative enough for ~5-person home use
        without pulling a 300MB speaker-id model."""
        try:
            import librosa
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                path = tmp.name
            try:
                y, sr = librosa.load(path, sr=16000, mono=True)
            finally:
                try: os.unlink(path)
                except Exception: pass
            if len(y) < sr // 4:  # < 250 ms — too short
                return None
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512,
            )
            logmel = librosa.power_to_db(mel + 1e-10).mean(axis=1)
            # L2-normalise
            norm = np.linalg.norm(logmel) + 1e-9
            return (logmel / norm).astype(np.float32)
        except Exception as e:
            log.debug("voice embedding failed: %s", e)
            return None

    # ── Enrolment ───────────────────────────────────────────────────
    def enrol(
        self,
        name: str,
        frames_b64: Optional[List[str]] = None,
        audio_clips_bytes: Optional[List[bytes]] = None,
    ) -> Dict[str, Any]:
        """Enrol or update a person. At least one modality (face or voice)
        must contribute at least one embedding. Returns a result dict."""
        name = _normalize(name)
        if not name:
            return {"ok": False, "error": "name required"}
        if name.startswith("_"):
            return {"ok": False, "error": "reserved name"}

        # Face embeddings
        face_embs: List[np.ndarray] = []
        for f in (frames_b64 or []):
            emb = self._face_embedding(f)
            if emb is not None:
                face_embs.append(emb)

        # Voice embeddings
        voice_embs: List[np.ndarray] = []
        for clip in (audio_clips_bytes or []):
            emb = self._voice_embedding(clip)
            if emb is not None:
                voice_embs.append(emb)

        if not face_embs and not voice_embs:
            return {"ok": False, "error": "no usable face or voice samples extracted"}

        # Merge with any existing profile (allows incremental enrolment)
        existing = self.profile["people"].get(name, {})
        if face_embs:
            mean_face = np.mean(np.stack(face_embs), axis=0)
            existing["face_embedding"] = mean_face.tolist()
            existing["n_face_examples"] = (
                existing.get("n_face_examples", 0) + len(face_embs)
            )
        if voice_embs:
            mean_voice = np.mean(np.stack(voice_embs), axis=0)
            existing["voice_embedding"] = mean_voice.tolist()
            existing["n_voice_examples"] = (
                existing.get("n_voice_examples", 0) + len(voice_embs)
            )
        existing["name"] = name
        self.profile["people"][name] = existing
        self._save()

        return {
            "ok": True,
            "name": name,
            "n_face": len(face_embs),
            "n_voice": len(voice_embs),
            "total_face": existing.get("n_face_examples", 0),
            "total_voice": existing.get("n_voice_examples", 0),
        }

    def remove(self, name: str) -> bool:
        name = _normalize(name)
        if name in self.profile["people"]:
            del self.profile["people"][name]
            self._save()
            return True
        return False

    def list_all(self) -> List[Dict[str, Any]]:
        out = []
        for n, info in self.profile["people"].items():
            out.append({
                "name": n,
                "n_face_examples": info.get("n_face_examples", 0),
                "n_voice_examples": info.get("n_voice_examples", 0),
            })
        return out

    # ── Self-name (for name-call alert) ─────────────────────────────
    def set_self_name(self, name: str) -> None:
        self.profile["self_name"] = _normalize(name) or None
        self._save()

    def get_self_name(self) -> Optional[str]:
        return self.profile.get("self_name")

    # ── Inference ───────────────────────────────────────────────────
    def identify_face(self, frame_b64: str) -> Optional[Tuple[str, float]]:
        """Return (name, similarity) of the best face match, or None.

        Logs the best similarity even when it falls below the acceptance
        threshold, because a near-miss and a total failure to detect a face
        look identical from the interface but need different fixes.
        """
        emb = self._face_embedding(frame_b64)
        if emb is None:
            log.info("face id: no face embedding from frame (no face detected?)")
            return None
        best_name, best_sim = None, -1.0
        for n, info in self.profile["people"].items():
            ref = info.get("face_embedding")
            if not ref:
                continue
            ref_arr = np.array(ref, dtype=np.float32)
            if ref_arr.shape != emb.shape:
                log.warning("face id: shape mismatch for '%s' (%s vs %s), re-enrol needed",
                            n, emb.shape, ref_arr.shape)
                continue
            sim = _cosine(emb, ref_arr)
            if sim > best_sim:
                best_sim, best_name = sim, n
        if best_name is None:
            log.info("face id: no enrolled face embeddings to compare against")
            return None
        thr = float(os.environ.get(FACE_THRESHOLD_ENV, FACE_THRESHOLD))
        if best_sim >= thr:
            log.info("face id: MATCH '%s' (similarity=%.3f >= %.2f)",
                     best_name, best_sim, thr)
            return best_name, best_sim
        log.info("face id: best '%s' similarity=%.3f below threshold %.2f "
                 "(lower with %s if this is a genuine match)",
                 best_name, best_sim, thr, FACE_THRESHOLD_ENV)
        return None

    def identify_voice(self, audio_bytes: bytes) -> Optional[Tuple[str, float]]:
        emb = self._voice_embedding(audio_bytes)
        if emb is None:
            return None
        best_name, best_sim = None, -1.0
        for n, info in self.profile["people"].items():
            ref = info.get("voice_embedding")
            if not ref:
                continue
            sim = _cosine(emb, np.array(ref, dtype=np.float32))
            if sim > best_sim:
                best_sim, best_name = sim, n
        if best_name and best_sim >= VOICE_THRESHOLD:
            return best_name, best_sim
        return None

    # ── Keyword watch-list ──────────────────────────────────────────
    def set_keywords(self, words: List[str]) -> None:
        cleaned = []
        for w in words or []:
            w = _normalize(str(w)).lower()
            if w and w not in cleaned:
                cleaned.append(w)
        self.profile["keywords"] = cleaned
        self._save()

    def get_keywords(self) -> List[str]:
        return list(self.profile.get("keywords", DEFAULT_KEYWORDS))

    # ── Name-call + keyword detection ───────────────────────────────
    def detect_name_calls(self, transcript: str) -> Dict[str, Any]:
        """Scan a speech transcript for:
          - the user's own name → CRITICAL event
          - a watch-list keyword (fire, help, ...) → CRITICAL event
          - any enrolled person's name → INFORM event
        Returns {"self_called", "keywords_heard", "mentioned", "events"}."""
        empty = {"self_called": False, "keywords_heard": [],
                 "mentioned": [], "events": []}
        if not transcript:
            return empty
        text = transcript.lower()
        events: List[Event] = []
        mentioned: List[str] = []
        keywords_heard: List[str] = []
        self_called = False

        self_name = (self.profile.get("self_name") or "").strip().lower()
        if self_name and self._name_in_text(self_name, text):
            self_called = True
            events.append(Event(
                source="speech",
                label="Your name was called",
                priority=Priority.CRITICAL,
                confidence=0.95,
                text=transcript[:120],
                extra={"trigger": "self_name", "self_name": self_name},
            ))

        for kw in self.profile.get("keywords", DEFAULT_KEYWORDS):
            if kw and self._name_in_text(kw, text):
                keywords_heard.append(kw)
                events.append(Event(
                    source="speech",
                    label=f"Keyword: \"{kw}\"",
                    priority=Priority.CRITICAL,
                    confidence=0.9,
                    text=transcript[:120],
                    extra={"trigger": "keyword", "keyword": kw},
                ))

        for n in self.profile.get("people", {}):
            ln = n.lower().strip()
            if ln == self_name:
                continue
            if self._name_in_text(ln, text):
                mentioned.append(n)
                events.append(Event(
                    source="speech",
                    label=f"{n} was mentioned",
                    priority=Priority.INFORM,
                    confidence=0.85,
                    text=transcript[:120],
                    extra={"trigger": "person_mention", "person": n},
                ))

        return {"self_called": self_called, "keywords_heard": keywords_heard,
                "mentioned": mentioned, "events": events}

    @staticmethod
    def _name_in_text(needle: str, haystack: str) -> bool:
        """Whole-word match — avoid 'Sam' matching inside 'same'."""
        if not needle or not haystack:
            return False
        # First name only — Whisper often drops surnames.
        first = needle.split()[0]
        # Word-boundary match.
        return re.search(rf"\b{re.escape(first)}\b", haystack) is not None


# ── Singleton ───────────────────────────────────────────────────────
_registry_singleton: Optional[PeopleRegistry] = None


def get_people() -> PeopleRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = PeopleRegistry()
    return _registry_singleton
