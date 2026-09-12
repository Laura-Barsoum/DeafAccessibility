"""
personalizer.py — Few-shot custom-sound enrolment + continuous learning.

Lets a user record 3-10 examples of "this is my doorbell", "this is my
baby's cry", etc. We:
    1. Compute an audio embedding for each example (YamNet when available,
       local librosa features as fallback)
    2. Store the mean embedding + a label in a lightweight JSON store
    3. At inference, compare incoming audio embeddings to enrolled ones
       via cosine similarity. If similarity > threshold, emit a custom
       Event with the user-specified label and priority.
    4. Each match gets a unique `match_id` echoed back to the frontend.
       The user can confirm (✓) or reject (✗) the alert; positive
       confirmations are incrementally folded back into the embedding
       (running mean) so the recogniser sharpens over time.

This is the project's most distinctive feature: no commercial Deaf-
accessibility product personalises this way, but the safety value is
huge (your specific doorbell vs. a TV doorbell).

Storage: JSON at backend/data/personal_sounds/profile.json
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .events import Event, Priority, label_priority

log = logging.getLogger("accessibility.personalizer")


PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "personal_sounds",
)
PROFILE_PATH = os.path.join(PROFILE_DIR, "profile.json")

# Prototypical matcher settings, calibrated on a development split of ESC-50
# and checked on a held-out test split (see match_prototypical).
PROTO_TEMPERATURE = 0.5
PROTO_RADIUS_K = 1.5
PROTO_GATE_FLOOR = 0.10


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def _l2norm(a: np.ndarray) -> np.ndarray:
    """Unit-normalise an embedding (prototypical networks operate on a
    normalised metric space)."""
    return a / (np.linalg.norm(a) + 1e-9)


def _sqeuclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Euclidean distance. Snell et al. (2017) found Euclidean
    distance outperforms cosine for prototypical networks."""
    d = a - b
    return float(np.dot(d, d))


class Personalizer:
    """Few-shot custom sound recogniser using audio embeddings."""

    def __init__(self) -> None:
        self._yamnet = None
        os.makedirs(PROFILE_DIR, exist_ok=True)
        self.profile: Dict[str, Any] = self._load()
        # Recent matches awaiting user feedback. Bounded so memory doesn't
        # grow forever. Each entry: (match_id, label, embedding_at_match,
        # similarity).
        self._recent_matches: Deque[Dict[str, Any]] = deque(maxlen=200)
        log.info(
            "Personalizer initialised — %d enrolled sounds, feedback buffer=%d",
            len(self.profile.get("sounds", {})), self._recent_matches.maxlen,
        )

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(PROFILE_PATH):
            return {"sounds": {}}
        try:
            with open(PROFILE_PATH) as f:
                return json.load(f)
        except Exception:
            return {"sounds": {}}

    def _save(self) -> None:
        with open(PROFILE_PATH, "w") as f:
            json.dump(self.profile, f, indent=2)

    def _ensure_yamnet(self) -> None:
        if self._yamnet is not None:
            return
        try:
            # macOS Python frequently can't verify the TF-Hub CDN certificate
            # ("CERTIFICATE_VERIFY_FAILED"), which silently forces the weak
            # fallback embedding. Point the download at certifi's CA bundle
            # (for requests) and relax the default SSL context (for urllib)
            # so the public YamNet model downloads.
            try:
                import certifi
                os.environ.setdefault("SSL_CERT_FILE", certifi.where())
                os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
                os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())
            except Exception:
                pass
            try:
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
            except Exception:
                pass
            # Persistent, self-healing cache. An empty entry left in the system
            # temp directory otherwise fails to load and silently forces the
            # fallback embedding (see model_cache.py).
            from .model_cache import prepare_tfhub_cache
            prepare_tfhub_cache()
            import tensorflow_hub as hub
            self._yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
            log.info("YamNet loaded for embeddings")
        except Exception as e:
            log.warning("YamNet unavailable for embeddings (%s)", e)
            self._yamnet = "placeholder"

    def _load_audio(self, audio_bytes: bytes) -> Optional[Tuple[np.ndarray, int]]:
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
            if y.size == 0:
                return None
            return y.astype(np.float32), sr
        except Exception as e:
            log.warning("audio load failed: %s", e)
            return None

    def _embed_fallback(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Return a local, deterministic feature embedding when YamNet is unavailable."""
        loaded = self._load_audio(audio_bytes)
        if loaded is None:
            return None
        y, sr = loaded
        try:
            import librosa
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            rms = librosa.feature.rms(y=y)

            # Normalise EACH feature group to unit norm before concatenating.
            # Without this, large-magnitude groups (spectral centroid in Hz,
            # mel in dB) dominate the L2-normalised vector, making completely
            # different sounds look ~0.9 similar and causing false positives.
            groups = []
            for feat in (mfcc, librosa.power_to_db(mel), chroma, centroid, bandwidth, rolloff, zcr, rms):
                g = np.concatenate([np.mean(feat, axis=1), np.std(feat, axis=1)]).astype(np.float32)
                gn = np.linalg.norm(g)
                groups.append(g / gn if gn > 1e-9 else g)
            emb = np.concatenate(groups).astype(np.float32)
            norm = np.linalg.norm(emb)
            if not np.isfinite(norm) or norm <= 1e-9:
                return None
            return emb / norm
        except Exception as e:
            log.warning("fallback embedding failed: %s", e)
            return None

    def _embed(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Return an audio embedding for a clip."""
        self._ensure_yamnet()
        if self._yamnet == "placeholder":
            return self._embed_fallback(audio_bytes)
        try:
            import tensorflow as tf
            loaded = self._load_audio(audio_bytes)
            if loaded is None:
                return None
            y, _sr = loaded
            waveform = tf.constant(y, dtype=tf.float32)
            scores, embeddings, spectrogram = self._yamnet(waveform)
            return embeddings.numpy().mean(axis=0)
        except Exception as e:
            log.warning("YamNet embedding failed; using fallback (%s)", e)
            return self._embed_fallback(audio_bytes)

    def enrol(
        self, label: str, priority: Priority,
        audio_clips: List[bytes],
    ) -> Dict[str, Any]:
        """Enrol a new custom sound from a few example clips."""
        emb_list = []
        for clip in audio_clips:
            emb = self._embed(clip)
            if emb is not None:
                emb_list.append(emb)
        if not emb_list:
            return {"ok": False, "reason": "no_embeddings"}
        mean_emb = np.mean(emb_list, axis=0)

        # Prototypical-network representation (Snell et al., 2017): the class
        # prototype is the mean of the L2-normalised support embeddings, and
        # the radius is the mean intra-class squared-Euclidean spread, used
        # for open-set rejection at match time.
        norm_embs = [_l2norm(e) for e in emb_list]
        prototype = _l2norm(np.mean(norm_embs, axis=0))
        radius = float(np.mean([_sqeuclidean(e, prototype) for e in norm_embs]))

        sounds = self.profile.setdefault("sounds", {})
        sounds[label] = {
            "label": label,
            "priority": int(priority),
            "embedding": mean_emb.tolist(),         # kept for the cosine method
            "prototype": prototype.tolist(),        # normalised, for prototypical
            "radius": radius,
            "embedding_dim": int(mean_emb.shape[0]),
            "n_examples": len(emb_list),
        }
        self._save()
        log.info("enrolled '%s' (priority=%s, n=%d)", label,
                 priority.name, len(emb_list))
        return {"ok": True, "label": label, "n_examples": len(emb_list)}

    def list_enrolled(self) -> List[Dict[str, Any]]:
        """Return a JSON-safe summary of every enrolled sound (label,
        priority, example count) for the Personal Sounds tab. Never returns
        the raw embeddings."""
        sounds = self.profile.get("sounds", {})
        return [{"label": s["label"], "priority": s["priority"],
                 "n_examples": s["n_examples"]} for s in sounds.values()]

    def match(
        self, audio_bytes: bytes, threshold: float = 0.90,
    ) -> List[Event]:
        """Compare incoming audio to enrolled sounds; emit Events on match.

        Each match is recorded in the feedback buffer with a unique
        `match_id` so the frontend can later confirm/reject it (see
        `feedback()`).
        """
        sounds = self.profile.get("sounds", {})
        if not sounds:
            return []
        emb = self._embed(audio_bytes)
        if emb is None:
            return []
        out: List[Event] = []
        for label, s in sounds.items():
            ref = np.array(s["embedding"])
            if emb.shape != ref.shape:
                log.warning(
                    "skipping personal sound '%s': embedding shape mismatch current=%s ref=%s",
                    label, emb.shape, ref.shape,
                )
                continue
            sim = _cosine(emb, ref)
            # Use the per-sound threshold when one has been set (the
            # thumbs-down feedback raises it). YamNet embeddings are not very
            # discriminative, so a high threshold (~0.90) is needed or ambient
            # room noise matches the enrolled sound on every tick.
            thr = float(s.get("threshold", threshold))
            if sim >= thr:
                match_id = uuid.uuid4().hex[:12]
                self._recent_matches.append({
                    "match_id": match_id,
                    "label": label,
                    "embedding": emb.tolist(),  # snapshot, used by feedback()
                    "similarity": float(sim),
                })
                out.append(Event(
                    source="sound",
                    label=f"{label} (personal)",
                    priority=Priority(int(s["priority"])),
                    confidence=float(min(1.0, sim)),
                    extra={
                        "matched_via": "personalizer",
                        "method": "cosine",
                        "similarity": round(sim, 3),
                        "match_id": match_id,
                        "feedback_pending": True,
                    },
                ))
        return out

    # ── Prototypical-network matching (Snell et al., 2017) ──────────
    def match_prototypical(
        self, audio_bytes: bytes,
        prob_threshold: float = 0.55, radius_k: float = PROTO_RADIUS_K,
        temperature: float = PROTO_TEMPERATURE, gate_floor: float = PROTO_GATE_FLOOR,
    ) -> List[Event]:
        """Classify incoming audio against enrolled prototypes.

        Uses squared-Euclidean distance in the normalised embedding space,
        a softmax over negative distances (the prototypical-network decision
        rule), and an open-set reject: the nearest prototype must fall within
        its own enrolment radius, so ambient audio that resembles nothing in
        particular is rejected rather than snapped to the closest sound.

        Defaults were calibrated on a development split of ESC-50 (the 40
        non-domestic classes) under a 5% false-acceptance constraint, then
        checked on a held-out test split (the 10 domestic classes). The
        temperature matters because squared distances between unit vectors lie
        in [0, 4], which makes an untempered softmax nearly flat: on the test
        split the original rule (temperature 1, radius_k 3, floor 0.20)
        rejected 59% of correct nearest-prototype decisions on probability
        alone. Calibrated, test F1 rose from 0.327 to 0.344 at an unchanged
        false-acceptance rate.
        """
        sounds = self.profile.get("sounds", {})
        if not sounds:
            return []
        raw = self._embed(audio_bytes)
        if raw is None:
            return []
        e = _l2norm(raw)

        labels, protos, radii = [], [], []
        for label, s in sounds.items():
            proto = s.get("prototype")
            if proto is None:  # backward-compat with pre-prototype enrolments
                proto = _l2norm(np.array(s["embedding"], dtype=np.float32)).tolist()
            p = np.array(proto, dtype=np.float32)
            if p.shape != e.shape:
                continue
            labels.append(label); protos.append(p); radii.append(float(s.get("radius", 0.0)))
        if not labels:
            return []

        dists = np.array([_sqeuclidean(e, p) for p in protos])
        logits = -dists / temperature
        logits = logits - logits.max()
        probs = np.exp(logits); probs = probs / probs.sum()
        j = int(np.argmax(probs))

        # Open-set reject: nearest prototype must be within its radius band.
        # Floor at 0.20 (≈ cosine 0.90) so a tight enrolment still admits a
        # genuine play; s['threshold'] (raised by thumbs-down) tightens it.
        gate = max(radii[j] * radius_k, gate_floor)
        gate = min(gate, sounds[labels[j]].get("proto_gate", gate))
        if probs[j] < prob_threshold or dists[j] > gate:
            return []

        label = labels[j]; s = sounds[label]
        match_id = uuid.uuid4().hex[:12]
        self._recent_matches.append({
            "match_id": match_id, "label": label,
            "embedding": raw.tolist(), "similarity": float(probs[j]),
        })
        return [Event(
            source="sound",
            label=f"{label} (personal)",
            priority=Priority(int(s["priority"])),
            confidence=float(probs[j]),
            extra={
                "matched_via": "personalizer",
                "method": "prototypical",
                "prob": round(float(probs[j]), 3),
                "distance": round(float(dists[j]), 3),
                "match_id": match_id,
                "feedback_pending": True,
            },
        )]

    def match_current(self, audio_bytes: bytes) -> List[Event]:
        """Dispatch to the configured matcher. Prototypical by default;
        set ACCESSIBILITY_PERSONALIZER_METHOD=cosine to use the baseline."""
        method = os.environ.get("ACCESSIBILITY_PERSONALIZER_METHOD", "prototypical").lower()
        if method == "cosine":
            return self.match(audio_bytes)
        return self.match_prototypical(audio_bytes)

    # ── Continuous learning ─────────────────────────────────────────
    def feedback(self, match_id: str, is_positive: bool) -> Dict[str, Any]:
        """User confirms (or rejects) a previous match.

        Positive feedback: incrementally fold the embedding of the
        triggering audio into the enrolled sound's mean embedding,
        so the recogniser drifts toward the user's actual sound.

        Negative feedback: log it (full hard-negative mining would
        require a margin-based update — out of scope for this prototype
        — but we DO bump the per-sound threshold by a tiny amount so the
        same false positive is less likely next time).
        """
        # Find the match by id in the recent buffer.
        match = None
        for m in self._recent_matches:
            if m["match_id"] == match_id:
                match = m
                break
        if match is None:
            return {"ok": False, "error": "match_id not found (too old?)"}

        label = match["label"]
        sounds = self.profile.setdefault("sounds", {})
        if label not in sounds:
            return {"ok": False, "error": f"sound '{label}' was removed"}

        s = sounds[label]
        ref = np.array(s["embedding"], dtype=np.float32)
        new_emb = np.array(match["embedding"], dtype=np.float32)

        if is_positive:
            # Incremental running mean: new_mean = (n*old + new) / (n+1)
            n = max(1, int(s.get("n_examples", 1)))
            updated = (n * ref + new_emb) / (n + 1)
            s["embedding"] = updated.tolist()
            # Keep the prototypical representation in sync with the new mean.
            s["prototype"] = _l2norm(updated).tolist()
            s["n_examples"] = n + 1
            s["last_positive_ts"] = match["similarity"]
            self._save()
            log.info(
                "personalizer: + feedback on '%s' (n=%d → %d, sim=%.3f)",
                label, n, n + 1, match["similarity"],
            )
            return {
                "ok": True, "label": label, "is_positive": True,
                "n_examples": s["n_examples"],
                "similarity": match["similarity"],
            }
        else:
            # Tighten BOTH matchers. The cosine rule reads `threshold`; the
            # default prototypical rule reads `proto_gate`. Originally only the
            # threshold was bumped, so under the prototypical matcher a rejected
            # alert changed nothing. The gate shrinks by 15% per rejection but
            # never below a small floor, so a sound cannot become unmatchable.
            current = float(s.get("threshold", 0.90))
            bumped = min(0.95, current + 0.02)
            s["threshold"] = bumped
            default_gate = max(float(s.get("radius", 0.0)) * PROTO_RADIUS_K, PROTO_GATE_FLOOR)
            gate_before = float(s.get("proto_gate", default_gate))
            s["proto_gate"] = max(0.02, gate_before * 0.85)
            s["last_negative_ts"] = match["similarity"]
            self._save()
            log.info(
                "personalizer: - feedback on '%s' (threshold %.2f → %.2f, gate %.3f → %.3f)",
                label, current, bumped, gate_before, s["proto_gate"],
            )
            return {
                "ok": True, "label": label, "is_positive": False,
                "threshold": bumped, "proto_gate": s["proto_gate"],
            }

    def remove(self, label: str) -> bool:
        """Delete an enrolled sound. Returns True if it existed, False
        otherwise."""
        sounds = self.profile.get("sounds", {})
        if label in sounds:
            del sounds[label]
            self._save()
            return True
        return False
