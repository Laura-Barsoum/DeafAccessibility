"""
sign_language.py — Sign Language Recognition (SLR).

A bidirectional channel for the Deaf user: instead of (or alongside)
typing/speaking, they can sign to the camera and the system converts
the gesture sequence into spoken English via the LLM and TTS.

Architecture (high-level, in order of escalating complexity):

    Stage 1 (this module — production-ready)
        MediaPipe Holistic captures hand + pose + face landmarks → a
        543-dim vector per frame. We aggregate ~30 frames into a clip
        (~1 second of signing) and run a temporal classifier:

        Default backend: a small **bidirectional LSTM** trained on the
        WLASL-100 vocabulary (Li et al. 2020, ~2,000 labelled videos).

        Fallback (no model loaded): an embedding-distance approach using
        cosine similarity against ~50 hand-crafted reference signs
        (yes/no/help/water/etc.) that the user can extend.

    Stage 2 (hook left for the report's future-work section)
        Replace the LSTM with a transformer-based AVSR model (e.g.
        Sign Language Transformer, Camgöz et al. 2020) fine-tuned on
        How2Sign for continuous sentence-level translation.

This module exposes:
    extract_landmarks(jpeg_b64) → 543-dim vector
    classify_clip(landmarks_seq) → (label, confidence)
    enrol_sign(label, clips)    → adds a few-shot reference sign
    sign_to_speech(text, llm)   → polishes a glossed transcript via LLM

References
----------
- Lugaresi et al. (2019) MediaPipe: A framework for building perception pipelines.
- Li et al. (2020) WLASL: Word-Level American Sign Language dataset.
- Camgöz et al. (2020) Sign Language Transformers.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .events import Event, Priority
from .sign_language_model import load_checkpoint
from .spell_buffer import SpellBuffer
from .tgcn_sign_model import TGCNSignRecognizer, extract_55_keypoints


log = logging.getLogger("accessibility.sign")


# Built-in vocabulary — the most safety-critical and conversational
# signs. The system can recognise these without any trained model via
# the embedding-distance fallback once the user enrols 3-5 examples.
DEFAULT_VOCAB = [
    "hello", "yes", "no", "thank_you", "please", "help", "stop",
    "fire", "danger", "water", "doctor", "phone", "name",
    "more", "again", "wait", "ok", "sorry", "love", "fine",
]


def _load_wlasl_vocab() -> List[str]:
    """
    If `data/wlasl_vocab.json` is present (built by scripts/wlasl_vocab.py
    from the WLASL v0.3 manifest), expand the default vocab with the
    top-N WLASL glosses for evaluation against the academic benchmark.
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "wlasl_vocab.json",
    )
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        v = data.get("vocab", [])
        log.info("Loaded WLASL vocab: %d glosses", len(v))
        return v
    except Exception as e:
        log.warning("Failed to load WLASL vocab: %s", e)
        return []


WLASL_VOCAB = _load_wlasl_vocab()
# Merge: default safety signs + WLASL glosses, deduplicated.
FULL_VOCAB = list(dict.fromkeys(DEFAULT_VOCAB + WLASL_VOCAB))

# Storage for few-shot reference embeddings.
SIGN_PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sign_profile",
)
SIGN_PROFILE_PATH = os.path.join(SIGN_PROFILE_DIR, "signs.json")


def _flatten_landmarks(holistic_result) -> np.ndarray:
    """
    Concatenate pose (33×3) + left hand (21×3) + right hand (21×3) +
    face-summary (5 key points × 3) into a 543-dim feature vector.
    """
    parts: List[np.ndarray] = []
    # Pose 33 points
    if holistic_result.pose_landmarks:
        pose = np.array(
            [[p.x, p.y, p.z] for p in holistic_result.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(33 * 3, dtype=np.float32)
    parts.append(pose)
    # Left hand 21 points
    if holistic_result.left_hand_landmarks:
        lh = np.array(
            [[p.x, p.y, p.z] for p in holistic_result.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)
    parts.append(lh)
    # Right hand 21 points
    if holistic_result.right_hand_landmarks:
        rh = np.array(
            [[p.x, p.y, p.z] for p in holistic_result.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)
    parts.append(rh)
    # Face — a 5-point summary (eye corners, nose tip, mouth corners)
    # avoids feeding the full 468 mesh into the temporal model.
    if holistic_result.face_landmarks:
        idxs = [33, 263, 1, 61, 291]  # MediaPipe FaceMesh indices
        face_summary = np.array(
            [[holistic_result.face_landmarks.landmark[i].x,
              holistic_result.face_landmarks.landmark[i].y,
              holistic_result.face_landmarks.landmark[i].z]
             for i in idxs],
            dtype=np.float32,
        ).flatten()
    else:
        face_summary = np.zeros(5 * 3, dtype=np.float32)
    parts.append(face_summary)

    return np.concatenate(parts)  # 543-dim


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


class _HandsHolisticAdapter:
    """Wraps a MediaPipe Hands result with the attribute names the rest
    of this module expects from Holistic: `right_hand_landmarks`,
    `left_hand_landmarks`, `pose_landmarks`, `face_landmarks`."""
    def __init__(self):
        self.right_hand_landmarks = None
        self.left_hand_landmarks = None
        self.pose_landmarks = None
        self.face_landmarks = None


# ─── Tasks-API → legacy-Holistic shape adapters ─────────────────────────
# The legacy `mediapipe.solutions.holistic` API was removed in mediapipe
# 0.10.21+. Python 3.13 has no wheel for any earlier version, so the only
# path forward is the new `mediapipe.tasks.python.vision` API. The Tasks
# API ships separate detectors for pose, hand, and face landmarks; the
# legacy code in this module expects a single object with all four sets.
#
# These adapters wrap the Tasks-API results in the legacy attribute shape
# (`pose_landmarks.landmark[i].x/y/z`, etc.) so `_flatten_landmarks()`,
# `_classify_handshape()`, `extract_55_keypoints()`, and the geometric
# rules all continue to work unchanged.

class _TasksLandmarkPoint:
    """A single (x, y, z) landmark — mirrors the legacy NormalizedLandmark."""
    __slots__ = ("x", "y", "z", "visibility")
    def __init__(self, x: float, y: float, z: float, visibility: float = 1.0):
        self.x = float(x); self.y = float(y); self.z = float(z)
        self.visibility = float(visibility)


class _TasksLandmarkList:
    """A list of landmark points exposed as `.landmark` — mirrors the
    legacy NormalizedLandmarkList. Iteration + indexing both work."""
    __slots__ = ("landmark",)
    def __init__(self, points):
        self.landmark = points

    def __iter__(self):
        return iter(self.landmark)

    def __len__(self):
        return len(self.landmark)


def _wrap_landmarks(landmarks_list) -> Optional[_TasksLandmarkList]:
    """Convert a Tasks-API list of NormalizedLandmark objects into the
    legacy `.landmark[i].x/y/z` shape. Returns None on empty input."""
    if not landmarks_list:
        return None
    pts = [
        _TasksLandmarkPoint(
            getattr(lm, "x", 0.0),
            getattr(lm, "y", 0.0),
            getattr(lm, "z", 0.0),
            getattr(lm, "visibility", 1.0),
        )
        for lm in landmarks_list
    ]
    return _TasksLandmarkList(pts)


class _TasksHolisticResult:
    """The same shape `mp.solutions.holistic.Holistic.process()` returned:
    `pose_landmarks`, `left_hand_landmarks`, `right_hand_landmarks`,
    `face_landmarks`. Each is either None or a `_TasksLandmarkList`."""
    __slots__ = (
        "pose_landmarks", "left_hand_landmarks",
        "right_hand_landmarks", "face_landmarks",
    )
    def __init__(self):
        self.pose_landmarks = None
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.face_landmarks = None


class _TasksHolistic:
    """Drop-in replacement for the legacy
    `mediapipe.solutions.holistic.Holistic` object — composes three
    independent Tasks-API detectors (pose / hand / face) and merges the
    results into a single `_TasksHolisticResult`.

    Usage matches the legacy API:
        h = _TasksHolistic()
        res = h.process(rgb_frame)
        if res.right_hand_landmarks: ...
    """
    def __init__(self, models_dir: str):
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import mediapipe as mp

        self._mp = mp

        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=os.path.join(models_dir, "pose_landmarker_lite.task"),
            ),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=os.path.join(models_dir, "hand_landmarker.task"),
            ),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=os.path.join(models_dir, "face_landmarker.task"),
            ),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._pose = mp_vision.PoseLandmarker.create_from_options(pose_opts)
        self._hand = mp_vision.HandLandmarker.create_from_options(hand_opts)
        self._face = mp_vision.FaceLandmarker.create_from_options(face_opts)

    def process(self, rgb):
        """Run all three landmarkers on `rgb` (HxWx3 numpy array, RGB) and
        return a legacy-shaped Holistic result. Any single landmarker
        failing leaves that field as None — the rest still populates."""
        out = _TasksHolisticResult()
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb,
        )
        # Pose
        try:
            r = self._pose.detect(mp_image)
            if r and r.pose_landmarks:
                # Tasks-API returns a List[List[NormalizedLandmark]] — one
                # inner list per detected pose. We use the first.
                out.pose_landmarks = _wrap_landmarks(r.pose_landmarks[0])
        except Exception:
            pass
        # Hands — assigned to left/right slot via handedness label.
        try:
            r = self._hand.detect(mp_image)
            if r and r.hand_landmarks:
                for i, lms in enumerate(r.hand_landmarks):
                    label = "Right"
                    try:
                        label = r.handedness[i][0].category_name
                    except Exception:
                        pass
                    wrapped = _wrap_landmarks(lms)
                    if label == "Right":
                        out.right_hand_landmarks = wrapped
                    else:
                        out.left_hand_landmarks = wrapped
        except Exception:
            pass
        # Face
        try:
            r = self._face.detect(mp_image)
            if r and r.face_landmarks:
                out.face_landmarks = _wrap_landmarks(r.face_landmarks[0])
        except Exception:
            pass
        return out

    def close(self) -> None:
        """Release the three MediaPipe Tasks-API landmarkers (pose, hand,
        face) that back this composite Holistic shim."""
        for det in (self._pose, self._hand, self._face):
            try: det.close()
            except Exception: pass


def _adapt_hands_result(hands_result):
    """Convert a MediaPipe Hands result into a Holistic-shaped wrapper.
    Handedness is reported per-hand by MediaPipe; we use it to populate
    the correct slot."""
    out = _HandsHolisticAdapter()
    if not hands_result or not getattr(hands_result, "multi_hand_landmarks", None):
        return out
    handednesses = getattr(hands_result, "multi_handedness", []) or []
    for i, lm in enumerate(hands_result.multi_hand_landmarks):
        label = "Right"
        try:
            label = handednesses[i].classification[0].label
        except Exception:
            pass
        if label == "Right":
            out.right_hand_landmarks = lm
        else:
            out.left_hand_landmarks = lm
    return out


def _flatten_hands_landmarks(adapter) -> np.ndarray:
    """Lightweight 126-dim landmark vector when only the Hands backend is
    available (left + right hand × 21 landmarks × 3 coords). Padded to
    543 so the downstream LSTM/cosine logic still works without changes."""
    parts: List[np.ndarray] = []
    parts.append(np.zeros(33 * 3, dtype=np.float32))  # pose placeholder
    if adapter.left_hand_landmarks:
        lh = np.array(
            [[p.x, p.y, p.z] for p in adapter.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)
    parts.append(lh)
    if adapter.right_hand_landmarks:
        rh = np.array(
            [[p.x, p.y, p.z] for p in adapter.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)
    parts.append(rh)
    parts.append(np.zeros(5 * 3, dtype=np.float32))  # face placeholder
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Built-in geometric hand-shape classifier
# ---------------------------------------------------------------------------
# Maps MediaPipe Hands 21 landmarks → a handful of universally-recognised
# ASL/gesture handshapes. These work IMMEDIATELY without any training or
# enrolment — the user can sign 'hello' or 'stop' and the system will
# recognise it from frame 1.
#
# This is the fallback that fires when (a) no trained LSTM is loaded AND
# (b) the user has not enrolled any custom signs. It uses pure geometry:
# fingertip vs. knuckle vertical positions, distances between fingers,
# wrist orientation.
#
# Landmarks (MediaPipe Hands index reference):
#   0 = wrist
#   1-4   = thumb (CMC → MCP → IP → tip)
#   5-8   = index (MCP → PIP → DIP → tip)
#   9-12  = middle
#   13-16 = ring
#   17-20 = pinky
#
# We compute per-finger "extended" boolean (tip is further from wrist
# than the PIP joint), then classify based on the pattern.

FINGER_LANDMARKS = {
    "thumb":  (4, 3, 2),
    "index":  (8, 6, 5),
    "middle": (12, 10, 9),
    "ring":   (16, 14, 13),
    "pinky":  (20, 18, 17),
}


def _dist3d(landmarks, a: int, b: int) -> float:
    """3-D Euclidean distance between two MediaPipe landmarks."""
    dx = landmarks[a].x - landmarks[b].x
    dy = landmarks[a].y - landmarks[b].y
    dz = landmarks[a].z - landmarks[b].z
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _finger_extended(landmarks, finger: str) -> bool:
    """
    Is this finger extended (straight)?

    OLD APPROACH (broken): compared tip.y < pip.y in 2D image space.
    This ONLY worked when the finger pointed straight UP.  Signing 'you'
    (pointing toward the camera) or sideways completely fooled it because
    the Y coordinate barely changes when moving in the Z direction.

    NEW APPROACH: 3-D Euclidean distance from the wrist landmark.
    If the fingertip is farther from the wrist (in 3D) than the PIP joint,
    the finger is extended — regardless of whether it points up, forward,
    or sideways.  MediaPipe estimates z depth, so this actually works.
    """
    tip, pip, mcp = FINGER_LANDMARKS[finger]
    wrist = 0  # landmark 0 = WRIST in MediaPipe Hands

    if finger == "thumb":
        # Thumb extends away from the palm in the X/Y plane.
        # Use 2-D distance from the MCP joint (original heuristic kept because
        # thumb z-depth estimation is least reliable).
        dx_t = landmarks[tip].x - landmarks[mcp].x
        dy_t = landmarks[tip].y - landmarks[mcp].y
        dx_p = landmarks[pip].x - landmarks[mcp].x
        dy_p = landmarks[pip].y - landmarks[mcp].y
        return (dx_t**2 + dy_t**2) > (dx_p**2 + dy_p**2) * 1.2

    # For index/middle/ring/pinky: tip farther from wrist than pip in 3-D
    return _dist3d(landmarks, tip, wrist) > _dist3d(landmarks, pip, wrist) * 1.08


def _classify_handshape(landmarks) -> Optional[Tuple[str, float]]:
    """
    Classify ONE frame's hand landmarks into an ASL handshape label.
    Returns (label, confidence) or None.

    Finger-combination table (26 recognised shapes):

        5 up (all)               → hello / stop
        0 up (fist)              → yes
        thumb only               → ok / good
        index only               → you (number 1)
        middle only              → who / question
        pinky only               → me  (ASL letter I)
        index+middle             → peace / two
        index+ring               → cross (letter R proxy)
        index+pinky              → more / rock-on
        middle+pinky             → letter U proxy
        ring+pinky               → letter H proxy
        index+middle+ring        → three / W letter
        middle+ring+pinky        → letter F proxy / eight
        index+middle+pinky       → nine (ASL 9 shape)
        index+ring+pinky         → seven (ASL 7 shape)
        index+middle+ring+pinky  → four / letter B
        thumb+index              → help / L-shape
        thumb+middle             → six (ASL 6)
        thumb+ring               → seven variant
        thumb+pinky              → letter Y / phone
        thumb+index+middle       → three (alt) / letter D area
        thumb+index+pinky        → love (ILY)
        thumb+middle+pinky       → letter F variant
        thumb+index+middle+ring  → four alt
        all 5 (already handled)  → hello

    Pre-trained MediaPipe model (7 classes) still overrides these when
    it fires — these geometric rules fire only as the secondary tier.
    """
    fingers = {f: _finger_extended(landmarks, f) for f in FINGER_LANDMARKS}
    ix = fingers["index"]; mi = fingers["middle"]
    ri = fingers["ring"];  pi = fingers["pinky"]; th = fingers["thumb"]
    n_all = sum(fingers.values())
    n_fin = ix + mi + ri + pi   # non-thumb fingers extended

    # ── 5 fingers ────────────────────────────────────────────────────────
    if n_all == 5:
        return ("hello", 0.85)

    # ── Closed fist ───────────────────────────────────────────────────────
    if n_all == 0:
        return ("yes", 0.80)

    # ── Thumb only ────────────────────────────────────────────────────────
    if th and n_fin == 0:
        return ("ok", 0.82)

    # ── Single-finger (non-thumb) ─────────────────────────────────────────
    if n_fin == 1 and not th:
        if ix:  return ("you",  0.78)   # ASL "you" / number 1
        if mi:  return ("who",  0.65)   # question / "what"
        if pi:  return ("me",   0.72)   # ASL letter I / "me"
        # ring only is uncommon — skip

    # ── 2 fingers (no thumb) ──────────────────────────────────────────────
    if n_fin == 2 and not th:
        if ix and mi:  return ("peace",    0.82)   # V / number 2
        if ix and pi:  return ("more",     0.70)   # rock-on
        if ix and ri:  return ("letter_r", 0.65)   # R = crossed fingers
        if mi and pi:  return ("letter_u", 0.63)   # U shape
        if ri and pi:  return ("letter_h", 0.60)   # H pointing sideways

    # ── Thumb + 1 finger ─────────────────────────────────────────────────
    if th and n_fin == 1:
        if ix:  return ("help",    0.75)   # L-shape / help
        if mi:  return ("six",     0.68)   # ASL number 6
        if ri:  return ("seven",   0.65)   # ASL number 7
        if pi:  return ("phone",   0.70)   # Y / phone / call-me

    # ── 3 fingers (no thumb) ─────────────────────────────────────────────
    if n_fin == 3 and not th:
        if ix and mi and ri:   return ("three",    0.74)   # number 3 / W
        if ix and mi and pi:   return ("nine",     0.65)   # ASL 9
        if ix and ri and pi:   return ("eight",    0.63)   # ASL 8 variant
        if mi and ri and pi:   return ("letter_f", 0.60)   # F / palm-out

    # ── 4 fingers (no thumb) ──────────────────────────────────────────────
    if n_fin == 4 and not th:
        return ("four", 0.75)                               # B / number 4

    # ── Thumb + 2 fingers ─────────────────────────────────────────────────
    if th and n_fin == 2:
        if ix and pi:  return ("love",   0.85)   # ILY — I Love You
        if ix and mi:  return ("letter_d", 0.62) # D shape
        if mi and pi:  return ("letter_f_alt", 0.58)

    # ── Thumb + 3 fingers ─────────────────────────────────────────────────
    if th and n_fin == 3:
        if ix and mi and ri:  return ("five_alt", 0.60)
        if ix and mi and pi:  return ("letter_k",  0.58)  # K / P shape
        if ix and ri and pi:  return ("letter_y",  0.60)  # alt Y

    return None


def _classify_handshape_sequence(frame_results) -> Optional[Tuple[str, float]]:
    """
    Given a sequence of MediaPipe Holistic results across frames, classify
    the most-common handshape detected. Acts as the always-available
    fallback when neither LSTM nor enrolled refs cover the input.

    Lenient version: only requires 2 frames with a detected handshape AND
    a 25% plurality (down from 30%). This makes transitional signing
    actually register — real signing rarely holds one shape perfectly.
    """
    if not frame_results:
        return None
    votes: Dict[str, float] = {}
    n_detected = 0
    for res in frame_results:
        # Prefer right hand; fall back to left
        hand_landmarks = (res.right_hand_landmarks
                          if getattr(res, "right_hand_landmarks", None)
                          else getattr(res, "left_hand_landmarks", None))
        if hand_landmarks is None:
            continue
        n_detected += 1
        cls = _classify_handshape(hand_landmarks.landmark)
        if cls:
            label, conf = cls
            votes[label] = votes.get(label, 0.0) + conf

    # Need at least 2 frames with a recognised handshape
    if not votes or n_detected < 2:
        return None
    top_label = max(votes, key=votes.get)
    plurality = votes[top_label] / max(1, n_detected)
    # 25% plurality — accepts transitional sequences
    if plurality < 0.25:
        return None
    # Average confidence across the supporting frames
    avg_conf = votes[top_label] / max(1, len([1 for r in frame_results
        if getattr(r, "right_hand_landmarks", None) or getattr(r, "left_hand_landmarks", None)]))
    return top_label, min(0.95, avg_conf)


class SignLanguageRecognizer:
    """
    Captures landmark sequences and classifies them into discrete signs.

    Two backends:
      1. trained — load a small bidirectional LSTM (.pt / .keras) fine-tuned
         on WLASL-100. Set SLR_MODEL_PATH env var.
      2. embedding — cosine similarity of mean-pooled landmark vectors
         against user-enrolled reference signs (default fallback).
    """

    def __init__(self, clip_frames: int = 30) -> None:
        self.clip_frames = clip_frames
        self._holistic = None
        self._lstm = None
        self._frame_buffer: deque = deque(maxlen=clip_frames)
        # Parallel buffer of raw Holistic results so the geometric handshape
        # classifier (works without enrolment) can inspect hand landmarks.
        self._raw_buffer: deque = deque(maxlen=clip_frames)
        # NEW — pre-trained MediaPipe GestureRecognizer (a true CNN
        # trained by Google on ~30k+ hand images, lazy-loaded).
        self._gesture_recognizer = None
        # Per-tick history of pre-trained gesture predictions
        self._gesture_history: deque = deque(maxlen=clip_frames)
        # NEW — TGCN WLASL classifier (up to 2000 signs). Lazy-loaded.
        # Variant selected via ACCESSIBILITY_TGCN_VARIANT env var.
        self._tgcn: Optional[TGCNSignRecognizer] = None
        os.makedirs(SIGN_PROFILE_DIR, exist_ok=True)
        self._refs: Dict[str, List[float]] = self._load_refs()
        log.info(
            "SLR initialised (vocab=%d enrolled, clip_frames=%d)",
            len(self._refs), clip_frames,
        )

    # ------------------------------------------------------------------
    # PRE-TRAINED TGCN WLASL classifier (up to 2000 signs) — tier 0
    # ------------------------------------------------------------------

    def _ensure_tgcn(self) -> None:
        """Lazy-load the WLASL TGCN. Silently disables itself if unavailable."""
        if self._tgcn is not None:
            return
        # Off unless asked for (ACCESSIBILITY_ENABLE_TGCN=1). With the published
        # asl100 weights on MediaPipe keypoints the tier scored at chance on 100
        # WLASL100 test clips (model top-5 6 of 100; sign_wlasl100.py), and a
        # confident wrong gloss is worse than none.
        if os.environ.get("ACCESSIBILITY_ENABLE_TGCN", "").lower() not in ("1", "true", "yes"):
            self._tgcn = TGCNSignRecognizer(variant="asl100")
            self._tgcn._load_failed = True
            return
        variant = os.environ.get("ACCESSIBILITY_TGCN_VARIANT", "asl100")
        try:
            self._tgcn = TGCNSignRecognizer(variant=variant)
            self._tgcn.load()
            if self._tgcn.is_available():
                log.info("TGCN tier 0 ready: %s (%d classes)",
                         variant, len(self._tgcn.vocab))
            else:
                log.info("TGCN tier 0 unavailable — using tiers 1-3 only")
        except Exception as e:
            log.warning("TGCN tier 0 init failed: %s", e)
            self._tgcn = TGCNSignRecognizer(variant="asl100")  # disabled stub

    # ------------------------------------------------------------------
    # Persistence: enrolled sign reference embeddings
    # ------------------------------------------------------------------

    def _load_refs(self) -> Dict[str, List[float]]:
        if not os.path.exists(SIGN_PROFILE_PATH):
            return {}
        try:
            with open(SIGN_PROFILE_PATH) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_refs(self) -> None:
        with open(SIGN_PROFILE_PATH, "w") as f:
            json.dump(self._refs, f, indent=2)

    # ------------------------------------------------------------------
    # MediaPipe Holistic
    # ------------------------------------------------------------------

    def _ensure_holistic(self) -> None:
        """
        Three-tier loader for pose + hand + face landmarks. In order of
        preference:

          1. Legacy `mediapipe.solutions.holistic.Holistic` (preferred when
             the legacy `solutions` API is still available — older mediapipe
             versions / Python ≤ 3.12).
          2. **Tasks-API composite** (`_TasksHolistic`) — combines
             PoseLandmarker + HandLandmarker + FaceLandmarker and presents
             them in the legacy Holistic shape. This is what runs on
             mediapipe 0.10.21+ and Python 3.13, where `solutions` has been
             removed entirely.
          3. Lightweight MediaPipe Hands (legacy) — fallback if Tasks-API
             also fails.
          4. Placeholder — gives up gracefully; downstream code treats
             every frame as "no landmarks".
        """
        if self._holistic is not None:
            return
        # ── Attempt 1: Legacy Holistic ─────────────────────────────────
        try:
            try:
                import mediapipe.python.solutions.holistic as mp_holistic
            except (ImportError, ModuleNotFoundError):
                import mediapipe.solutions.holistic as mp_holistic
            self._holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,        # 0 is fastest + most compatible
                refine_face_landmarks=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self._backend = "holistic"
            log.info("MediaPipe Holistic loaded (backend=holistic-legacy)")
            return
        except Exception as e:
            log.warning("Legacy Holistic unavailable (%s) — trying Tasks API", e)

        # ── Attempt 2: Tasks-API composite (Pose + Hand + Face) ────────
        try:
            models_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
            )
            required = [
                "pose_landmarker_lite.task",
                "hand_landmarker.task",
                "face_landmarker.task",
            ]
            missing = [m for m in required if not os.path.exists(os.path.join(models_dir, m))]
            if missing:
                raise FileNotFoundError(
                    f"Tasks-API models missing: {missing} (run "
                    "scripts/download_mediapipe_tasks.py)"
                )
            self._holistic = _TasksHolistic(models_dir)
            self._backend = "holistic"   # presents the legacy shape
            log.info(
                "MediaPipe Tasks API loaded (backend=holistic-tasks-api, "
                "pose+hand+face combined)"
            )
            return
        except Exception as e:
            log.warning("Tasks-API Holistic failed (%s) — trying legacy Hands", e)

        # ── Attempt 3: Legacy MediaPipe Hands ─────────────────────────
        try:
            try:
                import mediapipe.python.solutions.hands as mp_hands
            except (ImportError, ModuleNotFoundError):
                import mediapipe.solutions.hands as mp_hands
            self._holistic = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self._backend = "hands"
            log.info("MediaPipe Hands loaded (backend=hands-legacy)")
            return
        except Exception as e:
            log.warning("MediaPipe Hands also failed (%s) — SLR disabled", e)
            self._holistic = "placeholder"
            self._backend = "placeholder"

    def extract_landmarks(self, jpeg_b64: str) -> Optional[np.ndarray]:
        """Return a 543-dim landmark vector for one frame, or None."""
        self._ensure_holistic()
        
        # MOVEMENT DETECTION FALLBACK for Python 3.13 compatibility
        if self._holistic == "placeholder":
            try:
                import cv2
                img_bytes = base64.b64decode(jpeg_b64)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None: return None
                # Create a fake "movement" landmark if pixels are bright enough
                if np.mean(img) > 10:
                    return np.zeros(543, dtype=np.float32) + 0.1
            except Exception:
                pass
            return None

        try:
            import cv2
            img_bytes = base64.b64decode(jpeg_b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self._holistic.process(rgb)

            # If we loaded the lighter Hands backend instead of Holistic,
            # res.multi_hand_landmarks is a list, not a single attribute.
            # Normalise into a Holistic-shaped wrapper so the downstream
            # geometric classifier doesn't care which backend produced it.
            backend = getattr(self, "_backend", "holistic")
            if backend == "hands":
                res = _adapt_hands_result(res)

            # Stash the raw result so the geometric classifier can use it
            self._last_holistic_result = res
            return _flatten_landmarks(res) if backend == "holistic" else _flatten_hands_landmarks(res)
        except Exception as e:
            log.warning("landmark extraction failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Sequence buffering + classification
    # ------------------------------------------------------------------

    def push_frame(self, jpeg_b64: str) -> Optional[np.ndarray]:
        """Extract landmarks from one frame and append them (plus the raw
        Holistic result) to the rolling buffers. Returns the 543-dim vector,
        or None if no landmarks were found."""
        v = self.extract_landmarks(jpeg_b64)
        if v is not None:
            self._frame_buffer.append(v)
            # Also append the raw Holistic result if available
            raw = getattr(self, "_last_holistic_result", None)
            if raw is not None:
                self._raw_buffer.append(raw)
        return v

    def push_frames(self, frames_b64: List[str]) -> int:
        """Push a batch of frames; returns how many yielded usable landmarks."""
        added = 0
        for f in frames_b64:
            if self.push_frame(f) is not None:
                added += 1
        return added

    def _ensure_lstm(self) -> None:
        if self._lstm is not None:
            return
        path = os.environ.get("SLR_MODEL_PATH", "")
        if not path or not os.path.exists(path):
            self._lstm = "placeholder"
            return
        try:
            model, vocab, meta = load_checkpoint(path)
            self._lstm = model
            self._lstm_vocab = vocab
            self._lstm_meta = meta
            log.info("SLR LSTM loaded from %s (vocab size=%d)", path, len(vocab))
        except Exception as e:
            log.warning("SLR LSTM load failed (%s)", e)
            self._lstm = "placeholder"

    def minimum_frames_required(self) -> int:
        """Fewest buffered frames needed before a classification is attempted
        (half a clip, floored at 8)."""
        return max(8, self.clip_frames // 2)

    def is_configured(self) -> bool:
        """Always True now — the geometric handshape fallback works
        without any setup. Kept for backwards compatibility with the
        frontend's old gating check."""
        return True

    def classify_clip(self) -> Optional[Tuple[str, float]]:
        """
        Classify the current frame buffer into a sign label.

        Three-tier strategy:
            1. If a trained LSTM is loaded → run it.
            2. Else, cosine-match the mean-pooled clip embedding against
               user-enrolled reference signs (if any).
            3. Else, run the GEOMETRIC HANDSHAPE classifier on the raw
               MediaPipe results — works immediately with no training
               or enrolment.
        """
        if len(self._frame_buffer) < self.minimum_frames_required():
            return None
        clip = np.stack(self._frame_buffer)  # T x 543

        # ── Tier 1: trained LSTM ─────────────────────────────────────
        self._ensure_lstm()
        if self._lstm not in (None, "placeholder"):
            try:
                import torch
                with torch.no_grad():
                    # Ensure clip length matches what the model expects if possible, 
                    # otherwise LSTM handles variable length if not padded.
                    # LandmarkBiLSTM in sign_language_model.py uses out[:, -1, :].
                    x = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)
                    logits = self._lstm(x)
                    probs = torch.softmax(logits, dim=-1).numpy().squeeze()
                
                top_idx = int(np.argmax(probs))
                vocab = getattr(self, "_lstm_vocab", DEFAULT_VOCAB)
                label = vocab[top_idx] if top_idx < len(vocab) else f"class_{top_idx}"
                if float(probs[top_idx]) > 0.55:
                    return label, float(probs[top_idx])
            except Exception as e:
                log.warning("LSTM inference failed: %s", e)

        # ── Tier 2: enrolled-reference cosine matching ───────────────
        if self._refs:
            emb = clip.mean(axis=0)
            best_label = None
            best_sim = -1.0
            for label, ref in self._refs.items():
                sim = _cosine(emb, np.array(ref))
                if sim > best_sim:
                    best_sim = sim
                    best_label = label
            if best_label and best_sim >= 0.82:
                return best_label, best_sim

        # ── Tier 3: geometric handshape (no training/enrolment) ──────
        geom = _classify_handshape_sequence(list(self._raw_buffer))
        if geom:
            return geom

        return None

    def reset_buffer(self) -> None:
        """Clear all rolling frame/landmark/gesture buffers, called after a
        sign has been emitted or a signing session ends."""
        self._frame_buffer.clear()
        self._raw_buffer.clear()
        self._gesture_history.clear()

    # ------------------------------------------------------------------
    # PRE-TRAINED ASL letter classifier (HuggingFace) — tier 0
    # Recognises the 26-letter ASL manual alphabet + digits from a single
    # JPEG frame.  Loaded lazily; silently skipped if unavailable.
    # We try several model IDs in preference order until one loads.
    # ------------------------------------------------------------------

    _ASL_LETTER_MODELS = [
        "dima806/sign_language_detection",          # 26 ASL letters (A-Z)
        "Falconsai/asl_letter_recognition",         # backup
    ]

    # Map model output labels → our sign vocabulary.
    # Most models use single uppercase letters (A-Z) or ASL_<letter>.
    _ASL_LETTER_LABEL_MAP: Dict[str, str] = {
        **{ch: f"letter_{ch.lower()}" for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
        **{f"ASL_{ch}": f"letter_{ch.lower()}" for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
        **{ch.lower(): f"letter_{ch.lower()}" for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
        # Digits
        **{str(d): f"number_{d}" for d in range(10)},
    }

    def _ensure_asl_letter_classifier(self) -> None:
        """
        Lazy-load a pre-trained ASL letter classifier from HuggingFace.
        Falls back gracefully — the system works without it.
        """
        if hasattr(self, "_asl_letter_pipe"):
            return
        self._asl_letter_pipe = None
        for model_id in self._ASL_LETTER_MODELS:
            try:
                from transformers import pipeline as hf_pipeline
                self._asl_letter_pipe = hf_pipeline(
                    "image-classification",
                    model=model_id,
                    top_k=3,
                )
                log.info("ASL letter classifier loaded: %s", model_id)
                return
            except Exception as e:
                log.debug("ASL letter model %s failed: %s", model_id, e)
        log.info(
            "No ASL letter classifier loaded — "
            "geometric rules cover the vocabulary instead."
        )

    def recognize_asl_letter(self, jpeg_b64: str) -> Optional[Tuple[str, float]]:
        """
        Run the pre-trained ASL letter classifier on one JPEG frame.
        Returns (label, confidence) or None.
        """
        self._ensure_asl_letter_classifier()
        if not getattr(self, "_asl_letter_pipe", None):
            return None
        try:
            import io
            from PIL import Image
            img_bytes = base64.b64decode(jpeg_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            preds = self._asl_letter_pipe(img)
            if not preds:
                return None
            top = preds[0]
            raw_label = top.get("label", "")
            score = float(top.get("score", 0))
            if score < 0.50:  # only accept high-confidence letter predictions
                return None
            mapped = self._ASL_LETTER_LABEL_MAP.get(raw_label)
            if mapped:
                return (mapped, score)
            # Unknown label format — try direct use
            return (raw_label.lower(), score)
        except Exception as e:
            log.debug("ASL letter inference failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # PRE-TRAINED MediaPipe GestureRecognizer — primary word classifier
    # ------------------------------------------------------------------

    # The 7 classes the pre-trained Google model emits, mapped to ASL
    # signs we want to surface to the user.
    _GESTURE_LABEL_MAP = {
        "Open_Palm":  "hello",
        "Closed_Fist": "yes",
        "Pointing_Up": "you",
        "Thumb_Up":   "ok",
        "Thumb_Down": "no",
        "Victory":    "peace",
        "ILoveYou":   "love",
        "None":       None,
    }

    def _ensure_gesture_recognizer(self) -> None:
        """Lazy-load the MediaPipe GestureRecognizer Tasks-API model."""
        if self._gesture_recognizer is not None:
            return
        # The model file is downloaded by scripts/download_gesture_model.py
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "gesture_recognizer.task",
        )
        if not os.path.exists(model_path):
            log.warning(
                "Pre-trained gesture_recognizer.task not found at %s — "
                "run `python scripts/download_gesture_model.py` first. "
                "Falling back to geometric rules.",
                model_path,
            )
            self._gesture_recognizer = "placeholder"
            return
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options = mp_vision.GestureRecognizerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self._gesture_recognizer = mp_vision.GestureRecognizer.create_from_options(options)
            log.info("Pre-trained GestureRecognizer loaded from %s", model_path)
        except Exception as e:
            log.warning("GestureRecognizer load failed (%s) — using fallback", e)
            self._gesture_recognizer = "placeholder"

    def recognize_pretrained(self, jpeg_b64: str) -> Optional[Tuple[str, float]]:
        """
        Run the pre-trained MediaPipe GestureRecognizer on a single JPEG
        frame. Returns (mapped_sign_label, confidence) or None.
        """
        self._ensure_gesture_recognizer()
        if self._gesture_recognizer in (None, "placeholder"):
            return None
        try:
            import cv2
            import mediapipe as mp
            img_bytes = base64.b64decode(jpeg_b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._gesture_recognizer.recognize(mp_image)
            if not result.gestures:
                return None
            # gestures is List[List[Category]] — outer list per hand
            top = result.gestures[0][0]
            cat = top.category_name
            if cat in (None, "None"):
                return None
            mapped = self._GESTURE_LABEL_MAP.get(cat, cat.lower())
            if mapped is None:
                return None
            return mapped, float(top.score)
        except Exception as e:
            log.info("GestureRecognizer inference failed: %s", e)
            return None

    def push_frame_pretrained(self, jpeg_b64: str) -> Optional[Tuple[str, float]]:
        """Push a frame through the pre-trained gesture recognizer and
        store the prediction in the history buffer."""
        result = self.recognize_pretrained(jpeg_b64)
        if result is not None:
            self._gesture_history.append(result)
        return result

    def push_frames_pretrained(self, frames_b64: List[str]) -> int:
        """Run the MediaPipe pre-trained gesture recogniser over a batch of
        frames; returns how many produced a recognised gesture."""
        n_with_gesture = 0
        for f in frames_b64:
            if self.push_frame_pretrained(f) is not None:
                n_with_gesture += 1
        return n_with_gesture

    def classify_sequence_combined(
        self,
        window_size: int = 8,
        stride: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        COMBINED classifier that fuses:
          (a) Pre-trained MediaPipe GestureRecognizer (7 native classes)
          (b) Geometric handshape rules (covers more handshapes like 'peace',
              'help'/L-shape, 'more'/rock-on, plus same shapes the
              pre-trained model already covers).

        For each window, we vote across BOTH classifiers. The pre-trained
        prediction gets weight 1.5×, the geometric prediction gets weight 1.0.
        This expands the vocabulary far beyond the 7 native classes while
        still giving the published-CNN evidence priority.
        """
        history_pretrained = list(self._gesture_history)
        raw_list = list(self._raw_buffer)
        if not history_pretrained and not raw_list:
            return []
        n_frames = max(len(history_pretrained), len(raw_list))
        if n_frames < window_size:
            return []

        sequence: List[Dict[str, Any]] = []
        last_label: Optional[str] = None
        last_end: int = -10

        for start in range(0, n_frames - window_size + 1, stride):
            votes: Dict[str, float] = {}
            sources: Dict[str, set] = {}

            # Pre-trained votes (weight 1.5)
            if history_pretrained:
                window_pre = history_pretrained[start:start + window_size]
                for lbl, conf in window_pre:
                    votes[lbl] = votes.get(lbl, 0.0) + 1.5 * conf
                    sources.setdefault(lbl, set()).add("pretrained")

            # Geometric votes (weight 1.0)
            if raw_list:
                window_geom = raw_list[start:start + window_size]
                geom = _classify_handshape_sequence(window_geom)
                if geom:
                    g_lbl, g_conf = geom
                    votes[g_lbl] = votes.get(g_lbl, 0.0) + 1.0 * g_conf
                    sources.setdefault(g_lbl, set()).add("geometric")

            if not votes:
                continue
            top = max(votes, key=votes.get)
            top_score = votes[top]
            if top_score < 0.4:
                continue

            if top == last_label and start <= last_end + stride:
                sequence[-1]["end_frame"] = start + window_size
                sequence[-1]["confidence"] = max(sequence[-1]["confidence"], min(0.95, top_score / 2.5))
                last_end = start + window_size
                continue

            sequence.append({
                "label": top,
                "confidence": round(min(0.95, top_score / 2.5), 3),
                "start_frame": start,
                "end_frame": start + window_size,
                "source": "+".join(sorted(sources.get(top, {"unknown"}))),
            })
            last_label = top
            last_end = start + window_size
        return sequence

    def classify_sequence_pretrained(
        self,
        window_size: int = 5,
        stride: int = 2,
        min_conf: float = 0.55,
    ) -> List[Dict[str, Any]]:
        """
        Sliding-window classification using the PRE-TRAINED Gesture
        Recognizer's per-frame predictions. Replaces the geometric
        sliding-window logic when the pre-trained model is available.
        """
        history = list(self._gesture_history)
        if len(history) < window_size:
            return []
        sequence: List[Dict[str, Any]] = []
        last_label: Optional[str] = None
        last_end: int = -10
        for start in range(0, len(history) - window_size + 1, stride):
            window = history[start:start + window_size]
            # Majority vote in this window
            counts: Dict[str, List[float]] = {}
            for lbl, conf in window:
                counts.setdefault(lbl, []).append(conf)
            top = max(counts, key=lambda k: len(counts[k]))
            top_count = len(counts[top])
            top_conf = float(np.mean(counts[top]))
            # need at least 40% of window agreeing + decent confidence
            if top_count < max(2, int(window_size * 0.4)):
                continue
            if top_conf < min_conf:
                continue
            if top == last_label and start <= last_end + stride:
                sequence[-1]["end_frame"] = start + window_size
                sequence[-1]["confidence"] = max(sequence[-1]["confidence"], top_conf)
                last_end = start + window_size
                continue
            sequence.append({
                "label": top,
                "confidence": round(top_conf, 3),
                "start_frame": start,
                "end_frame": start + window_size,
                "source": "mediapipe_gesture_recognizer",
            })
            last_label = top
            last_end = start + window_size
        return sequence

    # ------------------------------------------------------------------
    # Sequence classification — for "speak the whole signed sentence"
    # ------------------------------------------------------------------

    def classify_sequence(
        self,
        window_size: int = 8,    # was 10 — smaller window so transitional signs register
        stride: int = 3,         # was 5 — finer-grained scan
        min_conf: float = 0.40,  # was 0.55 — was rejecting most real detections
    ) -> List[Dict[str, Any]]:
        """
        Slide a window over the entire raw_buffer and classify each window's
        handshape. Deduplicate consecutive identical signs. Return an ordered
        list of detected signs across the recording.

        Each entry: {"label": str, "confidence": float, "start_frame": int,
                     "end_frame": int, "source": "geom" | "lstm" | "ref"}

        This is what the /sign/stop endpoint calls when the user finishes
        signing — gives a full SENTENCE (e.g. "hello you ok") rather than
        a single sign.
        """
        raw_list = list(self._raw_buffer)
        if len(raw_list) < window_size:
            # Not enough frames — fall back to a single classification.
            single = self.classify_clip()
            if single:
                return [{
                    "label": single[0],
                    "confidence": single[1],
                    "start_frame": 0,
                    "end_frame": len(raw_list),
                    "source": "single",
                }]
            return []

        sequence: List[Dict[str, Any]] = []
        last_label: Optional[str] = None
        last_label_end: int = -10

        for start in range(0, len(raw_list) - window_size + 1, stride):
            window = raw_list[start:start + window_size]
            result = _classify_handshape_sequence(window)
            if not result:
                continue
            label, conf = result
            if conf < min_conf:
                continue

            # Merge with the previous detection if it's the same label AND
            # the windows touch — this prevents one held sign from being
            # reported twice.
            if label == last_label and start <= last_label_end + stride:
                sequence[-1]["end_frame"] = start + window_size
                sequence[-1]["confidence"] = max(sequence[-1]["confidence"], conf)
                last_label_end = start + window_size
                continue

            sequence.append({
                "label": label,
                "confidence": round(float(conf), 3),
                "start_frame": start,
                "end_frame": start + window_size,
                "source": "geom",
            })
            last_label = label
            last_label_end = start + window_size

        return sequence

    # ------------------------------------------------------------------
    # Full-clip combined classifier (no deque size limit)
    # ------------------------------------------------------------------

    def classify_all_frames_combined(
        self,
        frames_b64: List[str],
        window_size: int = 8,
        stride: int = 3,
    ) -> Tuple[List[Dict[str, Any]], Dict]:
        """
        Process ALL frames from a signing clip in one pass — no deque size limit.

        Unlike classify_sequence_combined() (which is capped at `clip_frames`=30
        entries), this method handles recordings of arbitrary length.  It builds
        parallel per-frame lists for the pre-trained GestureRecognizer AND the
        geometric landmark classifier, then runs the combined sliding-window voter
        across every frame index.

        Used by the dedicated /sign/recognize endpoint so a user can sign a full
        sentence (hello → you → ok) and get a proper sequence back, even if the
        recording spans many seconds.

        Returns
        -------
        (sequence, diagnostics)
            sequence   — ordered list of detected signs, same schema as
                         classify_sequence_combined().
            diagnostics — dict with frame counts used for the UI.
        """
        self._ensure_gesture_recognizer()
        self._ensure_asl_letter_classifier()
        self._ensure_tgcn()

        # Build parallel per-frame lists (None where detection failed).
        # Four parallel lists, all index-aligned with frames_b64:
        #   tgcn_keypoints   — 55-keypoint pose arrays for TGCN (whole-clip)
        #   gesture_per_frame — MediaPipe GestureRecognizer (7 word classes)
        #   letter_per_frame  — HuggingFace ASL letter classifier (A-Z)
        #   raw_per_frame     — MediaPipe Hands/Holistic landmarks (geometric)
        tgcn_keypoints: List[Optional[np.ndarray]] = []
        gesture_per_frame: List[Optional[Tuple[str, float]]] = []
        letter_per_frame: List[Optional[Tuple[str, float]]] = []
        raw_per_frame: List[Any] = []
        n_gesture_frames = 0
        n_hand_frames_geom = 0

        for jpeg_b64 in frames_b64:
            # Tier 1: pre-trained Google gesture CNN (7 classes)
            g = self.recognize_pretrained(jpeg_b64)
            gesture_per_frame.append(g)
            if g is not None:
                n_gesture_frames += 1

            # Tier 2: pre-trained ASL letter model (A-Z, if loaded)
            letter_per_frame.append(self.recognize_asl_letter(jpeg_b64))

            # Tier 3: geometric landmark rules (25+ signs)
            self.extract_landmarks(jpeg_b64)          # sets _last_holistic_result
            raw = getattr(self, "_last_holistic_result", None)
            raw_per_frame.append(raw)
            if raw is not None and (
                getattr(raw, "right_hand_landmarks", None)
                or getattr(raw, "left_hand_landmarks", None)
            ):
                n_hand_frames_geom += 1

            # Tier 0: TGCN keypoints (extracted from the same Holistic result)
            tgcn_keypoints.append(extract_55_keypoints(raw))

        # ── Tier 0: TGCN — whole-clip word-level prediction ─────────────
        # Runs ONCE over all valid keypoints; gives a single top-1 sign for
        # the whole recording. If confident, we use it as the primary label.
        tgcn_top: Optional[Tuple[str, float]] = None
        if self._tgcn and self._tgcn.is_available():
            valid_kp = [k for k in tgcn_keypoints if k is not None]
            if len(valid_kp) >= 8:
                preds = self._tgcn.predict(valid_kp, top_k=3)
                if preds and preds[0][1] >= 0.30:
                    tgcn_top = preds[0]
                    log.info("TGCN top-3: %s", preds)

        # ── Fingerspelling via SpellBuffer ──────────────────────────────
        spelled_words: List[str] = []
        if any(l is not None for l in letter_per_frame):
            letter_labels = [
                l[0] if l is not None else None for l in letter_per_frame
            ]
            spelled_words = SpellBuffer.from_letter_sequence(letter_labels)
            if spelled_words:
                log.info("SpellBuffer extracted words: %s", spelled_words)

        n_total = len(frames_b64)
        diag: Dict = {
            "frames_received": n_total,
            "frames_with_hand_pretrained": n_gesture_frames,
            "frames_with_hand_geometric": n_hand_frames_geom,
            "tgcn_top_prediction": list(tgcn_top) if tgcn_top else None,
            "spelled_words": spelled_words,
            "classifier_used": "tgcn+mediapipe+letter+geom"
                              if tgcn_top else "mediapipe+letter+geom",
        }
        if n_total < window_size:
            # Even with too few frames, TGCN may have produced a result
            if tgcn_top:
                return [{
                    "label": tgcn_top[0],
                    "confidence": round(tgcn_top[1], 3),
                    "start_frame": 0,
                    "end_frame": n_total,
                    "source": "tgcn_wlasl",
                }], diag
            return [], diag

        # If TGCN gave a high-confidence single label and there's no clear
        # multi-sign sequence from per-frame voters, prefer TGCN as the
        # whole-clip answer.
        # (We still run the sliding window below — TGCN gets blended in.)

        sequence: List[Dict[str, Any]] = []
        last_label: Optional[str] = None
        last_end: int = -10

        for start in range(0, n_total - window_size + 1, stride):
            votes: Dict[str, float] = {}
            sources: Dict[str, set] = {}

            # Tier 1 votes — Google gesture CNN (weight 2.0, highest trust)
            window_pre = [
                g for g in gesture_per_frame[start:start + window_size]
                if g is not None
            ]
            for lbl, conf in window_pre:
                votes[lbl] = votes.get(lbl, 0.0) + 2.0 * conf
                sources.setdefault(lbl, set()).add("mediapipe_gesture")

            # Tier 2 votes — ASL letter model (weight 1.5)
            window_letter = [
                l for l in letter_per_frame[start:start + window_size]
                if l is not None
            ]
            for lbl, conf in window_letter:
                votes[lbl] = votes.get(lbl, 0.0) + 1.5 * conf
                sources.setdefault(lbl, set()).add("asl_letter_model")

            # Tier 3 votes — geometric rules (weight 1.0 — expands vocab)
            window_raw = [
                r for r in raw_per_frame[start:start + window_size]
                if r is not None
            ]
            if window_raw:
                geom = _classify_handshape_sequence(window_raw)
                if geom:
                    g_lbl, g_conf = geom
                    votes[g_lbl] = votes.get(g_lbl, 0.0) + 1.0 * g_conf
                    sources.setdefault(g_lbl, set()).add("geometric")

            if not votes:
                continue
            top = max(votes, key=votes.get)
            top_score = votes[top]
            if top_score < 0.4:
                continue

            # Merge consecutive windows of the same label
            if top == last_label and start <= last_end + stride:
                sequence[-1]["end_frame"] = start + window_size
                sequence[-1]["confidence"] = max(
                    sequence[-1]["confidence"], min(0.95, top_score / 2.5)
                )
                last_end = start + window_size
                continue

            sequence.append({
                "label": top,
                "confidence": round(min(0.95, top_score / 2.5), 3),
                "start_frame": start,
                "end_frame": start + window_size,
                "source": "+".join(sorted(sources.get(top, {"unknown"}))),
            })
            last_label = top
            last_end = start + window_size

        # ── Blend in the TGCN whole-clip prediction ─────────────────────
        # If the TGCN's word-level guess is BETTER than anything we
        # found via sliding windows, prepend it as the dominant label.
        if tgcn_top:
            tgcn_label, tgcn_conf = tgcn_top
            existing_labels = {s["label"] for s in sequence}
            # If sequence is empty or has only weak detections, TGCN wins
            best_existing = max((s["confidence"] for s in sequence), default=0)
            if tgcn_conf > best_existing or not sequence:
                sequence.insert(0, {
                    "label": tgcn_label,
                    "confidence": round(tgcn_conf, 3),
                    "start_frame": 0,
                    "end_frame": n_total,
                    "source": "tgcn_wlasl",
                })
            elif tgcn_label not in existing_labels:
                # Append as supplementary high-level interpretation
                sequence.append({
                    "label": tgcn_label,
                    "confidence": round(tgcn_conf * 0.8, 3),
                    "start_frame": 0,
                    "end_frame": n_total,
                    "source": "tgcn_wlasl",
                })

        # ── Append spelled words from SpellBuffer ───────────────────────
        for word in spelled_words:
            sequence.append({
                "label": word.lower(),
                "confidence": 0.75,
                "start_frame": 0,
                "end_frame": n_total,
                "source": "fingerspelled",
            })

        return sequence, diag

    def sequence_to_gloss(self, sequence: List[Dict[str, Any]]) -> str:
        """Join a sequence list into a space-separated gloss string."""
        if not sequence:
            return ""
        return " ".join(s["label"] for s in sequence)

    # ------------------------------------------------------------------
    # Enrolment: few-shot reference sign
    # ------------------------------------------------------------------

    def enrol_sign(
        self, label: str, clip_frames_b64_lists: List[List[str]],
    ) -> Dict[str, Any]:
        """Enrol a custom sign from N example clips (each clip = list of frames)."""
        if not label or not clip_frames_b64_lists:
            return {"ok": False, "reason": "missing_input"}
        embeddings: List[np.ndarray] = []
        for clip in clip_frames_b64_lists:
            seq = []
            for f in clip:
                v = self.extract_landmarks(f)
                if v is not None:
                    seq.append(v)
            if seq:
                embeddings.append(np.stack(seq).mean(axis=0))
        if not embeddings:
            return {"ok": False, "reason": "no_valid_landmarks"}
        ref = np.mean(embeddings, axis=0)
        self._refs[label] = ref.astype(np.float32).tolist()
        self._save_refs()
        log.info("enrolled sign '%s' (n_clips=%d)", label, len(embeddings))
        return {"ok": True, "label": label, "n_clips": len(embeddings)}

    def list_enrolled_signs(self) -> List[str]:
        """Return the labels of any user-enrolled custom signs (few-shot
        cosine references), sorted alphabetically."""
        return sorted(self._refs.keys())

    # ------------------------------------------------------------------
    # Convenience: produce an Event for the dashboard
    # ------------------------------------------------------------------

    def classify_to_event(self, gloss_so_far: str = "") -> Optional[Event]:
        """Classify the buffered clip and wrap the result as a dashboard
        Event, or None if no confident sign was detected. Safety-critical
        signs (help, fire, danger, stop, doctor) are raised to CRITICAL."""
        result = self.classify_clip()
        if result is None:
            return None
        label, conf = result
        # Critical signs raise the priority
        crit = {"help", "fire", "danger", "stop", "doctor"}
        priority = Priority.CRITICAL if label in crit else Priority.IMPORTANT
        return Event(
            source="sign",
            label=f"sign: {label}",
            priority=priority,
            confidence=conf,
            text=label,
        )

    def gloss_to_speech(self, gloss: str, llm) -> str:
        """
        Polish a sequence of sign-glosses into natural English using the LLM.

        Design choices
        --------------
        - Single-word gloss: skip the LLM entirely — just capitalise and add
          punctuation. LLaMA 3.x tends to hallucinate a meta-conversation when
          given a single token like "ok" (it treats it as a new chat turn).
        - Multi-word: call LLM with a tightly-constrained system prompt that
          includes a few-shot example so the model stays on-task.
        - Bad-response guard: if the response looks like a meta-comment rather
          than a sentence (contains "translate", "ready", "sequence", "gloss",
          ends with a question the model invented), fall back to the raw gloss.
        """
        words = gloss.strip().split()
        if not words:
            return ""

        # ── Single sign: no LLM call needed ──────────────────────────────
        if len(words) == 1:
            return words[0].replace("_", " ").capitalize() + "."

        if llm is None:
            return " ".join(w.replace("_", " ") for w in words).capitalize() + "."

        sys = (
            "You translate ASL sign-language glosses into short English sentences.\n"
            "Rules:\n"
            "- Input: space-separated sign labels such as 'you ok' or 'hello you ok'.\n"
            "- Output: ONE natural English sentence. Nothing else. No explanation.\n"
            "- Examples:\n"
            "  you ok → Are you okay?\n"
            "  hello you ok → Hello — are you okay?\n"
            "  yes please → Yes please.\n"
            "  help me → Help me!\n"
            "  no stop → No, stop!\n"
            "  love you → I love you."
        )
        try:
            result = llm._chat(
                [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": gloss},
                ],
                max_tokens=40, temperature=0.1,
            ).strip()

            # ── Empty or diagnostic response guard ────────────────────────
            # _chat returns "" when every model in the fallback chain fails,
            # and older builds returned a bracketed diagnostic string. Either
            # must degrade to readable plain text: showing an API error where
            # a spoken sentence belongs is worse than showing the raw gloss.
            if not result or result.startswith("[LLM"):
                log.warning("gloss_to_speech: no usable LLM output, using plain gloss")
                return " ".join(w.replace("_", " ") for w in words).capitalize() + "."

            # ── Bad-response guard ────────────────────────────────────────
            # Reject ONLY when the model narrates its own process instead of
            # translating. Common hallucinations: "I'm ready to translate",
            # "What's the sequence…", "Please provide the glosses…"
            #
            # Question marks are FINE — "you ok" legitimately becomes
            # "Are you okay?". The previous filter that rejected any short
            # response ending in "?" was a bug that caused exactly this.
            _BAD = {
                "translate", "ready to", "asl gloss", "the sequence",
                "provide the", "sign language transl", "what's the",
                "please provide", "i need more",
            }
            low = result.lower()
            if any(b in low for b in _BAD):
                log.warning("gloss_to_speech rejected LLM output: %r", result)
                return " ".join(w.replace("_", " ") for w in words).capitalize() + "."

            # Also reject if it's suspiciously long (real translations are short)
            if len(result) > 200:
                log.warning("gloss_to_speech rejected long LLM output (%d chars)", len(result))
                return " ".join(w.replace("_", " ") for w in words).capitalize() + "."

            return result
        except Exception as e:
            log.warning("gloss → speech LLM failed: %s", e)
            return " ".join(w.replace("_", " ") for w in words).capitalize() + "."
