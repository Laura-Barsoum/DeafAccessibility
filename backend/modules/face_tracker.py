"""
face_tracker.py — MediaPipe-based face tracking + speaker attribution.

For each video frame:
    - Detects faces via MediaPipe (Tasks-API FaceLandmarker; the legacy
      FaceMesh solution is used only if an older MediaPipe still ships it)
    - Estimates whether each face is "actively speaking" via mouth-aspect-ratio
      variation (a classic but reliable visual VAD signal)
    - Outputs: list of {face_id, position_horizontal (left/centre/right),
      speaking_score, gaze_direction}

This lets the system attribute speech to the visible speaker, even before
proper Pyannote diarization runs.
"""
from __future__ import annotations

import base64
import logging
import os
import threading
from collections import deque
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("accessibility.face")


# Lip / mouth landmarks (MediaPipe FaceMesh)
UPPER_LIP_TOP = 13
LOWER_LIP_BOT = 14
LIP_LEFT = 78
LIP_RIGHT = 308


_FACE_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "face_landmarker.task",
)


class _TasksFaceMesh:
    """Gives the Tasks-API FaceLandmarker the legacy FaceMesh interface.

    MediaPipe removed `mediapipe.solutions`, so the original FaceMesh import
    failed and this tracker silently fell back to a placeholder that did no
    work: speaker attribution never ran, and profiling showed the stage taking
    0 ms. The Tasks face model returns the same landmark topology, so the lip
    indices above are unchanged. A lock guards the detector, which is not
    safe for concurrent calls from the thread pool.
    """

    def __init__(self, model_path: str, max_faces: int = 4) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        self._mp = mp
        self._lock = threading.Lock()
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._det = mp_vision.FaceLandmarker.create_from_options(opts)

    def process(self, rgb: np.ndarray) -> SimpleNamespace:
        """Detect faces; return an object shaped like FaceMesh's result."""
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb))
        with self._lock:
            res = self._det.detect(image)
        faces = [
            SimpleNamespace(landmark=[SimpleNamespace(x=p.x, y=p.y, z=p.z) for p in pts])
            for pts in (getattr(res, "face_landmarks", None) or [])
        ]
        return SimpleNamespace(multi_face_landmarks=faces)


class FaceTracker:
    def __init__(self, history_len: int = 12) -> None:
        self._mesh = None
        self._history: Dict[int, deque] = {}  # face_id → mar history
        self._history_len = history_len
        log.info("FaceTracker initialised (lazy load)")

    def _ensure_loaded(self) -> None:
        if self._mesh is not None:
            return
        try:
            # DEEP IMPORT FIX FOR PYTHON 3.13 — try both paths
            try:
                import mediapipe.python.solutions.face_mesh as mp_face_mesh
            except (ImportError, ModuleNotFoundError):
                import mediapipe.solutions.face_mesh as mp_face_mesh
            
            self._mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=4,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            log.info("MediaPipe FaceMesh LOADED SUCCESSFULLY")
        except Exception as e:
            log.info("legacy FaceMesh unavailable (%s); trying Tasks FaceLandmarker", e)
            try:
                self._mesh = _TasksFaceMesh(_FACE_MODEL)
                log.info("MediaPipe FaceLandmarker (Tasks API) loaded for face tracking")
            except Exception as e2:
                log.warning("face tracking unavailable, speaker attribution disabled (%s)", e2)
                self._mesh = "placeholder"

    def analyse_frame(self, jpeg_b64: str) -> List[Dict[str, Any]]:
        """Return list of detected faces with speaking scores."""
        self._ensure_loaded()
        if self._mesh == "placeholder":
            return []

        try:
            import cv2
            jpeg_bytes = base64.b64decode(jpeg_b64)
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return []
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self._mesh.process(rgb)
            if not result.multi_face_landmarks:
                return []
        except Exception as e:
            log.warning("face frame failed: %s", e)
            return []

        h, w = img.shape[:2]
        out = []
        for fid, fl in enumerate(result.multi_face_landmarks):
            lm = fl.landmark
            # Position: face centre x relative to frame (0=left, 1=right)
            xs = [p.x for p in lm]
            cx = float(np.mean(xs))
            position = "left" if cx < 0.4 else ("right" if cx > 0.6 else "centre")

            # Mouth aspect ratio (vertical / horizontal)
            try:
                v = abs(lm[UPPER_LIP_TOP].y - lm[LOWER_LIP_BOT].y)
                hh = abs(lm[LIP_RIGHT].x - lm[LIP_LEFT].x)
                mar = v / max(hh, 1e-6)
            except Exception:
                mar = 0.0

            history = self._history.setdefault(fid, deque(maxlen=self._history_len))
            history.append(mar)
            speaking_score = float(np.std(history)) if len(history) > 4 else 0.0
            speaking = speaking_score > 0.02  # empirical threshold

            out.append({
                "face_id": fid,
                "position": position,
                "mar": round(mar, 4),
                "speaking_score": round(speaking_score, 4),
                "speaking": bool(speaking),
            })
        return out

    def analyse_frames(self, frames_b64: List[str]) -> Dict[str, Any]:
        """Aggregate over multiple frames; returns the speaker attribution
        verdict and per-face stats."""
        all_faces: List[Dict[str, Any]] = []
        for f in frames_b64:
            faces = self.analyse_frame(f)
            for face in faces:
                all_faces.append(face)
        if not all_faces:
            return {"speaker_attribution": None, "faces": []}

        # Aggregate speaking_score per face_id
        agg: Dict[int, Dict[str, Any]] = {}
        for face in all_faces:
            fid = face["face_id"]
            if fid not in agg:
                agg[fid] = {**face, "scores": []}
            agg[fid]["scores"].append(face["speaking_score"])

        for fid, a in agg.items():
            a["mean_speaking_score"] = float(np.mean(a["scores"]))
            del a["scores"]

        speakers = sorted(agg.values(), key=lambda x: -x["mean_speaking_score"])
        primary = speakers[0] if speakers and speakers[0]["mean_speaking_score"] > 0.02 else None
        return {
            "speaker_attribution": primary,
            "faces": speakers,
        }
