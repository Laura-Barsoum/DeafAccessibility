"""
hazard_detector.py — YOLO-based real-time hazard detection.

Adds a *visual radar* on top of the audio-only safety net: things you
should know about even before they make a sound — fire, knives, an
approaching vehicle, an unleashed dog, etc.

Architecture
    - Default: Ultralytics YOLOv11 (latest, 2024) "yolo11n.pt" — 80-class
      COCO model. Lightweight (~6 MB) and runs at 25-40 FPS on CPU.
    - We map COCO classes to a smaller set of "hazard" categories with
      priorities. Non-hazard detections are suppressed but counted.

Output
    A list of HazardEvent records, each with:
        label  ("fire", "knife", "car", ...)
        priority (CRITICAL / IMPORTANT / INFORM)
        bbox   (x1, y1, x2, y2) — normalised 0-1
        radar_position  (north / north-east / east / ... / centre)
        approaching     (True if the object's bbox is growing fast across frames)
        confidence

This produces both semantic events for the LLM and geometric data for
the frontend "radar" visualisation.

References
    - Wang et al. (2024). YOLOv11 (Ultralytics).
    - Lin et al. (2014). Microsoft COCO: 80-class detection benchmark.
"""
from __future__ import annotations

import base64
import logging
import math
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .events import Event, Priority

log = logging.getLogger("accessibility.hazard")


# COCO class index → (display name, priority).
# Indexes are the 80 standard COCO ids used by Ultralytics models.
HAZARD_MAP: Dict[int, Tuple[str, Priority]] = {
    # Vehicles — important if they're close/approaching
    1: ("bicycle", Priority.INFORM),
    2: ("car", Priority.IMPORTANT),
    3: ("motorcycle", Priority.IMPORTANT),
    5: ("bus", Priority.IMPORTANT),
    7: ("truck", Priority.IMPORTANT),
    # Animals — context-dependent
    16: ("dog", Priority.INFORM),
    17: ("cat", Priority.INFORM),
    # Edged tools — context-dependent (ok in kitchen, scary on street)
    43: ("knife", Priority.IMPORTANT),
    44: ("spoon", Priority.AMBIENT),
    # Fire / smoke proxies (COCO doesn't have fire; we match scene captions instead)
    # Person — needed for "someone behind you" alerts (combined with face direction)
    0: ("person", Priority.INFORM),
    # Stove / oven hot surfaces
    68: ("microwave", Priority.AMBIENT),
    69: ("oven", Priority.AMBIENT),
    71: ("sink", Priority.AMBIENT),
}

# Special non-COCO labels we add manually via OCR / scene-caption keywords.
EXTRA_HAZARD_KEYWORDS = {
    "fire": Priority.CRITICAL,
    "smoke": Priority.CRITICAL,
    "flame": Priority.CRITICAL,
    "explosion": Priority.CRITICAL,
    "blood": Priority.IMPORTANT,
    "broken glass": Priority.IMPORTANT,
}


def _iou(a, b) -> float:
    """Intersection-over-union for two normalised bboxes [x1,y1,x2,y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _radar_position(cx: float, cy: float) -> str:
    """Map normalised centre (0-1) to a compass-style radar bin."""
    # cx: 0=left, 1=right ; cy: 0=top, 1=bottom
    dx = cx - 0.5
    dy = cy - 0.5
    if abs(dx) < 0.18 and abs(dy) < 0.18:
        return "centre"
    angle = math.degrees(math.atan2(-dy, dx))  # 0=east, 90=north
    if angle < 0:
        angle += 360
    sectors = ["east", "north-east", "north", "north-west",
               "west", "south-west", "south", "south-east"]
    idx = int((angle + 22.5) // 45) % 8
    return sectors[idx]


class HazardDetector:
    def __init__(self, model_name: str = "yolo11n.pt") -> None:
        self.model_name = os.environ.get("YOLO_MODEL", model_name)
        self._yolo = None
        # Track each detected class id's bbox area over time, for
        # "approaching" estimation.
        self._bbox_area_history: Dict[int, deque] = {}
        log.info("HazardDetector initialised (model=%s, lazy load)", self.model_name)

    # ------------------------------------------------------------------
    # Lazy YOLO load. ultralytics auto-downloads weights on first use.
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._yolo is not None:
            return
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(self.model_name)
            log.info("YOLO loaded: %s", self.model_name)
        except Exception as e:
            log.warning("YOLO/Ultralytics unavailable (%s) — hazard detection disabled", e)
            self._yolo = "placeholder"

    # ------------------------------------------------------------------
    # Frame inference
    # ------------------------------------------------------------------

    def detect(self, jpeg_b64: str) -> List[Dict[str, Any]]:
        """
        Run YOLO on one frame and return a list of hazard dicts:
            { label, priority, bbox, radar_position, approaching, confidence }
        """
        self._ensure_loaded()
        if self._yolo == "placeholder":
            return []

        try:
            import cv2
            img_bytes = base64.b64decode(jpeg_b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return []
            h, w = img.shape[:2]
            results = self._yolo(img, verbose=False)
        except Exception as e:
            log.warning("YOLO inference failed: %s", e)
            return []

        out: List[Dict[str, Any]] = []
        if not results:
            return out
        r0 = results[0]
        if r0.boxes is None:
            return out

        # First pass: collect all valid detections, sorted by area (largest first).
        boxes = r0.boxes
        candidates = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id not in HAZARD_MAP:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox = [x1 / w, y1 / h, x2 / w, y2 / h]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            candidates.append({
                "cls_id": cls_id, "bbox": bbox, "cx": cx, "cy": cy,
                "area": area, "conf": float(boxes.conf[i].item()),
            })
        candidates.sort(key=lambda c: -c["area"])

        # ── DEDUPLICATION: for each class, keep at most ONE detection per
        # "spatial slot" — prevents the user being reported as multiple
        # phantom people in different radar positions. Two detections of
        # the same class within 25% bbox-IoU are considered the same.
        kept: List[Dict[str, Any]] = []
        for c in candidates:
            duplicate = False
            for k in kept:
                if k["cls_id"] != c["cls_id"]:
                    continue
                if _iou(k["bbox"], c["bbox"]) > 0.25:
                    duplicate = True
                    break
                # For 'person' specifically, suppress small body-part boxes
                # that overlap a larger one (head + torso double-detection)
                if c["cls_id"] == 0 and _iou(k["bbox"], c["bbox"]) > 0.10:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(c)

        # Special rule: 'person' is INFORM at most. NEVER 'approaching'
        # — sitting users naturally have bbox jitter that looks like
        # approach. Only vehicles trigger the approaching escalation.
        for c in kept:
            cls_id = c["cls_id"]
            label, priority = HAZARD_MAP[cls_id]
            history = self._bbox_area_history.setdefault(cls_id, deque(maxlen=10))
            approaching = False
            if history and label in ("car", "motorcycle", "bus", "truck"):
                growth = c["area"] - history[-1]
                approaching = growth > 0.02
                if approaching:
                    priority = Priority.CRITICAL
            history.append(c["area"])

            out.append({
                "label": label,
                "priority": int(priority),
                "priority_name": priority.name,
                "bbox": [round(b, 4) for b in c["bbox"]],
                "centre": [round(c["cx"], 4), round(c["cy"], 4)],
                "area": round(c["area"], 4),
                "radar_position": _radar_position(c["cx"], c["cy"]),
                "approaching": approaching,
                "confidence": c["conf"],
            })

        return out

    # ------------------------------------------------------------------
    # Convenience: detect on a list of frames, deduplicate, and emit
    # Event records for the fusion engine.
    # ------------------------------------------------------------------

    def detect_to_events(self, frames_b64: List[str]) -> Tuple[List[Event], List[Dict[str, Any]]]:
        """
        Returns:
            events: List[Event] for the fusion engine
            hazards: List[dict] for the frontend radar
        """
        if not frames_b64:
            return [], []
        # Use the middle frame for the primary detection (saves compute)
        mid = frames_b64[len(frames_b64) // 2]
        hazards = self.detect(mid)

        events: List[Event] = []
        for h in hazards:
            label_full = f"{h['label']} ({h['radar_position']})"
            if h["approaching"]:
                label_full += " — approaching"
            events.append(Event(
                source="vision",
                label=label_full,
                priority=Priority(int(h["priority"])),
                confidence=h["confidence"],
                direction=h["radar_position"],
                extra={"bbox": h["bbox"], "approaching": h["approaching"]},
            ))
        return events, hazards

    # ------------------------------------------------------------------
    # Scene-caption based hazard keywords (no YOLO required)
    # ------------------------------------------------------------------

    @staticmethod
    def keywords_in_scene(caption: str) -> List[Event]:
        if not caption:
            return []
        out = []
        low = caption.lower()
        for kw, prio in EXTRA_HAZARD_KEYWORDS.items():
            if kw in low:
                out.append(Event(
                    source="vision",
                    label=f"scene mentions: {kw}",
                    priority=prio,
                    confidence=0.6,
                ))
        return out
