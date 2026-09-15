"""
tgcn_sign_model.py — Temporal Graph Convolutional Network (TGCN) for WLASL.

This is the architecture from Li et al. (2020), "Word-level Deep Sign Language
Recognition from Video: A New Large-scale Dataset and Methods Comparison". The
TGCN processes 55 body+hand keypoints across 50 video frames and outputs a
class probability over the WLASL vocabulary.

Pre-trained checkpoints (~3 MB to ~50 MB each) cover four vocabulary sizes:
    asl100  — 100 most common WLASL signs   (hidden=64,  stages=20)
    asl300  — 300 most common               (hidden=256, stages=24)
    asl1000 — 1 000 signs                   (hidden=256, stages=24)
    asl2000 — full WLASL vocabulary         (hidden=256, stages=24)

Loading
-------
The wrapper class `TGCNSignRecognizer` loads backend/data/tgcn/<variant>/
pytorch_model.bin when present, otherwise downloads the variant from Hugging
Face (sharonn18/tgcn-wlasl, or the repo in ACCESSIBILITY_TGCN_REPO). A
checkpoint whose weights do not all fit the network is refused: this file once
defined an attention block and a flattening classifier that no checkpoint
contains and loaded with strict=False, so the model predicted with random
weights. Without a checkpoint the recognizer disables itself and the system
keeps working via MediaPipe + geometric rules.

Inference path
--------------
    JPEG frame ─► MediaPipe Holistic ─► 55 keypoints (x,y)
                                            │
                            50-frame buffer │
                                            ▼
                            TGCN forward → top-k WLASL gloss

Citation
--------
Li, D. et al. (2020). Word-level Deep Sign Language Recognition from Video.
WACV 2020. https://arxiv.org/abs/1910.11006
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("accessibility.tgcn")


# The WLASL-100 glosses in the order of WLASL_v0.3.json. Whether the asl100
# checkpoint uses this order or sorted order is set by CLASS_ORDER. A class
# file next to the checkpoint takes priority. (This list once had 102
# entries; "wait" and "yellow" are not among the 100 glosses.)
WLASL100_DEFAULT_ORDER = [
    "book", "drink", "computer", "before", "chair", "go", "clothes", "who",
    "candy", "cousin", "deaf", "fine", "help", "no", "thin", "walk", "year",
    "yes", "all", "black", "cool", "finish", "hot", "like", "many", "mother",
    "now", "orange", "table", "thanksgiving", "what", "woman", "bed", "blue",
    "bowling", "can", "dog", "family", "fish", "graduate", "hat", "hearing",
    "kiss", "language", "later", "man", "shirt", "study", "tall", "white",
    "wrong", "accident", "apple", "bird", "change", "color", "corn", "cow",
    "dance", "dark", "doctor", "eat", "enjoy", "forget", "give", "last",
    "meet", "pink", "pizza", "play", "school", "secretary", "short", "time",
    "want", "work", "africa", "basketball", "birthday", "brown", "but",
    "cheat", "city", "cook", "decide", "full", "how", "jacket", "letter",
    "medicine", "need", "paint", "paper", "pull", "purple", "right", "same",
    "son", "tell", "thursday",
]


# ---------------------------------------------------------------------------
# TGCN architecture (Li et al. 2020)
# ---------------------------------------------------------------------------

def _build_modules():
    """Lazy-imports torch and returns (nn, F, Parameter). Centralised so the
    rest of the module imports without torch installed."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def make_GraphConvolution(node_n: int):
    """Factory returning a GraphConvolution class bound to a fixed node count
    (55 skeleton keypoints). Defined as a factory so the learned adjacency
    matrix is sized correctly without a global constant."""
    torch, nn, F = _build_modules()

    class GraphConvolution(nn.Module):
        """Self-adjacent GCN layer (adj matrix is learned)."""
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            self.att = nn.Parameter(torch.FloatTensor(node_n, node_n))
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features))
            else:
                self.register_parameter("bias", None)
            self.reset_parameters()

        def reset_parameters(self):
            stdv = 1.0 / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            self.att.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)

        def forward(self, x):
            support = torch.matmul(x, self.weight)
            output = torch.matmul(self.att, support)
            if self.bias is not None:
                return output + self.bias
            return output

    return GraphConvolution


def build_TGCN(num_class: int, hidden_feature: int, num_stage: int,
               input_feature: int = 100, p_dropout: float = 0.3,
               node_n: int = 55):
    """
    Construct `GCN_muti_att` as released with WLASL: an input graph
    convolution, `num_stage` residual GC_Blocks, a mean over the 55 keypoints
    and a linear classifier, matching the published checkpoints key for key.
    """
    torch, nn, F = _build_modules()
    GraphConvolution = make_GraphConvolution(node_n)

    class GC_Block(nn.Module):
        """A residual graph-conv block: two GraphConvolution layers with
        batch-norm, Tanh and dropout, added back to the input (residual)."""
        def __init__(self, in_features, p_drop):
            super().__init__()
            self.gc1 = GraphConvolution(in_features, in_features)
            self.bn1 = nn.BatchNorm1d(node_n * in_features)
            self.gc2 = GraphConvolution(in_features, in_features)
            self.bn2 = nn.BatchNorm1d(node_n * in_features)
            self.do = nn.Dropout(p_drop)
            self.act_f = nn.Tanh()

        def forward(self, x):
            y = self.gc1(x)
            b, n, f = y.shape
            y = self.bn1(y.view(b, -1)).view(b, n, f)
            y = self.act_f(y); y = self.do(y)
            y = self.gc2(y)
            b, n, f = y.shape
            y = self.bn2(y.view(b, -1)).view(b, n, f)
            y = self.act_f(y); y = self.do(y)
            return y + x

    class GCN_muti_att(nn.Module):
        """The WLASL TGCN (Li et al., 2020). Input is (B, 55 keypoints,
        100 features): each keypoint's x and y over 50 frames."""
        def __init__(self):
            super().__init__()
            self.num_stage = num_stage
            self.gc1 = GraphConvolution(input_feature, hidden_feature)
            self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)
            self.gcbs = nn.ModuleList([
                GC_Block(hidden_feature, p_dropout) for _ in range(num_stage)
            ])
            self.do = nn.Dropout(p_dropout)
            self.act_f = nn.Tanh()
            self.fc_out = nn.Linear(hidden_feature, num_class)

        def forward(self, x):
            y = self.gc1(x)
            b, n, f = y.shape
            y = self.bn1(y.view(b, -1)).view(b, n, f)
            y = self.act_f(y); y = self.do(y)
            for gcb in self.gcbs:
                y = gcb(y)
            return self.fc_out(torch.mean(y, dim=1))

    return GCN_muti_att()


# ---------------------------------------------------------------------------
# MediaPipe → 55-keypoint extractor
# ---------------------------------------------------------------------------

# How the 55 keypoints are laid out, their coordinate range and the class
# order. The checkpoints were trained on OpenPose keypoints and the release
# does not state these, so they were settled on 14 WLASL clips outside the
# test split (scripts/report_experiments/sign_wlasl100.py --select). Settings
# saved beside a checkpoint override them.
KEYPOINT_LAYOUT = "openpose_swapped"
KEYPOINT_COORDS = "signed"
CLASS_ORDER = "alphabetical"

# "legacy": the layout this file used before the checkpoint was checked.
POSE_INDICES_55 = [
    0,   # nose
    2, 5,    # eyes (inner)
    7, 8,    # ears
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
]
assert len(POSE_INDICES_55) == 13

# "openpose": the 13 BODY_25 points WLASL keeps, in OpenPose order (nose, neck,
# right arm, left arm, mid-hip, eyes, ears), from MediaPipe pose indices; a
# pair is the midpoint of two landmarks. Left/right are the signer's own.
OPENPOSE_POSE_13 = [0, (11, 12), 12, 14, 16, 11, 13, 15, (23, 24), 5, 2, 8, 7]


def extract_55_keypoints(holistic_result, layout: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Returns a (55, 2) array of (x, y) coordinates in [0, 1] for one frame,
    or None if no usable pose/hand information was detected. Missing points
    stay at 0, as OpenPose leaves them.

    MediaPipe names hands as if the image were mirrored, so on camera or
    video frames its "Right" hand is the signer's left. "openpose" puts that
    hand first, as OpenPose's hand_left comes first; "openpose_swapped" puts
    it second.
    """
    if holistic_result is None:
        return None
    layout = layout or KEYPOINT_LAYOUT

    pts = np.zeros((55, 2), dtype=np.float32)
    have_any = False

    pose_lms = getattr(holistic_result, "pose_landmarks", None)
    if pose_lms:
        have_any = True
        lm = pose_lms.landmark
        if layout == "legacy":
            for i, src in enumerate(POSE_INDICES_55):
                pts[i] = (lm[src].x, lm[src].y)
        else:
            for i, src in enumerate(OPENPOSE_POSE_13):
                if isinstance(src, tuple):
                    a, b = lm[src[0]], lm[src[1]]
                    pts[i] = ((a.x + b.x) / 2, (a.y + b.y) / 2)
                else:
                    pts[i] = (lm[src].x, lm[src].y)

    rh = getattr(holistic_result, "right_hand_landmarks", None)
    lh = getattr(holistic_result, "left_hand_landmarks", None)
    first, second = (lh, rh) if layout == "openpose_swapped" else (rh, lh)
    for offset, hand in ((13, first), (34, second)):
        if hand:
            have_any = True
            for i, p in enumerate(hand.landmark):
                pts[offset + i] = (p.x, p.y)

    return pts if have_any else None


def to_model_coords(keypoint_seq: List[np.ndarray], coords: Optional[str] = None) -> List[np.ndarray]:
    """[0, 1] image coordinates as the model expects them: unchanged ("unit"),
    mapped to [-1, 1] as WLASL maps OpenPose pixels ("signed"), or centred on
    the signer's neck and scaled by shoulder width over the clip ("body"), with
    undetected points left at 0."""
    coords = coords or KEYPOINT_COORDS
    if coords == "signed":
        return [2.0 * p - 1.0 for p in keypoint_seq]
    if coords == "body" and keypoint_seq:
        arr = np.stack(keypoint_seq).astype(np.float32)            # (T, 55, 2), layout "openpose"
        present = np.any(arr != 0, axis=2)
        with_pose = present[:, 1] & present[:, 2] & present[:, 5]  # neck and both shoulders
        if with_pose.any():
            centre = arr[with_pose, 1].mean(axis=0)
            scale = float(np.linalg.norm(arr[with_pose, 2] - arr[with_pose, 5], axis=1).mean())
        else:
            centre, scale = np.array([0.5, 0.5], np.float32), 0.25
        out = (arr - centre) / (2.0 * max(scale, 1e-3))
        out[~present] = 0.0
        return list(out.astype(np.float32))
    return list(keypoint_seq)


def _resample_frames(keypoint_seq: List[np.ndarray], target_frames: int = 50
                     ) -> np.ndarray:
    """
    The TGCN expects exactly `num_samples=50` frames. Linearly resample the
    input (longer or shorter) to 50 frames.

    Returns a (55, target_frames * 2) array, flattened (x, y) over time per
    keypoint — matching the model's expected input shape.
    """
    n = len(keypoint_seq)
    if n == 0:
        return np.zeros((55, target_frames * 2), dtype=np.float32)

    # Resample to exactly target_frames
    if n != target_frames:
        idx = np.linspace(0, n - 1, target_frames)
        idx = np.clip(idx.astype(int), 0, n - 1)
        seq = [keypoint_seq[i] for i in idx]
    else:
        seq = keypoint_seq

    arr = np.stack(seq, axis=0)               # (T=50, 55, 2)
    arr = arr.transpose(1, 0, 2)              # (55, T=50, 2)
    arr = arr.reshape(55, -1)                 # (55, T*2 = 100)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class TGCNSignRecognizer:
    """
    Lazy-loaded WLASL TGCN classifier. Use:

        rec = TGCNSignRecognizer(variant="asl100")
        rec.load()                         # downloads from HF on first call
        label, score = rec.predict(keypoint_seq)

    `keypoint_seq` is a list of (55, 2) numpy arrays from extract_55_keypoints().
    """

    VARIANT_CONFIG = {
        "asl100":  dict(num_class=100,  hidden=64,  stages=20),
        "asl300":  dict(num_class=300,  hidden=256, stages=24),
        "asl1000": dict(num_class=1000, hidden=256, stages=24),
        "asl2000": dict(num_class=2000, hidden=256, stages=24),
    }

    # Repos to try in order. The user can override with ACCESSIBILITY_TGCN_REPO.
    DEFAULT_REPOS = ["sharonn18/tgcn-wlasl"]
    LOCAL_DIR = Path(__file__).resolve().parent.parent / "data" / "tgcn"

    def __init__(self, variant: str = "asl100") -> None:
        if variant not in self.VARIANT_CONFIG:
            raise ValueError(f"Unknown TGCN variant {variant!r}")
        self.variant = variant
        self.model = None
        self.vocab: List[str] = []
        self.config: dict = {}
        self.layout = KEYPOINT_LAYOUT
        self.coords = KEYPOINT_COORDS
        self.threshold = 0.30          # lowest confidence at which the cascade shows a TGCN sign
        self._loaded = False
        self._load_failed = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Attempt to load the pre-trained checkpoint. Returns True on success.

        Tries — in order:
            1. Local override path  ACCESSIBILITY_TGCN_LOCAL_PATH (a .bin file)
            2. HuggingFace repo from ACCESSIBILITY_TGCN_REPO env var
            3. The DEFAULT_REPOS list above
        """
        if self._loaded:
            return True
        if self._load_failed:
            return False

        try:
            import torch  # noqa: F401
        except ImportError:
            log.info("TGCN disabled — torch not installed")
            self._load_failed = True
            return False

        cfg = self.VARIANT_CONFIG[self.variant]
        model = build_TGCN(
            num_class=cfg["num_class"],
            hidden_feature=cfg["hidden"],
            num_stage=cfg["stages"],
        )

        ckpt_path = self._resolve_checkpoint()
        if not ckpt_path:
            self._load_failed = True
            return False

        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            # Some checkpoints prefix with "module." — strip it
            state = {k.replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                # Untrained layers would predict at random; refuse the checkpoint.
                log.warning("TGCN checkpoint does not fit the network (%d missing, %d unexpected keys, "
                            "first missing: %s); TGCN disabled", len(missing), len(unexpected), missing[:3])
                self._load_failed = True
                return False
            model.eval()
            self.model = model

            # Settings and vocabulary saved alongside the checkpoint
            self.config = self._load_config(ckpt_path)
            self.layout = self.config.get("layout", KEYPOINT_LAYOUT)
            self.coords = self.config.get("coords", KEYPOINT_COORDS)
            self.threshold = float(self.config.get("threshold", self.threshold))
            self.vocab = self._load_vocab(ckpt_path)
            log.info("TGCN loaded: variant=%s, classes=%d, vocab=%d entries",
                     self.variant, cfg["num_class"], len(self.vocab))
            self._loaded = True
            return True
        except Exception as e:
            log.warning("TGCN checkpoint failed to load: %s", e)
            self._load_failed = True
            return False

    def _resolve_checkpoint(self) -> Optional[str]:
        """Returns a local .bin path or None."""
        # 1. Explicit local override
        local = os.environ.get("ACCESSIBILITY_TGCN_LOCAL_PATH", "").strip()
        if local and os.path.exists(local):
            log.info("TGCN: using local checkpoint %s", local)
            return local
        # Weights trained on MediaPipe keypoints (sign_train_tgcn.py) come first.
        for sub in (f"{self.variant}_mediapipe", self.variant):
            bundled = self.LOCAL_DIR / sub / "pytorch_model.bin"
            if bundled.exists():
                log.info("TGCN: using local checkpoint %s", bundled)
                return str(bundled)

        # 2. HuggingFace download
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            log.info("TGCN: huggingface_hub not installed — pip install huggingface_hub")
            return None

        repos = []
        env_repo = os.environ.get("ACCESSIBILITY_TGCN_REPO", "").strip()
        if env_repo:
            repos.append(env_repo)
        repos.extend(self.DEFAULT_REPOS)

        for repo in repos:
            try:
                # Most repos use 'checkpoints/<variant>/pytorch_model.bin'
                for fname in (
                    f"checkpoints/{self.variant}/pytorch_model.bin",
                    f"{self.variant}/pytorch_model.bin",
                    f"{self.variant}.bin",
                ):
                    try:
                        path = hf_hub_download(repo_id=repo, filename=fname)
                        log.info("TGCN: downloaded %s/%s", repo, fname)
                        return path
                    except Exception:
                        continue
            except Exception as e:
                log.debug("TGCN: repo %s skipped (%s)", repo, e)

        log.info(
            "TGCN: no checkpoint found. Set ACCESSIBILITY_TGCN_REPO to a "
            "HuggingFace repo containing checkpoints/%s/pytorch_model.bin, "
            "or set ACCESSIBILITY_TGCN_LOCAL_PATH to a local .bin file.",
            self.variant,
        )
        return None

    def _load_config(self, ckpt_path: str) -> dict:
        """Settings saved beside a checkpoint trained by sign_train_tgcn.py
        (coordinates, class order, display threshold); empty for others."""
        f = Path(ckpt_path).parent / "config.json"
        try:
            return json.loads(f.read_text()) if f.exists() else {}
        except Exception as e:
            log.warning("TGCN: unreadable %s (%s); using defaults", f, e)
            return {}

    def _load_vocab(self, ckpt_path: str) -> List[str]:
        """Look for a sibling vocabulary file: classes.txt or vocab.json."""
        ckpt_dir = Path(ckpt_path).parent
        for fname in ("classes.txt", "vocab.txt", "labels.txt"):
            f = ckpt_dir / fname
            if f.exists():
                return [line.strip() for line in f.read_text().splitlines() if line.strip()]
        # Fall back to the WLASL-100 glosses for asl100, in the selected order
        if self.variant == "asl100":
            order = list(WLASL100_DEFAULT_ORDER)
            return sorted(order) if self.config.get("class_order", CLASS_ORDER) == "alphabetical" else order
        # Otherwise generate placeholder labels
        n = self.VARIANT_CONFIG[self.variant]["num_class"]
        return [f"wlasl_{i:04d}" for i in range(n)]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True only if a TGCN checkpoint was successfully loaded, so callers
        can fall back to the geometric tiers when it is not."""
        return self._loaded and self.model is not None

    def predict(
        self, keypoint_seq: List[np.ndarray], top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Run the model on a list of per-frame (55, 2) keypoint arrays.
        Returns the top-k (label, confidence) tuples sorted desc.
        """
        if not self.is_available() or len(keypoint_seq) < 8:
            return []

        try:
            import torch
            x = _resample_frames(to_model_coords(keypoint_seq, self.coords), target_frames=50)   # (55, 100)
            x = torch.from_numpy(x).unsqueeze(0)                    # (1, 55, 100)
            with torch.no_grad():
                logits = self.model(x).squeeze(0)
                probs = torch.softmax(logits, dim=-1).numpy()
            top_idx = np.argsort(probs)[-top_k:][::-1]
            out: List[Tuple[str, float]] = []
            for idx in top_idx:
                label = self.vocab[idx] if idx < len(self.vocab) else f"class_{idx}"
                out.append((label, float(probs[idx])))
            return out
        except Exception as e:
            log.warning("TGCN inference failed: %s", e)
            return []
