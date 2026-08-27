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
The wrapper class `TGCNSignRecognizer` tries to download the requested variant
from HuggingFace on first use. The HuggingFace repo is configurable via the
env var `ACCESSIBILITY_TGCN_REPO`. If download fails (offline, missing repo,
no torch) the recognizer silently disables itself — the system keeps working
via MediaPipe + geometric rules.

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
import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("accessibility.tgcn")


# Default WLASL-100 vocabulary order — the ORIGINAL order that the asl100
# checkpoint was trained against. If a checkpoint ships its own class
# mapping, that takes priority.
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
    "son", "tell", "thursday", "wait", "yellow",
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
    Construct the full `GCN_muti_att` network — multi-stage residual GCN
    with a multi-head attention block in the middle.
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
        """The full WLASL TGCN (Li et al., 2020): an input graph-conv layer,
        `num_stage` residual GC_Blocks, an 8-head self-attention block over
        the keypoint dimension, a final graph-conv, and a linear classifier
        over the sign vocabulary. Input is (B, 55 keypoints, 100 features)."""
        def __init__(self):
            super().__init__()
            self.num_stage = num_stage
            self.gc1 = GraphConvolution(input_feature, hidden_feature)
            self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)
            self.gcbs = nn.ModuleList([
                GC_Block(hidden_feature, p_dropout) for _ in range(num_stage)
            ])
            self.attention = nn.MultiheadAttention(hidden_feature, num_heads=8)
            self.gc7 = GraphConvolution(hidden_feature, hidden_feature)
            self.do = nn.Dropout(p_dropout)
            self.act_f = nn.Tanh()
            self.fc = nn.Linear(node_n * hidden_feature, num_class)

        def forward(self, x):
            # x: (B, 55, 100)
            y = self.gc1(x)
            b, n, f = y.shape
            y = self.bn1(y.view(b, -1)).view(b, n, f)
            y = self.act_f(y); y = self.do(y)
            for gcb in self.gcbs:
                y = gcb(y)
            y_perm = y.permute(1, 0, 2)               # (55, B, hidden)
            attn_out, _ = self.attention(y_perm, y_perm, y_perm)
            y = attn_out.permute(1, 0, 2)
            y = self.gc7(y)
            b, n, f = y.shape
            y = self.act_f(y); y = self.do(y)
            return self.fc(y.view(b, -1))

    return GCN_muti_att()


# ---------------------------------------------------------------------------
# MediaPipe Holistic → 55-keypoint extractor
# ---------------------------------------------------------------------------

# Map the 55 keypoints the TGCN expects.
#   13 upper-body landmarks from MediaPipe pose (indices into pose[0..32])
#   21 right-hand landmarks
#   21 left-hand landmarks
#   = 55 total
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


def extract_55_keypoints(holistic_result) -> Optional[np.ndarray]:
    """
    Returns a (55, 2) array of (x, y) normalized coordinates for one frame,
    or None if no usable pose/hand information was detected.
    """
    if holistic_result is None:
        return None

    pts = np.zeros((55, 2), dtype=np.float32)
    have_any = False

    # ── Pose (13) ────────────────────────────────────────────────────────
    pose_lms = getattr(holistic_result, "pose_landmarks", None)
    if pose_lms:
        have_any = True
        for i, src in enumerate(POSE_INDICES_55):
            lm = pose_lms.landmark[src]
            pts[i, 0] = lm.x
            pts[i, 1] = lm.y

    # ── Right hand (21) ──────────────────────────────────────────────────
    rh = getattr(holistic_result, "right_hand_landmarks", None)
    if rh:
        have_any = True
        for i, lm in enumerate(rh.landmark):
            pts[13 + i, 0] = lm.x
            pts[13 + i, 1] = lm.y

    # ── Left hand (21) ───────────────────────────────────────────────────
    lh = getattr(holistic_result, "left_hand_landmarks", None)
    if lh:
        have_any = True
        for i, lm in enumerate(lh.landmark):
            pts[34 + i, 0] = lm.x
            pts[34 + i, 1] = lm.y

    return pts if have_any else None


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
    DEFAULT_REPOS = [
        "kasrahabib/tgcn-wlasl",
        "zhengshu/tgcn-wlasl",
        "asl-research/tgcn-wlasl",
    ]

    def __init__(self, variant: str = "asl100") -> None:
        if variant not in self.VARIANT_CONFIG:
            raise ValueError(f"Unknown TGCN variant {variant!r}")
        self.variant = variant
        self.model = None
        self.vocab: List[str] = []
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
            if missing:
                log.warning("TGCN load: %d missing keys (first: %s)",
                            len(missing), missing[:3])
            if unexpected:
                log.warning("TGCN load: %d unexpected keys", len(unexpected))
            model.eval()
            self.model = model

            # Try to load a vocab file alongside the checkpoint
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

    def _load_vocab(self, ckpt_path: str) -> List[str]:
        """Look for a sibling vocabulary file: classes.txt or vocab.json."""
        ckpt_dir = Path(ckpt_path).parent
        for fname in ("classes.txt", "vocab.txt", "labels.txt"):
            f = ckpt_dir / fname
            if f.exists():
                return [line.strip() for line in f.read_text().splitlines() if line.strip()]
        # Fall back to the WLASL-100 default order for asl100
        if self.variant == "asl100":
            return list(WLASL100_DEFAULT_ORDER)
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
            x = _resample_frames(keypoint_seq, target_frames=50)   # (55, 100)
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
