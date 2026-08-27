"""
sign_language_model.py — Torch model helpers for WLASL-trained sign recognition.

This keeps training/inference checkpoint format explicit instead of relying
on pickled model objects with ad-hoc attributes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn


DEFAULT_INPUT_SIZE = 543
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.3


class LandmarkBiLSTM(nn.Module):
    def __init__(
        self,
        *,
        input_size: int = DEFAULT_INPUT_SIZE,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


def build_checkpoint(
    model: LandmarkBiLSTM,
    *,
    vocab: Sequence[str],
    clip_frames: int,
    extra_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    meta = {
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "clip_frames": clip_frames,
    }
    if extra_meta:
        meta.update(extra_meta)
    return {
        "model_type": "landmark_bilstm",
        "state_dict": model.state_dict(),
        "vocab": list(vocab),
        "meta": meta,
    }


def load_checkpoint(
    path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> Tuple[nn.Module, List[str], Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location)

    if isinstance(checkpoint, nn.Module):
        vocab = list(getattr(checkpoint, "vocab", []))
        meta = {
            "clip_frames": getattr(checkpoint, "clip_frames", 30),
            "legacy_pickle": True,
        }
        checkpoint.eval()
        return checkpoint, vocab, meta

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported sign-model checkpoint type: {type(checkpoint)!r}")

    if "state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing 'state_dict'")

    vocab = list(checkpoint.get("vocab") or [])
    meta = dict(checkpoint.get("meta") or {})
    model_type = checkpoint.get("model_type", "landmark_bilstm")
    if model_type != "landmark_bilstm":
        raise ValueError(f"Unsupported sign-model type: {model_type}")

    model = LandmarkBiLSTM(
        input_size=int(meta.get("input_size", DEFAULT_INPUT_SIZE)),
        hidden_size=int(meta.get("hidden_size", DEFAULT_HIDDEN_SIZE)),
        num_layers=int(meta.get("num_layers", DEFAULT_NUM_LAYERS)),
        dropout=float(meta.get("dropout", DEFAULT_DROPOUT)),
        num_classes=len(vocab),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocab, meta
