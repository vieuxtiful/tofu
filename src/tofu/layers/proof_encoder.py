"""Typeface-invariant embedding support for Phase 1 glyph retrieval.

The production package keeps torch optional.  Image preparation and split
construction therefore live at module scope, while the trainable model is
created only by :func:`build_model` after a caller explicitly asks for it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

CANVAS_HEIGHT = 48
CANVAS_WIDTH = 192
EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 12


@dataclass(frozen=True)
class TypefacePair:
    text: str
    left_font: str
    right_font: str
    split: str


def stable_bucket(value: str, buckets: int = 10) -> int:
    if buckets < 2:
        raise ValueError("buckets must be at least 2")
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def split_for(text: str, held_out_buckets: Sequence[int] = (8, 9)) -> str:
    return "holdout" if stable_bucket(text) in set(held_out_buckets) else "train"


def cross_face_pairs(
    texts: Iterable[str],
    train_fonts: Sequence[str],
    holdout_fonts: Sequence[str],
) -> list[TypefacePair]:
    pairs: list[TypefacePair] = []
    for text in sorted({value.strip() for value in texts if value.strip()}):
        split = split_for(text)
        fonts = holdout_fonts if split == "holdout" else train_fonts
        for left_index, left in enumerate(fonts):
            for right in fonts[left_index + 1 :]:
                pairs.append(TypefacePair(text, left, right, split))
    return pairs


def prepare(image: Any, height: int = CANVAS_HEIGHT, width: int = CANVAS_WIDTH):
    import cv2
    import numpy as np

    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("proof image must be a non-empty 2D array")
    array = np.clip(array, 0.0, 1.0)
    scale = min(width / array.shape[1], height / array.shape[0])
    resized_width = max(1, int(round(array.shape[1] * scale)))
    resized_height = max(1, int(round(array.shape[0] * scale)))
    resized = cv2.resize(array, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width), dtype=np.float32)
    top = (height - resized_height) // 2
    left = (width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def canonical_reading_direction(image: Any, *, vertical: bool = False):
    import numpy as np

    array = np.asarray(image)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("proof image must be a non-empty 2D array")
    return np.rot90(array, 1).copy() if vertical else array.copy()


def build_model(embedding_dim: int = EMBEDDING_DIM):
    import torch.nn as nn

    class ProofEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 24, 5, stride=2, padding=2),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Conv2d(24, 48, 3, stride=2, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.Conv2d(48, 96, 3, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 8)),
            )
            self.project = nn.Linear(96 * 8, embedding_dim)

        def forward(self, image):
            import torch.nn.functional as functional

            features = self.features(image).flatten(1)
            return functional.normalize(self.project(features), dim=1)

    return ProofEncoder()


def build_multitask_model(
    embedding_dim: int = EMBEDDING_DIM,
    max_length: int = MAX_SEQUENCE_LENGTH,
):
    import torch.nn as nn

    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    class MultitaskProofEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 24, 5, stride=2, padding=2),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Conv2d(24, 48, 3, stride=2, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.Conv2d(48, 96, 3, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 8)),
            )
            self.project = nn.Linear(96 * 8, embedding_dim)
            self.length_project = nn.Linear(96 * 8, max_length)

        def forward_with_length(self, image):
            import torch.nn.functional as functional

            features = self.features(image).flatten(1)
            embedding = functional.normalize(self.project(features), dim=1)
            return embedding, self.length_project(features)

        def forward(self, image):
            return self.forward_with_length(image)[0]

    return MultitaskProofEncoder()


def build_checkpoint_model(checkpoint: dict[str, Any]):
    if checkpoint.get("architecture") == "multitask":
        return build_multitask_model(
            checkpoint.get("embedding_dim", EMBEDDING_DIM),
            checkpoint.get("max_length", MAX_SEQUENCE_LENGTH),
        )
    return build_model(checkpoint.get("embedding_dim", EMBEDDING_DIM))


def batch_hard_loss(embeddings, labels, margin: float = 0.2):
    import torch

    distances = 1.0 - embeddings @ embeddings.T
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = ~labels[:, None].eq(labels[None, :])
    positive = distances.masked_fill(~same, float("-inf")).max(dim=1).values
    negative = distances.masked_fill(~different, float("inf")).min(dim=1).values
    valid = torch.isfinite(positive) & torch.isfinite(negative)
    if not valid.any():
        return embeddings.sum() * 0.0
    return torch.relu(positive[valid] - negative[valid] + margin).mean()


def supervised_contrastive_loss(embeddings, labels, temperature: float = 0.1):
    import torch

    logits = embeddings @ embeddings.T / temperature
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(identity, 0.0)
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    counts = positives.sum(dim=1)
    valid = counts > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    mean_positive = (log_probability * positives).sum(dim=1) / counts.clamp_min(1)
    return -mean_positive[valid].mean()


def rank(encoder, observed: Any, candidates: Sequence[tuple[str, Any]]):
    import torch

    if not candidates:
        return None
    encoder.eval()
    images = [prepare(observed), *(prepare(image) for _, image in candidates)]
    tensor = torch.from_numpy(__import__("numpy").stack(images)).unsqueeze(1)
    with torch.no_grad():
        vectors = encoder(tensor)
        scores = vectors[1:] @ vectors[0]
    order = torch.argsort(scores, descending=True).tolist()
    ranked = [
        {"text": candidates[index][0], "support": round(float(scores[index]), 6)}
        for index in order
    ]
    return {
        "best": ranked[0]["text"],
        "support": ranked[0]["support"],
        "margin": (
            round(ranked[0]["support"] - ranked[1]["support"], 6)
            if len(ranked) > 1 else None
        ),
        "candidates": ranked,
    }
