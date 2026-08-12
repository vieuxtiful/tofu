"""Detached geometric evidence for Proof sequence length."""

from __future__ import annotations

from typing import Any

MAX_SEQUENCE_LENGTH = 12


def length_class(text: str, max_length: int = MAX_SEQUENCE_LENGTH) -> int:
    if not text:
        raise ValueError("text must not be empty")
    return min(len(text), max_length) - 1


def build_model(max_length: int = MAX_SEQUENCE_LENGTH):
    import torch.nn as nn

    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    return nn.Sequential(
        nn.Conv2d(1, 16, 5, stride=2, padding=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 48, 3, stride=2, padding=1),
        nn.BatchNorm2d(48),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 8)),
        nn.Flatten(),
        nn.Linear(48 * 2 * 8, max_length),
    )


def predict(model, image: Any) -> dict[str, float | int]:
    import torch

    from tofu.layers import proof_encoder

    tensor = torch.from_numpy(proof_encoder.prepare(image)).unsqueeze(0).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(tensor)[0], dim=0)
    return {
        "length": int(probability.argmax()) + 1,
        "support": round(float(probability.max()), 6),
    }
