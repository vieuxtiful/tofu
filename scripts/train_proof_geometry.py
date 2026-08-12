"""Train detached sequence-geometry evidence without touching identity weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _samples(texts, fonts, max_length):
    import numpy as np

    from tofu.layers import proof, proof_encoder, proof_geometry

    rows = []
    for text in texts:
        target = proof_geometry.length_class(text, max_length)
        for font in fonts:
            clean = proof.render(text, font)
            if clean is None:
                continue
            variants = [clean]
            for process in ("blur", "resample", "abrasion", "occlusion"):
                variants.append(proof.degrade(clean, process, proof.MODERATE, (text, font, process)))
            padded = np.pad(clean, ((5, 11), (13, 7)))
            variants.append(padded)
            for image in variants:
                rows.append((proof_encoder.prepare(image), target, text, "horizontal"))
                rows.append((proof_encoder.prepare(np.rot90(image, -1)), target, text, "vertical"))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--train-font", action="append", required=True)
    parser.add_argument("--holdout-font", action="append", required=True)
    parser.add_argument("--ground-truth-file", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import torch

    from tofu.layers import proof_encoder, proof_geometry

    torch.manual_seed(args.seed)
    reserved = set(_lines(args.ground_truth_file)) if args.ground_truth_file else set()
    reserved |= {char for text in reserved for char in text if not char.isspace()}
    texts = _lines(args.texts)
    train_texts = [
        text for text in texts
        if text not in reserved and proof_encoder.split_for(text) == "train"
    ]
    holdout_texts = sorted(reserved | {
        text for text in texts
        if text not in reserved and proof_encoder.split_for(text) == "holdout"
    })
    train_rows = _samples(train_texts, args.train_font, args.max_length)
    holdout_rows = _samples(holdout_texts, args.holdout_font, args.max_length)
    model = proof_geometry.build_model(args.max_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    images = torch.from_numpy(np.stack([row[0] for row in train_rows])).unsqueeze(1)
    labels = torch.tensor([row[1] for row in train_rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=args.max_length).float()
    weights = torch.where(counts > 0, len(labels) / counts.clamp_min(1), 0.0)
    losses = []
    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), labels, weight=weights)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    holdout_images = torch.from_numpy(np.stack([row[0] for row in holdout_rows])).unsqueeze(1)
    holdout_labels = torch.tensor([row[1] for row in holdout_rows], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        predictions = model(holdout_images).argmax(dim=1)
    correct = int((predictions == holdout_labels).sum())
    orientation = {}
    for value in ("horizontal", "vertical"):
        indices = [index for index, row in enumerate(holdout_rows) if row[3] == value]
        orientation[value] = {
            "scored": len(indices),
            "top1": sum(int(predictions[index]) == int(holdout_labels[index]) for index in indices),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": 1,
        "kind": "detached_geometry",
        "model": model.state_dict(),
        "max_length": args.max_length,
    }, args.out)
    report = {
        "schema": 1,
        "kind": "detached_geometry_training",
        "seed": args.seed,
        "epochs": args.epochs,
        "train_strings": len(train_texts),
        "holdout_strings": len(holdout_texts),
        "train_samples": len(train_rows),
        "holdout_samples": len(holdout_rows),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "holdout_length_top1": correct / len(holdout_rows),
        "orientation": orientation,
        "identity_checkpoint_loaded": False,
        "checkpoint": str(args.out),
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
