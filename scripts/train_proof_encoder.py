"""Train the Phase 1 typeface-invariant proof encoder on synthetic text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _read_lines(path: Path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _samples(texts, fonts, processes, vertical_augmentation=False):
    import numpy as np

    from tofu.layers import proof, proof_encoder

    rows = []
    for label, text in enumerate(texts):
        for font in fonts:
            clean = proof.render(text, font)
            if clean is None:
                continue
            rows.append((proof_encoder.prepare(clean), label, text, font, "clean"))
            if vertical_augmentation:
                rows.append((proof_encoder.prepare(np.rot90(clean, -1)), label, text, font, "vertical"))
            for process in processes:
                damaged = proof.degrade(clean, process, proof.MODERATE, seed=(text, font, process))
                rows.append((proof_encoder.prepare(damaged), label, text, font, process))
                if vertical_augmentation:
                    rows.append((
                        proof_encoder.prepare(np.rot90(damaged, -1)),
                        label,
                        text,
                        font,
                        f"vertical-{process}",
                    ))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--train-font", action="append", required=True)
    parser.add_argument("--holdout-font", action="append", required=True)
    parser.add_argument("--ground-truth", action="append", default=[])
    parser.add_argument("--ground-truth-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--vertical-augmentation", action="store_true")
    parser.add_argument("--length-weight", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=12)
    args = parser.parse_args()

    import numpy as np
    import torch

    from tofu.layers import proof_encoder

    torch.manual_seed(args.seed)
    texts = _read_lines(args.texts)
    ground_truth = {text.strip() for text in args.ground_truth if text.strip()}
    if args.ground_truth_file:
        ground_truth.update(_read_lines(args.ground_truth_file))
    reserved = ground_truth | {
        char for text in ground_truth for char in text if not char.isspace()
    }
    available = [text for text in texts if text not in reserved]
    train_texts = [text for text in available if proof_encoder.split_for(text) == "train"]
    holdout_texts = sorted(reserved | {
        text for text in available if proof_encoder.split_for(text) == "holdout"
    })
    train_rows = _samples(
        train_texts,
        args.train_font,
        ("blur", "resample", "abrasion"),
        args.vertical_augmentation,
    )
    holdout_rows = _samples(
        holdout_texts, args.holdout_font, (), args.vertical_augmentation
    )
    if not train_rows or not holdout_rows:
        raise SystemExit("both train and held-out samples are required")

    multitask = args.length_weight > 0
    model = (
        proof_encoder.build_multitask_model(max_length=args.max_length)
        if multitask else proof_encoder.build_model()
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    images = torch.from_numpy(np.stack([row[0] for row in train_rows])).unsqueeze(1)
    labels = torch.tensor([row[1] for row in train_rows], dtype=torch.long)
    lengths = torch.tensor(
        [min(len(row[2]), args.max_length) - 1 for row in train_rows],
        dtype=torch.long,
    )
    model.train()
    losses = []
    for _ in range(args.epochs):
        optimizer.zero_grad()
        if multitask:
            embeddings, length_logits = model.forward_with_length(images)
            contrastive = proof_encoder.supervised_contrastive_loss(embeddings, labels)
            length_loss = torch.nn.functional.cross_entropy(length_logits, lengths)
            loss = contrastive + args.length_weight * length_loss
        else:
            loss = proof_encoder.supervised_contrastive_loss(model(images), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    holdout_images = torch.from_numpy(np.stack([row[0] for row in holdout_rows])).unsqueeze(1)
    with torch.no_grad():
        if multitask:
            vectors, holdout_length_logits = model.forward_with_length(holdout_images)
        else:
            vectors = model(holdout_images)
        similarities = vectors @ vectors.T
    correct = 0
    for index, row in enumerate(holdout_rows):
        scores = similarities[index].clone()
        scores[index] = float("-inf")
        prediction = holdout_rows[int(scores.argmax())][2]
        correct += prediction == row[2]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    holdout_lengths = [min(len(row[2]), args.max_length) for row in holdout_rows]
    length_correct = None
    if multitask:
        predictions = holdout_length_logits.argmax(dim=1).add(1).tolist()
        length_correct = sum(
            predicted == truth for predicted, truth in zip(predictions, holdout_lengths)
        )
    torch.save({
        "schema": 2 if multitask else 1,
        "architecture": "multitask" if multitask else "embedding",
        "model": model.state_dict(),
        "embedding_dim": proof_encoder.EMBEDDING_DIM,
        "max_length": args.max_length if multitask else None,
    }, args.out)
    report = {
        "schema": 2 if multitask else 1,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_strings": len(train_texts),
        "holdout_strings": len(holdout_texts),
        "reserved_ground_truth": sorted(reserved),
        "train_fonts": args.train_font,
        "holdout_fonts": args.holdout_font,
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "holdout_cross_face_top1": correct / len(holdout_rows),
        "vertical_augmentation": args.vertical_augmentation,
        "length_weight": args.length_weight,
        "holdout_length_top1": (
            length_correct / len(holdout_rows) if length_correct is not None else None
        ),
        "checkpoint": str(args.out),
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
